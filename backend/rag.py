"""Hybrid Full-PDF RAG pipeline supporting Offline (Ollama) and Online (Gemini API) modes."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Sequence

from backend.config import Settings
from backend.gemini_client import GeminiClient
from backend.ollama_client import OllamaClient
from backend.parser import chunk_pages, extract_pages
from backend.vector_store import DocumentRecord, SearchResult, VectorStore


ProviderName = Literal["ollama", "gemini"]

# Character budgets to keep Full-PDF reading safe for both local CPU and cloud 1M context.
OFFLINE_DIRECT_CHAR_LIMIT = 24_000
OFFLINE_SECTION_CHAR_LIMIT = 18_000
ONLINE_DIRECT_CHAR_LIMIT = 2_400_000

SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer the user's question using ONLY the provided document contents. "
    "Always cite sources inline using the format [filename | Page X]. "
    "If the provided documents do not contain enough information to answer, "
    "say clearly what is missing instead of guessing."
)

MAP_SYSTEM_PROMPT = (
    "You are reading one section of a larger document to help answer a user's question. "
    "Extract all relevant facts, definitions, formulas, and details from this section "
    "with citations in [filename | Page X] format. If nothing in this section is relevant, "
    "reply briefly with: NO_RELEVANT_INFO."
)


@dataclass(frozen=True)
class IngestResult:
    """Result returned after indexing (or deduplicating) an uploaded document."""

    document: DocumentRecord
    already_indexed: bool


@dataclass(frozen=True)
class ChatResponse:
    """Full non-streaming answer with supporting citations."""

    answer: str
    provider: ProviderName
    model: str
    sources: List[SearchResult]


def build_context_block(sources: Sequence[SearchResult]) -> str:
    """Format document chunks into a labeled full-document context block."""
    parts: List[str] = []
    for src in sources:
        parts.append(f"[{src.filename} | Page {src.page_number}]\n{src.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    question: str,
    context_text: str,
    history: Optional[List[Dict[str, str]]] = None,
    context_label: str = "Full document contents",
) -> List[Dict[str, str]]:
    """Build chat messages for Ollama combining system instructions, history, and context."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for msg in history[-6:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_prompt = (
        f"{context_label}:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and cite [filename | Page X] for every claim."
    )
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_gemini_input(
    question: str,
    context_text: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build a structured input string for Gemini Interactions API."""
    sections: List[str] = []
    if history:
        recent = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in history[-6:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if recent:
            sections.append("Recent conversation:\n" + "\n".join(recent))

    sections.append(f"Full document contents:\n\n{context_text}")
    sections.append(
        f"Question: {question}\n\n"
        "Answer clearly and cite [filename | Page X] for every claim."
    )
    return "\n\n".join(sections)


def _batch_chunks_by_chars(
    chunks: Sequence[SearchResult],
    max_chars: int,
) -> List[List[SearchResult]]:
    """Group sequential chunks into batches under max_chars each for Map-Reduce."""
    batches: List[List[SearchResult]] = []
    current: List[SearchResult] = []
    current_len = 0

    for chunk in chunks:
        chunk_len = len(chunk.text) + 40
        if current and current_len + chunk_len > max_chars:
            batches.append(current)
            current = [chunk]
            current_len = chunk_len
        else:
            current.append(chunk)
            current_len += chunk_len

    if current:
        batches.append(current)
    return batches


def _serialize_source(s: SearchResult) -> Dict[str, Any]:
    return {
        "doc_id": s.doc_id,
        "filename": s.filename,
        "page_number": s.page_number,
        "chunk_index": s.chunk_index,
        "score": round(s.score, 4),
        "embed_provider": s.embed_provider,
        "text": s.text,
    }


class RAGService:
    """Coordinates document ingestion and Full-PDF Offline/Online Q&A."""

    def __init__(
        self,
        store: VectorStore,
        ollama: OllamaClient,
        gemini: GeminiClient,
        settings: Settings,
        offline_direct_char_limit: int = OFFLINE_DIRECT_CHAR_LIMIT,
        offline_section_char_limit: int = OFFLINE_SECTION_CHAR_LIMIT,
    ) -> None:
        self.store = store
        self.ollama = ollama
        self.gemini = gemini
        self.settings = settings
        self.offline_direct_char_limit = offline_direct_char_limit
        self.offline_section_char_limit = offline_section_char_limit
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    def _default_embed_model(self, provider: ProviderName) -> str:
        return (
            self.settings.gemini_embed_model
            if provider == "gemini"
            else self.settings.embed_model
        )

    def _default_chat_model(self, provider: ProviderName) -> str:
        return (
            self.settings.gemini_chat_model
            if provider == "gemini"
            else self.settings.chat_model
        )

    async def _embed_texts(
        self,
        texts: List[str],
        provider: ProviderName,
        model: str,
        api_key: Optional[str] = None,
    ) -> List[List[float]]:
        if provider == "gemini":
            return await self.gemini.embed(texts=texts, model=model, api_key=api_key)
        return await self.ollama.embed(texts=texts, model=model)

    async def ingest_file(
        self,
        filename: str,
        content: bytes,
        provider: ProviderName = "ollama",
        embed_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> IngestResult:
        """Parse, chunk, embed via chosen provider, and store if not already indexed."""
        safe_name = Path(filename).name
        file_hash = hashlib.sha256(content).hexdigest()
        chosen_embed = embed_model or self._default_embed_model(provider)

        existing = self.store.get_by_hash(
            file_hash,
            embed_provider=provider,
            embed_model=chosen_embed,
        )
        if existing is not None:
            return IngestResult(document=existing, already_indexed=True)

        pages = extract_pages(safe_name, content)
        chunks = chunk_pages(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        embeddings = await self._embed_texts(
            texts=[c.text for c in chunks],
            provider=provider,
            model=chosen_embed,
            api_key=api_key,
        )

        record = self.store.add_document(
            filename=safe_name,
            file_hash=file_hash,
            page_count=len(pages),
            chunks=chunks,
            embeddings=embeddings,
            embed_provider=provider,
            embed_model=chosen_embed,
        )

        dest = self.settings.upload_dir / f"{record.doc_id}_{safe_name}"
        dest.write_bytes(content)

        return IngestResult(document=record, already_indexed=False)

    def load_full_documents(
        self,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[SearchResult]:
        """Load all chunks of selected documents in exact page/reading order."""
        return self.store.get_document_chunks(doc_ids=doc_ids)

    async def _map_reduce_offline(
        self,
        question: str,
        chunks: Sequence[SearchResult],
        model: str,
    ) -> str:
        """Read large PDFs section-by-section on Ollama and combine relevant findings."""
        batches = _batch_chunks_by_chars(chunks, self.offline_section_char_limit)
        findings: List[str] = []

        for idx, batch in enumerate(batches, start=1):
            section_text = build_context_block(batch)
            messages = [
                {"role": "system", "content": MAP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Section {idx}/{len(batches)}:\n\n{section_text}\n\n"
                        f"Question: {question}"
                    ),
                },
            ]
            note = await self.ollama.chat(messages=messages, model=model)
            cleaned = note.strip()
            if cleaned and "NO_RELEVANT_INFO" not in cleaned.upper():
                findings.append(f"Section {idx} notes:\n{cleaned}")

        if not findings:
            return (
                "Full document sections checked, but no relevant information was found "
                "for your question."
            )
        return "\n\n---\n\n".join(findings)

    async def ask(
        self,
        question: str,
        provider: ProviderName = "ollama",
        model: Optional[str] = None,
        doc_ids: Optional[Sequence[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,
    ) -> ChatResponse:
        """Read the full selected PDF(s) and generate a complete answer."""
        chosen_model = model or self._default_chat_model(provider)
        sources = self.load_full_documents(doc_ids=doc_ids)
        if not sources:
            return ChatResponse(
                answer="No indexed documents found matching your selection. Please upload a PDF or note first.",
                provider=provider,
                model=chosen_model,
                sources=[],
            )

        full_text = build_context_block(sources)

        if provider == "gemini":
            context_text = full_text[:ONLINE_DIRECT_CHAR_LIMIT]
            user_input = build_gemini_input(
                question=question,
                context_text=context_text,
                history=history,
            )
            answer = await self.gemini.chat(
                system_instruction=SYSTEM_PROMPT,
                user_prompt=user_input,
                model=chosen_model,
                api_key=api_key,
            )
        else:
            if len(full_text) <= self.offline_direct_char_limit:
                messages = build_messages(
                    question=question,
                    context_text=full_text,
                    history=history,
                )
            else:
                synthesized = await self._map_reduce_offline(
                    question=question,
                    chunks=sources,
                    model=chosen_model,
                )
                messages = build_messages(
                    question=question,
                    context_text=synthesized,
                    history=history,
                    context_label="Notes extracted across all PDF sections",
                )
            answer = await self.ollama.chat(messages=messages, model=chosen_model)

        return ChatResponse(
            answer=answer,
            provider=provider,
            model=chosen_model,
            sources=sources,
        )

    async def ask_stream(
        self,
        question: str,
        provider: ProviderName = "ollama",
        model: Optional[str] = None,
        doc_ids: Optional[Sequence[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Read the full selected PDF(s), emit sources, and stream the answer."""
        chosen_model = model or self._default_chat_model(provider)
        sources = self.load_full_documents(doc_ids=doc_ids)

        yield {
            "type": "sources",
            "provider": provider,
            "model": chosen_model,
            "sources": [_serialize_source(s) for s in sources],
        }

        if not sources:
            yield {
                "type": "token",
                "content": "No indexed documents found matching your selection. Please upload a PDF or note first.",
            }
            yield {"type": "done"}
            return

        full_text = build_context_block(sources)

        if provider == "gemini":
            context_text = full_text[:ONLINE_DIRECT_CHAR_LIMIT]
            user_input = build_gemini_input(
                question=question,
                context_text=context_text,
                history=history,
            )
            async for token in self.gemini.chat_stream(
                system_instruction=SYSTEM_PROMPT,
                user_prompt=user_input,
                model=chosen_model,
                api_key=api_key,
            ):
                yield {"type": "token", "content": token}
        else:
            if len(full_text) <= self.offline_direct_char_limit:
                messages = build_messages(
                    question=question,
                    context_text=full_text,
                    history=history,
                )
            else:
                synthesized = await self._map_reduce_offline(
                    question=question,
                    chunks=sources,
                    model=chosen_model,
                )
                messages = build_messages(
                    question=question,
                    context_text=synthesized,
                    history=history,
                    context_label="Notes extracted across all PDF sections",
                )
            async for token in self.ollama.chat_stream(messages=messages, model=chosen_model):
                yield {"type": "token", "content": token}

        yield {"type": "done"}

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store and remove any saved upload."""
        deleted = self.store.delete_document(doc_id)
        if deleted:
            for path in self.settings.upload_dir.glob(f"{doc_id}_*"):
                path.unlink(missing_ok=True)
        return deleted

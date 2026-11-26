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
StyleName = Literal["simple", "detailed", "hinglish", "notes"]

OFFLINE_DIRECT_CHAR_LIMIT = 24_000
OFFLINE_SECTION_CHAR_LIMIT = 18_000
ONLINE_DIRECT_CHAR_LIMIT = 2_400_000

BASE_SYSTEM_PROMPT = (
    "You are a helpful, clear teacher and document assistant. "
    "Base your answers strictly on the provided document contents—do not invent facts "
    "that aren't in the documents. "
    "Cite sources naturally at the end of a paragraph or bullet using [filename | Page X] "
    "(or [filename | Page X, Y]) instead of interrupting every phrase. "
    "If something asked is not in the documents, say so politely and briefly."
)

STYLE_INSTRUCTIONS: Dict[str, str] = {
    "simple": (
        "Explain in plain, easy-to-understand language like a friendly teacher. "
        "Use short paragraphs, everyday analogies or simple examples when helpful, "
        "and clear headings or bullet points so it's effortless to read."
    ),
    "detailed": (
        "Give a thorough, step-by-step explanation covering every important concept, "
        "architecture flow, and detail from the documents. Organize with clear headings."
    ),
    "hinglish": (
        "Explain in natural, easy Hinglish (Roman Hindi + English technical terms), "
        "just like a friend or teacher explaining clearly. Keep technical terms in English "
        "and use simple relatable examples."
    ),
    "notes": (
        "Format the answer as clean, exam-ready study notes with bold headings, "
        "crisp bullet points, key definitions, and main takeaways."
    ),
}

MAP_SYSTEM_PROMPT = (
    "You are reading one section of a larger document to help answer a user's question. "
    "Extract all relevant facts, definitions, formulas, and details from this section "
    "with citations in [filename | Page X] format. If nothing in this section is relevant, "
    "reply briefly with: NO_RELEVANT_INFO."
)


def build_system_prompt(
    style: StyleName = "simple",
    custom_instruction: Optional[str] = None,
) -> str:
    """Combine base grounding rules with the user's chosen explanation style."""
    style_text = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["simple"])
    parts = [BASE_SYSTEM_PROMPT, f"Explanation style: {style_text}"]
    if custom_instruction and custom_instruction.strip():
        parts.append(f"Additional user preference: {custom_instruction.strip()}")
    return "\n\n".join(parts)


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
    system_prompt: str = BASE_SYSTEM_PROMPT,
    history: Optional[List[Dict[str, str]]] = None,
    context_label: str = "Full document contents",
) -> List[Dict[str, str]]:
    """Build chat messages for Ollama combining system instructions, history, and context."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-6:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_prompt = (
        f"{context_label}:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer in the requested explanation style and include [filename | Page X] citations naturally."
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
        "Answer in the requested explanation style and include [filename | Page X] citations naturally."
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
    """Coordinates instant document ingestion and Full-PDF Offline/Online Q&A."""

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

    def _default_chat_model(self, provider: ProviderName) -> str:
        return (
            self.settings.gemini_chat_model
            if provider == "gemini"
            else self.settings.chat_model
        )

    async def ingest_file(
        self,
        filename: str,
        content: bytes,
        provider: ProviderName = "ollama",
        embed_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> IngestResult:
        """Parse, chunk, and store document pages immediately without requiring embedding APIs."""
        safe_name = Path(filename).name
        file_hash = hashlib.sha256(content).hexdigest()
        chosen_embed = embed_model or "full-pdf"

        existing = self.store.get_by_hash(file_hash)
        if existing is not None:
            return IngestResult(document=existing, already_indexed=True)

        pages = extract_pages(safe_name, content)
        chunks = chunk_pages(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # Full-PDF Mode reads chunks sequentially in reading order, so zero-vectors suffice
        # and uploads succeed instantly offline or online without extra API quota.
        placeholder_embeddings = [[0.0] for _ in chunks]

        record = self.store.add_document(
            filename=safe_name,
            file_hash=file_hash,
            page_count=len(pages),
            chunks=chunks,
            embeddings=placeholder_embeddings,
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
        style: StyleName = "simple",
        custom_instruction: Optional[str] = None,
        doc_ids: Optional[Sequence[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,
    ) -> ChatResponse:
        """Read the full selected PDF(s) and generate a complete answer."""
        chosen_model = model or self._default_chat_model(provider)
        sources = self.load_full_documents(doc_ids=doc_ids)
        if not sources:
            return ChatResponse(
                answer="No uploaded files found matching your selection. Please upload a PDF or note first.",
                provider=provider,
                model=chosen_model,
                sources=[],
            )

        sys_prompt = build_system_prompt(style=style, custom_instruction=custom_instruction)
        full_text = build_context_block(sources)

        if provider == "gemini":
            context_text = full_text[:ONLINE_DIRECT_CHAR_LIMIT]
            user_input = build_gemini_input(
                question=question,
                context_text=context_text,
                history=history,
            )
            answer = await self.gemini.chat(
                system_instruction=sys_prompt,
                user_prompt=user_input,
                model=chosen_model,
                api_key=api_key,
            )
        else:
            if len(full_text) <= self.offline_direct_char_limit:
                messages = build_messages(
                    question=question,
                    context_text=full_text,
                    system_prompt=sys_prompt,
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
                    system_prompt=sys_prompt,
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
        style: StyleName = "simple",
        custom_instruction: Optional[str] = None,
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
                "content": "No uploaded files found matching your selection. Please upload a PDF or note first.",
            }
            yield {"type": "done"}
            return

        sys_prompt = build_system_prompt(style=style, custom_instruction=custom_instruction)
        full_text = build_context_block(sources)

        if provider == "gemini":
            context_text = full_text[:ONLINE_DIRECT_CHAR_LIMIT]
            user_input = build_gemini_input(
                question=question,
                context_text=context_text,
                history=history,
            )
            async for token in self.gemini.chat_stream(
                system_instruction=sys_prompt,
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
                    system_prompt=sys_prompt,
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
                    system_prompt=sys_prompt,
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

"""Hybrid AI Chat pipeline supporting general Q&A and optional Full-PDF context in Offline/Online modes."""

from dataclasses import dataclass, field
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
    "You are a helpful, knowledgeable AI assistant and friendly teacher. "
    "You can answer any question across coding, science, math, study topics, writing, "
    "and general knowledge. "
    "If attached document excerpts are also provided below, use them carefully alongside "
    "your knowledge and cite [filename | Page X] naturally when referencing information from them."
)

STYLE_INSTRUCTIONS: Dict[str, str] = {
    "simple": (
        "Explain in plain, easy-to-understand language like a friendly teacher. "
        "Use short paragraphs, everyday analogies or simple examples when helpful, "
        "and clear headings or bullet points so it's effortless to read."
    ),
    "detailed": (
        "Give a thorough, step-by-step explanation covering every important concept, "
        "example, and detail. Organize with clear headings."
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
    "You are reading one section of an attached document to help answer a user's question. "
    "Extract all relevant facts, definitions, formulas, and details from this section "
    "with citations in [filename | Page X] format. If nothing in this section is relevant, "
    "reply briefly with: NO_RELEVANT_INFO."
)


def build_system_prompt(
    style: StyleName = "simple",
    custom_instruction: Optional[str] = None,
) -> str:
    """Combine general assistant behavior with the user's chosen explanation style."""
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
    """Full non-streaming answer with optional document and web search citations."""

    answer: str
    provider: ProviderName
    model: str
    sources: List[SearchResult]
    web_sources: List[Dict[str, str]] = field(default_factory=list)


def build_context_block(sources: Sequence[SearchResult]) -> str:
    """Format document chunks into a labeled context block."""
    parts: List[str] = []
    for src in sources:
        parts.append(f"[{src.filename} | Page {src.page_number}]\n{src.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    question: str,
    context_text: Optional[str] = None,
    system_prompt: str = BASE_SYSTEM_PROMPT,
    history: Optional[List[Dict[str, str]]] = None,
    context_label: str = "Attached document contents",
) -> List[Dict[str, str]]:
    """Build chat messages for Ollama with optional attached document context."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-8:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    if context_text:
        user_prompt = (
            f"{context_label}:\n\n{context_text}\n\n"
            f"Question: {question}"
        )
    else:
        user_prompt = question

    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_gemini_input(
    question: str,
    context_text: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build a structured input string for Gemini Interactions API."""
    sections: List[str] = []
    if history:
        recent = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in history[-8:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if recent:
            sections.append("Conversation so far:\n" + "\n".join(recent))

    if context_text:
        sections.append(f"Attached document contents:\n\n{context_text}")

    sections.append(question if not sections else f"User question: {question}")
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
    """Coordinates general Offline/Online AI chat with optional document attachments."""

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
        """Parse, chunk, and store optional document pages immediately."""
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
            return "Attached document sections had no direct match; answer using general knowledge."
        return "\n\n---\n\n".join(findings)

    async def _prepare_context(
        self,
        question: str,
        provider: ProviderName,
        model: str,
        sources: Sequence[SearchResult],
    ) -> tuple[Optional[str], str]:
        """Prepare optional context string + label for the selected provider."""
        if not sources:
            return None, ""

        full_text = build_context_block(sources)
        if provider == "gemini":
            return full_text[:ONLINE_DIRECT_CHAR_LIMIT], "Attached document contents"

        if len(full_text) <= self.offline_direct_char_limit:
            return full_text, "Attached document contents"

        synthesized = await self._map_reduce_offline(
            question=question,
            chunks=sources,
            model=model,
        )
        return synthesized, "Notes extracted across attached document sections"

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
        web_search: bool = True,
    ) -> ChatResponse:
        """Answer any question directly, optionally using documents and Google Search."""
        chosen_model = model or self._default_chat_model(provider)
        sources = self.load_full_documents(doc_ids=doc_ids)
        sys_prompt = build_system_prompt(style=style, custom_instruction=custom_instruction)
        context_text, context_label = await self._prepare_context(
            question=question,
            provider=provider,
            model=chosen_model,
            sources=sources,
        )

        web_sources: List[Dict[str, str]] = []
        if provider == "gemini":
            user_input = build_gemini_input(
                question=question,
                context_text=context_text,
                history=history,
            )
            answer, web_sources = await self.gemini.chat_with_sources(
                system_instruction=sys_prompt,
                user_prompt=user_input,
                model=chosen_model,
                api_key=api_key,
                use_google_search=web_search,
            )
        else:
            messages = build_messages(
                question=question,
                context_text=context_text,
                system_prompt=sys_prompt,
                history=history,
                context_label=context_label or "Attached document contents",
            )
            answer = await self.ollama.chat(messages=messages, model=chosen_model)

        return ChatResponse(
            answer=answer,
            provider=provider,
            model=chosen_model,
            sources=sources,
            web_sources=web_sources,
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
        web_search: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream answer for any question, optionally using documents and Google Search."""
        chosen_model = model or self._default_chat_model(provider)
        sources = self.load_full_documents(doc_ids=doc_ids)

        yield {
            "type": "sources",
            "provider": provider,
            "model": chosen_model,
            "sources": [_serialize_source(s) for s in sources],
        }

        sys_prompt = build_system_prompt(style=style, custom_instruction=custom_instruction)
        context_text, context_label = await self._prepare_context(
            question=question,
            provider=provider,
            model=chosen_model,
            sources=sources,
        )

        if provider == "gemini":
            user_input = build_gemini_input(
                question=question,
                context_text=context_text,
                history=history,
            )
            async for evt in self.gemini.chat_stream_events(
                system_instruction=sys_prompt,
                user_prompt=user_input,
                model=chosen_model,
                api_key=api_key,
                use_google_search=web_search,
            ):
                yield evt
        else:
            messages = build_messages(
                question=question,
                context_text=context_text,
                system_prompt=sys_prompt,
                history=history,
                context_label=context_label or "Attached document contents",
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

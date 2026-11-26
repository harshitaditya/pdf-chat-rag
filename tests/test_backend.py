"""End-to-end tests covering Full-PDF Mode in Offline (Ollama), Online (Gemini), and Map-Reduce."""

from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend.app import create_app
from backend.config import Settings
from backend.gemini_client import GeminiClient
from backend.ollama_client import OllamaClient
from backend.parser import PageText, chunk_pages, extract_pages
from backend.rag import RAGService
from backend.vector_store import VectorStore


class FakeOllamaClient(OllamaClient):
    """Deterministic local fake for Ollama embeddings and chat generation."""

    def __init__(self) -> None:
        super().__init__(base_url="http://fake-ollama:11434")
        self.chat_calls: List[List[Dict[str, str]]] = []

    async def list_models(self) -> List[str]:
        return ["qwen2.5:7b", "nomic-embed-text"]

    async def embed(self, texts: List[str], model: str) -> List[List[float]]:
        return [[0.5, 0.5, 0.5] for _ in texts]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.chat_calls.append(messages)
        user_msg = messages[-1]["content"]
        if user_msg.startswith("Section "):
            return "Extracted section fact [big.md | Page 1]."
        return "Full PDF B-Tree answer [notes.md | Page 1]."

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        self.chat_calls.append(messages)
        for part in ["Full ", "PDF ", "streamed answer."]:
            yield part


class FakeGeminiClient(GeminiClient):
    """Deterministic fake for Gemini API embeddings and Interactions streaming."""

    def __init__(self) -> None:
        super().__init__(api_key="fake-gemini-key")
        self.prompts: List[str] = []

    async def embed(
        self,
        texts: List[str],
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
    ) -> List[List[float]]:
        self._resolve_key(api_key)
        return [[0.2, 0.4, 0.6, 0.8] for _ in texts]

    async def chat(
        self,
        system_instruction: str,
        user_prompt: str,
        model: str = "gemini-4-hydrogen",
        api_key: Optional[str] = None,
    ) -> str:
        self._resolve_key(api_key)
        self.prompts.append(user_prompt)
        return f"Gemini ({model}) full-document answer [ml.md | Page 1]."

    async def chat_stream(
        self,
        system_instruction: str,
        user_prompt: str,
        model: str = "gemini-4-hydrogen",
        api_key: Optional[str] = None,
    ) -> AsyncIterator[str]:
        self._resolve_key(api_key)
        self.prompts.append(user_prompt)
        for part in ["Transformers ", "use self-attention ", "[ml.md | Page 1]."]:
            yield part


def _make_blank_pdf_bytes() -> bytes:
    """Create a valid PDF with one blank page (no extractable text)."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_chunk_pages_preserves_page_numbers_and_overlap() -> None:
    pages = [
        PageText(page_number=1, text="Alpha sentence. " * 40),
        PageText(page_number=2, text="Beta short page."),
    ]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 3
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
    assert chunks[-1].text == "Beta short page."


def test_extract_pages_rejects_empty_pdf() -> None:
    pdf_bytes = _make_blank_pdf_bytes()
    try:
        extract_pages("scanned.pdf", pdf_bytes)
    except ValueError as exc:
        assert "No extractable text" in str(exc)
    else:
        raise AssertionError("Expected ValueError for blank PDF")


def test_ui_root_has_simple_pdf_chat_and_no_top_k(tmp_path: Path) -> None:
    cfg = Settings(db_path=tmp_path / "rag.db", upload_dir=tmp_path / "uploads")
    app = create_app(
        app_settings=cfg,
        ollama_client=FakeOllamaClient(),
        gemini_client=FakeGeminiClient(),
    )
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "PDF Chat" in resp.text
    assert "topKInput" not in resp.text


def test_hybrid_full_pdf_offline_online_and_map_reduce(tmp_path: Path) -> None:
    cfg = Settings(
        db_path=tmp_path / "rag.db",
        upload_dir=tmp_path / "uploads",
        chunk_size=300,
        chunk_overlap=50,
    )
    fake_ollama = FakeOllamaClient()
    fake_gemini = FakeGeminiClient()
    rag = RAGService(
        store=VectorStore(cfg.db_path),
        ollama=fake_ollama,
        gemini=fake_gemini,
        settings=cfg,
        offline_direct_char_limit=600,
        offline_section_char_limit=350,
    )
    app = create_app(app_settings=cfg, rag_service=rag)
    client = TestClient(app)

    # 1. Health reports full_pdf mode
    health = client.get("/health").json()
    assert health["mode"] == "full_pdf"
    assert health["ollama"]["reachable"] is True
    assert health["gemini"]["configured"] is True

    # 2. Upload small doc via Offline Ollama
    notes = (
        b"# Database Notes\n\n"
        b"B-Tree indexing keeps keys sorted in balanced pages.\n\n"
        b"ACID guarantees atomicity, consistency, isolation, and durability."
    )
    off_up = client.post(
        "/documents/upload?provider=ollama",
        files=[("files", ("notes.md", notes, "text/markdown"))],
    )
    assert off_up.status_code == 200
    off_doc = off_up.json()[0]

    # 3. Ask Offline on small doc (reads full doc in one pass, no map calls)
    off_chat = client.post(
        "/chat",
        json={
            "question": "Summarize the whole notes file",
            "provider": "ollama",
            "doc_ids": [off_doc["doc_id"]],
        },
    )
    assert off_chat.status_code == 200
    assert len(fake_ollama.chat_calls) == 1
    assert "B-Tree indexing" in fake_ollama.chat_calls[0][-1]["content"]
    assert "ACID guarantees" in fake_ollama.chat_calls[0][-1]["content"]

    # 4. Upload a larger doc exceeding offline_direct_char_limit -> triggers Map-Reduce
    big_text = ("Section detail about distributed consensus and Paxos. " * 25).encode()
    big_up = client.post(
        "/documents/upload?provider=ollama",
        files=[("files", ("big.md", big_text, "text/markdown"))],
    )
    assert big_up.status_code == 200
    big_doc = big_up.json()[0]

    fake_ollama.chat_calls.clear()
    big_stream = client.post(
        "/chat/stream",
        json={
            "question": "Summarize every section",
            "provider": "ollama",
            "doc_ids": [big_doc["doc_id"]],
        },
    )
    assert big_stream.status_code == 200
    assert '"type": "token"' in big_stream.text
    assert '"type": "done"' in big_stream.text
    # At least 2 map calls + 1 final stream synthesis call
    assert len(fake_ollama.chat_calls) >= 3

    # 5. Cross-mode Online Gemini reading an Offline-indexed doc directly
    on_chat = client.post(
        "/chat",
        json={
            "question": "Read everything in notes.md",
            "provider": "gemini",
            "model": "gemini-4-hydrogen",
            "doc_ids": [off_doc["doc_id"]],
        },
    )
    assert on_chat.status_code == 200
    assert "B-Tree indexing" in fake_gemini.prompts[-1]
    assert "ACID guarantees" in fake_gemini.prompts[-1]

    # 6. Cleanup
    assert client.delete(f"/documents/{off_doc['doc_id']}").status_code == 200
    assert client.delete(f"/documents/{big_doc['doc_id']}").status_code == 200

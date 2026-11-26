"""End-to-end tests covering Full-PDF Mode, explanation styles, instant upload, and streaming."""

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
    """Deterministic local fake for Ollama chat generation."""

    def __init__(self) -> None:
        super().__init__(base_url="http://fake-ollama:11434")
        self.chat_calls: List[List[Dict[str, str]]] = []

    async def list_models(self) -> List[str]:
        return ["qwen2.5:7b"]

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
    """Deterministic fake for Gemini Interactions API."""

    def __init__(self, api_key: str = "fake-gemini-key") -> None:
        super().__init__(api_key=api_key)
        self.prompts: List[str] = []
        self.system_instructions: List[str] = []

    async def chat(
        self,
        system_instruction: str,
        user_prompt: str,
        model: str = "gemini-4-hydrogen",
        api_key: Optional[str] = None,
    ) -> str:
        self._resolve_key(api_key)
        self.system_instructions.append(system_instruction)
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
        self.system_instructions.append(system_instruction)
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


def test_ui_root_has_style_picker_and_no_top_k(tmp_path: Path) -> None:
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
    assert "styleSelect" in resp.text
    assert "customInstructionInput" in resp.text
    assert "topKInput" not in resp.text


def test_instant_upload_styles_and_map_reduce(tmp_path: Path) -> None:
    cfg = Settings(
        db_path=tmp_path / "rag.db",
        upload_dir=tmp_path / "uploads",
        chunk_size=300,
        chunk_overlap=50,
    )
    fake_ollama = FakeOllamaClient()
    # Even with NO Gemini API key configured, PDF upload succeeds immediately.
    fake_gemini = FakeGeminiClient(api_key="")
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

    # 1. Instant upload under gemini provider without needing an API key
    notes = (
        b"# Database Notes\n\n"
        b"B-Tree indexing keeps keys sorted in balanced pages.\n\n"
        b"ACID guarantees atomicity, consistency, isolation, and durability."
    )
    up_resp = client.post(
        "/documents/upload?provider=gemini",
        files=[("files", ("notes.md", notes, "text/markdown"))],
    )
    assert up_resp.status_code == 200
    doc = up_resp.json()[0]

    # 2. Ask Online with Hinglish style + custom instruction (passing key via header)
    on_chat = client.post(
        "/chat",
        headers={"X-Gemini-Api-Key": "test-header-key"},
        json={
            "question": "Explain this PDF",
            "provider": "gemini",
            "model": "gemini-4-hydrogen",
            "style": "hinglish",
            "custom_instruction": "Use real-life examples",
            "doc_ids": [doc["doc_id"]],
        },
    )
    assert on_chat.status_code == 200
    assert "Hinglish" in fake_gemini.system_instructions[-1]
    assert "Use real-life examples" in fake_gemini.system_instructions[-1]
    assert "B-Tree indexing" in fake_gemini.prompts[-1]

    # 3. Large doc triggers Offline Map-Reduce and streams
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
            "style": "notes",
            "doc_ids": [big_doc["doc_id"]],
        },
    )
    assert big_stream.status_code == 200
    assert '"type": "token"' in big_stream.text
    assert '"type": "done"' in big_stream.text
    assert len(fake_ollama.chat_calls) >= 3

    # 4. Cleanup
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 200
    assert client.delete(f"/documents/{big_doc['doc_id']}").status_code == 200

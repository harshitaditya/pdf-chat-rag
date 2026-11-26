"""End-to-end tests covering General Ask-Anything Chat, optional PDF context, styles, and streaming."""

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
        return "General or PDF answer from Ollama."

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        self.chat_calls.append(messages)
        for part in ["Hello! ", "Here is your answer."]:
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
        return f"Gemini ({model}) response."

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
        for part in ["REST APIs ", "use HTTP methods."]:
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


def test_ui_root_has_ai_chat_and_style_picker(tmp_path: Path) -> None:
    cfg = Settings(db_path=tmp_path / "rag.db", upload_dir=tmp_path / "uploads")
    app = create_app(
        app_settings=cfg,
        ollama_client=FakeOllamaClient(),
        gemini_client=FakeGeminiClient(),
    )
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AI Chat" in resp.text
    assert "styleSelect" in resp.text
    assert "customInstructionInput" in resp.text
    assert "topKInput" not in resp.text


def test_general_chat_without_pdfs_and_with_optional_pdf(tmp_path: Path) -> None:
    cfg = Settings(
        db_path=tmp_path / "rag.db",
        upload_dir=tmp_path / "uploads",
        chunk_size=300,
        chunk_overlap=50,
    )
    fake_ollama = FakeOllamaClient()
    fake_gemini = FakeGeminiClient(api_key="test-key")
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

    # 1. Pure general chat WITHOUT any uploaded PDFs (Online Gemini)
    gen_resp = client.post(
        "/chat/stream",
        json={
            "question": "Explain REST APIs simply",
            "provider": "gemini",
            "style": "hinglish",
            "custom_instruction": "Give a real-life example",
            "doc_ids": [],
        },
    )
    assert gen_resp.status_code == 200
    assert '"type": "token"' in gen_resp.text
    assert '"type": "done"' in gen_resp.text
    assert "REST APIs simply" in fake_gemini.prompts[-1]
    assert "Hinglish" in fake_gemini.system_instructions[-1]
    assert "Give a real-life example" in fake_gemini.system_instructions[-1]

    # 2. Pure general chat WITHOUT any uploaded PDFs (Offline Ollama)
    off_gen = client.post(
        "/chat",
        json={"question": "Write a Python function to reverse a string", "provider": "ollama"},
    )
    assert off_gen.status_code == 200
    assert off_gen.json()["sources"] == []
    assert "reverse a string" in fake_ollama.chat_calls[-1][-1]["content"]

    # 3. Upload an optional file and chat with it attached
    notes = b"# Database Notes\n\nB-Tree indexing keeps keys sorted in balanced pages."
    up_resp = client.post(
        "/documents/upload?provider=gemini",
        files=[("files", ("notes.md", notes, "text/markdown"))],
    )
    assert up_resp.status_code == 200
    doc_id = up_resp.json()[0]["doc_id"]

    with_doc = client.post(
        "/chat",
        json={
            "question": "Explain B-Tree from my file",
            "provider": "gemini",
            "doc_ids": [doc_id],
        },
    )
    assert with_doc.status_code == 200
    assert "B-Tree indexing" in fake_gemini.prompts[-1]
    assert len(with_doc.json()["sources"]) == 1

    # 4. Unselecting doc_ids=[] ignores uploaded file and answers as general chat
    fake_gemini.prompts.clear()
    no_doc = client.post(
        "/chat",
        json={
            "question": "What is Docker?",
            "provider": "gemini",
            "doc_ids": [],
        },
    )
    assert no_doc.status_code == 200
    assert no_doc.json()["sources"] == []
    assert "B-Tree indexing" not in fake_gemini.prompts[-1]

    # 5. Cleanup
    assert client.delete(f"/documents/{doc_id}").status_code == 200

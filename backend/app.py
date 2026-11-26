"""FastAPI application exposing Hybrid (Offline Ollama + Online Gemini) Full-PDF API and UI."""

import json
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import STATIC_DIR, Settings, settings as default_settings
from backend.gemini_client import GeminiClient, GeminiError
from backend.ollama_client import OllamaClient, OllamaError
from backend.rag import ProviderName, RAGService, StyleName
from backend.vector_store import VectorStore


class HistoryMessage(BaseModel):
    """Prior turn in a conversation."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Request payload for /chat and /chat/stream."""

    question: str = Field(..., min_length=1)
    provider: ProviderName = "ollama"
    model: Optional[str] = None
    style: StyleName = "simple"
    custom_instruction: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    history: Optional[List[HistoryMessage]] = None


class SourceOut(BaseModel):
    """Citation metadata returned alongside generated answers."""

    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float
    embed_provider: str
    text: str


class ChatResponseOut(BaseModel):
    """Structured response for non-streaming /chat."""

    answer: str
    provider: ProviderName
    model: str
    sources: List[SourceOut]


class DocumentOut(BaseModel):
    """Indexed document summary."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    embed_provider: str
    embed_model: str
    created_at: str
    already_indexed: Optional[bool] = None


def create_app(
    app_settings: Optional[Settings] = None,
    ollama_client: Optional[OllamaClient] = None,
    gemini_client: Optional[GeminiClient] = None,
    rag_service: Optional[RAGService] = None,
) -> FastAPI:
    """Application factory supporting isolated settings and provider clients."""
    cfg = app_settings or default_settings
    store = rag_service.store if rag_service else VectorStore(cfg.db_path)
    ollama = rag_service.ollama if rag_service else (
        ollama_client or OllamaClient(base_url=cfg.ollama_base_url)
    )
    gemini = rag_service.gemini if rag_service else (
        gemini_client
        or GeminiClient(
            api_key=cfg.gemini_api_key,
            base_url=cfg.gemini_base_url,
        )
    )
    rag = rag_service or RAGService(
        store=store,
        ollama=ollama,
        gemini=gemini,
        settings=cfg,
    )

    app = FastAPI(
        title="Hybrid Local & Cloud Full-PDF Chat",
        description="Full-PDF & Notes Q&A powered by Offline Ollama and Online Gemini API.",
        version="0.4.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.state.rag = rag
    app.state.ollama = ollama
    app.state.gemini = gemini
    app.state.store = store
    app.state.settings = cfg

    @app.get("/", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="UI assets not found.")
        return FileResponse(str(index_path))

    @app.get("/health")
    async def health(
        x_gemini_api_key: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        ollama_status = await ollama.health()
        gemini_status = gemini.health(api_key=x_gemini_api_key)
        docs = store.list_documents()
        return {
            "status": "ok",
            "mode": "full_pdf",
            "documents_indexed": len(docs),
            "defaults": {
                "ollama_chat_model": cfg.chat_model,
                "ollama_embed_model": cfg.embed_model,
                "gemini_chat_model": cfg.gemini_chat_model,
                "gemini_embed_model": cfg.gemini_embed_model,
            },
            "ollama": ollama_status,
            "gemini": gemini_status,
        }

    @app.get("/models")
    async def list_models(
        provider: Optional[Literal["ollama", "gemini"]] = Query(default=None),
        x_gemini_api_key: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        if provider == "gemini":
            return {"provider": "gemini", "models": gemini.DEFAULT_CHAT_MODELS}
        if provider == "ollama":
            try:
                models = await ollama.list_models()
                return {"provider": "ollama", "models": models}
            except OllamaError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        ollama_status = await ollama.health()
        gemini_status = gemini.health(api_key=x_gemini_api_key)
        return {
            "ollama": ollama_status,
            "gemini": gemini_status,
        }

    @app.get("/documents", response_model=List[DocumentOut])
    async def list_documents() -> List[DocumentOut]:
        return [
            DocumentOut(
                doc_id=d.doc_id,
                filename=d.filename,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                embed_provider=d.embed_provider,
                embed_model=d.embed_model,
                created_at=d.created_at,
            )
            for d in store.list_documents()
        ]

    @app.post("/documents/upload", response_model=List[DocumentOut])
    async def upload_documents(
        files: List[UploadFile] = File(...),
        provider: ProviderName = Query(default="ollama"),
        embed_model: Optional[str] = Query(default=None),
        x_gemini_api_key: Optional[str] = Header(default=None),
    ) -> List[DocumentOut]:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        results: List[DocumentOut] = []
        for f in files:
            raw = await f.read()
            filename = f.filename or "untitled.txt"
            try:
                res = await rag.ingest_file(
                    filename=filename,
                    content=raw,
                    provider=provider,
                    embed_model=embed_model,
                    api_key=x_gemini_api_key,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except (OllamaError, GeminiError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            doc = res.document
            results.append(
                DocumentOut(
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page_count=doc.page_count,
                    chunk_count=doc.chunk_count,
                    embed_provider=doc.embed_provider,
                    embed_model=doc.embed_model,
                    created_at=doc.created_at,
                    already_indexed=res.already_indexed,
                )
            )
        return results

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str) -> Dict[str, Any]:
        deleted = rag.delete_document(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return {"deleted": True, "doc_id": doc_id}

    @app.post("/chat", response_model=ChatResponseOut)
    async def chat(
        req: ChatRequest,
        x_gemini_api_key: Optional[str] = Header(default=None),
    ) -> ChatResponseOut:
        history_dicts = (
            [{"role": m.role, "content": m.content} for m in req.history]
            if req.history
            else None
        )
        try:
            res = await rag.ask(
                question=req.question,
                provider=req.provider,
                model=req.model,
                style=req.style,
                custom_instruction=req.custom_instruction,
                doc_ids=req.doc_ids,
                history=history_dicts,
                api_key=x_gemini_api_key,
            )
        except (OllamaError, GeminiError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return ChatResponseOut(
            answer=res.answer,
            provider=res.provider,
            model=res.model,
            sources=[
                SourceOut(
                    doc_id=s.doc_id,
                    filename=s.filename,
                    page_number=s.page_number,
                    chunk_index=s.chunk_index,
                    score=round(s.score, 4),
                    embed_provider=s.embed_provider,
                    text=s.text,
                )
                for s in res.sources
            ],
        )

    @app.post("/chat/stream")
    async def chat_stream(
        req: ChatRequest,
        x_gemini_api_key: Optional[str] = Header(default=None),
    ) -> StreamingResponse:
        history_dicts = (
            [{"role": m.role, "content": m.content} for m in req.history]
            if req.history
            else None
        )

        async def event_generator():
            try:
                async for event in rag.ask_stream(
                    question=req.question,
                    provider=req.provider,
                    model=req.model,
                    style=req.style,
                    custom_instruction=req.custom_instruction,
                    doc_ids=req.doc_ids,
                    history=history_dicts,
                    api_key=x_gemini_api_key,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except (OllamaError, GeminiError) as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


app = create_app()

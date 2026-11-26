"""Configuration settings for the Hybrid (Offline Ollama + Online Gemini) RAG backend."""

from dataclasses import dataclass
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data")))
STATIC_DIR = BASE_DIR / "static"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables with sensible defaults."""

    # Offline (Ollama)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Online (Gemini API)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    )
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-4-hydrogen")
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

    # Storage & RAG tuning
    db_path: Path = Path(os.getenv("RAG_DB_PATH", str(DATA_DIR / "rag_store.db")))
    upload_dir: Path = Path(os.getenv("RAG_UPLOAD_DIR", str(DATA_DIR / "uploads")))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    default_top_k: int = int(os.getenv("RAG_TOP_K", "5"))


settings = Settings()

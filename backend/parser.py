"""Document text extraction and overlapping chunking utilities."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import List

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class PageText:
    """Extracted text for a single document page or section."""

    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    """A single indexed text chunk with source page metadata."""

    chunk_index: int
    page_number: int
    text: str


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trailing spaces while preserving paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(filename: str, content: bytes) -> List[PageText]:
    """Extract cleaned text per page from PDF, Markdown, or plain-text bytes."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        reader = PdfReader(BytesIO(content))
        pages: List[PageText] = []
        for idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _normalize_whitespace(raw)
            if cleaned:
                pages.append(PageText(page_number=idx, text=cleaned))
        if not pages:
            raise ValueError(
                f"No extractable text found in '{filename}'. "
                "If this PDF is a scanned image, OCR is required first."
            )
        return pages

    # Plain text / Markdown files are decoded as UTF-8 with replacement fallback.
    decoded = content.decode("utf-8", errors="replace")
    cleaned = _normalize_whitespace(decoded)
    if not cleaned:
        raise ValueError(f"File '{filename}' contains no readable text.")
    return [PageText(page_number=1, text=cleaned)]


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[TextChunk]:
    """Split extracted pages into overlapping chunks while preserving page numbers."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: List[TextChunk] = []
    chunk_idx = 0

    for page in pages:
        text = page.text
        if len(text) <= chunk_size:
            chunks.append(
                TextChunk(chunk_index=chunk_idx, page_number=page.page_number, text=text)
            )
            chunk_idx += 1
            continue

        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Prefer breaking on a sentence or word boundary when not at the end.
            if end < text_len:
                window = text[start:end]
                boundary = max(
                    window.rfind("\n\n"),
                    window.rfind(". "),
                    window.rfind("\n"),
                    window.rfind(" "),
                )
                if boundary > chunk_size // 2:
                    end = start + boundary + 1

            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_idx,
                        page_number=page.page_number,
                        text=piece,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break
            next_start = end - chunk_overlap
            start = next_start if next_start > start else end

    return chunks

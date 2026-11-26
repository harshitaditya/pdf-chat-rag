# DocMind RAG — Offline (Ollama) & Online (Gemini) PDF Chat

A full-stack RAG web application to upload PDFs, Markdown, and text notes and ask questions with page citations—supporting **both** **Offline mode (Ollama)** and **Online mode (Google Gemini API)**.

## Features

- **Full PDF Mode (No Top-K Limit):** Reads all pages of selected PDFs end-to-end—sending full documents directly to Online Gemini (1M context) and automatically using section-by-section Map-Reduce for large PDFs in Offline Ollama mode.
- **Dual Mode Switcher:** Toggle between **Offline (Ollama)** (`qwen2.5:7b` / `gemma4:12b` + `nomic-embed-text`) and **Online (Gemini)** (`gemini-4-hydrogen` / `gemini-4-neon` / `gemini-4-carbon` + `gemini-embedding-001`).
- **Modern Single-Server Web UI:** Served directly by FastAPI at `http://localhost:8000` with drag-and-drop uploads, document filter checkboxes, live SSE streaming, and expandable source citations (`filename • Page X`).
- **Persistent Vector Store:** SQLite + NumPy (`data/rag_store.db`) with SHA-256 deduplication and per-provider embedding tracking.

## Project Structure

- `backend/config.py` — Settings for Ollama, Gemini, chunking, and storage.
- `backend/parser.py` — `.pdf` (page-aware), `.md`, and `.txt` extraction and overlapping chunking.
- `backend/vector_store.py` — Persistent SQLite + NumPy cosine similarity store.
- `backend/ollama_client.py` — Async Ollama client (`/api/tags`, `/api/embed`, `/api/chat`).
- `backend/gemini_client.py` — Async Gemini REST client (`embedContent`/`batchEmbedContents` & Interactions API).
- `backend/rag.py` — Hybrid RAG ingestion, retrieval, and streaming orchestration.
- `backend/app.py` — FastAPI server + static UI host.
- `static/` — `index.html`, `styles.css`, `app.js`.
- `tests/test_backend.py` — Automated pytest suite.

## Quick Start

### Option A: Online Mode (Fastest — No Ollama Install Needed)
1. Start the server:
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   .venv/bin/uvicorn backend.app:app --reload --port 8000
   ```
2. Open `http://localhost:8000` in your browser.
3. Switch to **Online (Gemini)** and enter your Gemini API key (or export `GEMINI_API_KEY` before starting the server).

### Option B: Offline Mode (100% Local via Ollama)
1. Install Ollama and pull models once:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull nomic-embed-text
   ollama pull qwen2.5:7b
   ```
2. Start the server and open `http://localhost:8000` in **Offline (Ollama)** mode.

## Run Tests

```bash
.venv/bin/pytest -v
```

## Deploy Live (Cloud Hosting)

Because this app runs a Python FastAPI backend, static hosts like GitHub Pages cannot run the Python server directly. You can deploy it for free/low-cost using the included `Dockerfile` on **Render**, **Railway**, **Fly.io**, or **Google Cloud Run**:

1. Connect your `harshitaditya/pdf-chat-rag` repo (select **Docker** runtime).
2. Optional: set `GEMINI_API_KEY` as an environment variable (or let users paste their own key in the UI's Settings popup for Online mode).

*(Note: Cloud deployments support **Online (Gemini)** mode; **Offline (Ollama)** mode works when running locally on a machine with Ollama installed.)*

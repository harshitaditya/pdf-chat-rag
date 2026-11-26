# AI Chat — Online (Gemini) & Offline (Ollama) Assistant

A clean, simple AI chat web app where you can **ask anything** (coding, study doubts, general questions) using **Online (Google Gemini API)** or **Offline (Ollama)**—with optional PDF/Markdown file attachments.

## Features

- **Ask Anything (General Chat):** Works right away without needing to upload any files—supports conversation history, streaming replies, and clean Markdown formatting.
- **Explanation Styles & Custom Instructions:** Switch between **Simple**, **Detailed**, **Hinglish**, and **Exam Notes**, or add your own custom instructions in Settings.
- **Optional File Context:** Attach `.pdf`, `.md`, or `.txt` files anytime to discuss or summarize their full contents with page citations.
- **Dual Mode Switcher:** Toggle between **Online (Gemini)** (`gemini-4-hydrogen`, `gemini-3.5-flash-lite`, etc.) and **Offline (Ollama)** (`qwen2.5:7b`, `gemma4:12b`, etc.).

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
   cd /home/aditya/.gemini/antigravity/scratch/local-rag-chat
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

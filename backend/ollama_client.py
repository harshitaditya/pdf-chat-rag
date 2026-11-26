"""Async HTTP client for Ollama's local REST API."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error response."""


class OllamaClient:
    """Thin async client wrapping Ollama /api/tags, /api/embed, and /api/chat."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def list_models(self) -> List[str]:
        """Return names of locally installed Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is installed and running (`ollama serve`)."
            ) from exc

        return [m["name"] for m in data.get("models", []) if "name" in m]

    async def health(self) -> Dict[str, Any]:
        """Check whether Ollama is reachable and report available models."""
        try:
            models = await self.list_models()
            return {"reachable": True, "models": models, "error": None}
        except OllamaError as exc:
            return {"reachable": False, "models": [], "error": str(exc)}

    async def embed(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts using Ollama /api/embed."""
        if not texts:
            return []

        payload = {"model": model, "input": texts}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/embed", json=payload)
                if resp.status_code == 404:
                    raise OllamaError(
                        f"Embedding model '{model}' not found in Ollama. "
                        f"Pull it first with: ollama pull {model}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except OllamaError:
            raise
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Ollama embedding request failed at {self.base_url}: {exc}"
            ) from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Ollama returned an unexpected embeddings payload.")
        return embeddings

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a non-streaming chat completion request to Ollama."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                if resp.status_code == 404:
                    raise OllamaError(
                        f"Chat model '{model}' not found in Ollama. "
                        f"Pull it first with: ollama pull {model}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except OllamaError:
            raise
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Ollama chat request failed at {self.base_url}: {exc}"
            ) from exc

        message = data.get("message", {})
        return str(message.get("content", ""))

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Yield streamed response tokens from Ollama /api/chat."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    if resp.status_code == 404:
                        raise OllamaError(
                            f"Chat model '{model}' not found in Ollama. "
                            f"Pull it first with: ollama pull {model}"
                        )
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield str(token)
                        if chunk.get("done"):
                            break
        except OllamaError:
            raise
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Ollama streaming request failed at {self.base_url}: {exc}"
            ) from exc

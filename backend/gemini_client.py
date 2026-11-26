"""Async REST client for the Google Gemini API (Online mode)."""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


class GeminiError(RuntimeError):
    """Raised when the Gemini API key is missing or an API request fails."""


class GeminiClient:
    """Async Gemini API client using Interactions API for chat and embedContent for vectors."""

    DEFAULT_CHAT_MODELS = [
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-3.5-flash-lite",
    ]
    DEFAULT_EMBED_MODELS = [
        "gemini-embedding-001",
        "gemini-embedding-2",
    ]
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.default_api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _resolve_key(self, api_key: Optional[str] = None) -> str:
        key = (api_key or self.default_api_key).strip()
        if not key:
            raise GeminiError(
                "Gemini API key is required for Online mode. "
                "Open Settings and paste your Google AI Studio API key."
            )
        return key

    def health(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Report whether a Gemini API key is configured and list supported models."""
        has_server_key = bool(self.default_api_key)
        has_effective_key = bool((api_key or self.default_api_key).strip())
        return {
            "configured": has_effective_key,
            "has_server_key": has_server_key,
            "chat_models": list(self.DEFAULT_CHAT_MODELS),
            "embed_models": list(self.DEFAULT_EMBED_MODELS),
        }

    @staticmethod
    def _extract_error_message(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            err = body.get("error", {})
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
        except Exception:
            pass
        return resp.text[:300] or f"HTTP {resp.status_code}"

    async def embed(
        self,
        texts: List[str],
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate vector embeddings for texts via Gemini REST API."""
        if not texts:
            return []

        key = self._resolve_key(api_key)
        headers = {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
        }
        model_path = model if model.startswith("models/") else f"models/{model}"

        if model_path == "models/gemini-embedding-001":
            return await self._batch_embed_contents(texts, model_path, headers)
        return await self._single_embed_loop(texts, model_path, headers)

    async def _batch_embed_contents(
        self,
        texts: List[str],
        model_path: str,
        headers: Dict[str, str],
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        batch_size = 50
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    payload = {
                        "requests": [
                            {
                                "model": model_path,
                                "content": {"parts": [{"text": t}]},
                            }
                            for t in batch
                        ]
                    }
                    resp = await client.post(
                        f"{self.base_url}/{model_path}:batchEmbedContents",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code >= 400:
                        raise GeminiError(
                            f"Gemini embedding failed ({resp.status_code}): "
                            f"{self._extract_error_message(resp)}"
                        )
                    data = resp.json()
                    for item in data.get("embeddings", []):
                        vectors.append([float(v) for v in item.get("values", [])])
        except GeminiError:
            raise
        except httpx.HTTPError as exc:
            raise GeminiError(f"Gemini embedding request failed: {exc}") from exc

        if len(vectors) != len(texts):
            raise GeminiError("Gemini returned an unexpected number of embeddings.")
        return vectors

    async def _single_embed_loop(
        self,
        texts: List[str],
        model_path: str,
        headers: Dict[str, str],
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for text in texts:
                    payload = {
                        "model": model_path,
                        "content": {"parts": [{"text": text}]},
                    }
                    resp = await client.post(
                        f"{self.base_url}/{model_path}:embedContent",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code >= 400:
                        raise GeminiError(
                            f"Gemini embedding failed ({resp.status_code}): "
                            f"{self._extract_error_message(resp)}"
                        )
                    data = resp.json()
                    values = data.get("embedding", {}).get("values", [])
                    vectors.append([float(v) for v in values])
        except GeminiError:
            raise
        except httpx.HTTPError as exc:
            raise GeminiError(f"Gemini embedding request failed: {exc}") from exc
        return vectors

    @staticmethod
    def _extract_interaction_text(data: Dict[str, Any]) -> str:
        """Collect text parts from the Interactions API response steps."""
        chunks: List[str] = []
        for step in data.get("steps", []):
            if step.get("type") != "model_output":
                continue
            for content in step.get("content", []):
                if content.get("type") == "text" and content.get("text"):
                    chunks.append(str(content["text"]))
        return "".join(chunks)

    async def chat(
        self,
        system_instruction: str,
        user_prompt: str,
        model: str = "gemini-4-hydrogen",
        api_key: Optional[str] = None,
    ) -> str:
        """Send a non-streaming request to Gemini Interactions API with transient retry."""
        key = self._resolve_key(api_key)
        headers = {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "system_instruction": system_instruction,
            "input": user_prompt,
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/interactions",
                        headers=headers,
                        json=payload,
                    )
                    if (
                        resp.status_code in self.RETRYABLE_STATUS_CODES
                        and attempt < self.max_retries
                    ):
                        await asyncio.sleep(1.2 * (attempt + 1))
                        continue
                    if resp.status_code >= 400:
                        raise GeminiError(
                            f"Gemini chat failed ({resp.status_code}): "
                            f"{self._extract_error_message(resp)}"
                        )
                    return self._extract_interaction_text(resp.json())
            except GeminiError:
                raise
            except httpx.HTTPError as exc:
                last_err = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(1.2 * (attempt + 1))
                    continue
                raise GeminiError(f"Gemini chat request failed: {exc}") from exc

        raise GeminiError(f"Gemini chat request failed: {last_err}")

    async def chat_stream(
        self,
        system_instruction: str,
        user_prompt: str,
        model: str = "gemini-4-hydrogen",
        api_key: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text tokens from Gemini Interactions API via SSE with 503 retry."""
        key = self._resolve_key(api_key)
        headers = {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "system_instruction": system_instruction,
            "input": user_prompt,
            "stream": True,
        }

        for attempt in range(self.max_retries + 1):
            yielded_any = False
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/interactions",
                        headers=headers,
                        json=payload,
                    ) as resp:
                        if (
                            resp.status_code in self.RETRYABLE_STATUS_CODES
                            and attempt < self.max_retries
                        ):
                            await resp.aread()
                            await asyncio.sleep(1.2 * (attempt + 1))
                            continue
                        if resp.status_code >= 400:
                            await resp.aread()
                            raise GeminiError(
                                f"Gemini stream failed ({resp.status_code}): "
                                f"{self._extract_error_message(resp)}"
                            )
                        async for raw_line in resp.aiter_lines():
                            line = raw_line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            payload_str = line[len("data:") :].strip()
                            if not payload_str or payload_str == "[DONE]":
                                continue
                            event = json.loads(payload_str)
                            event_type = event.get("event_type")
                            if event_type == "error":
                                err = event.get("error", {})
                                raise GeminiError(
                                    str(err.get("message", "Gemini streaming error"))
                                )
                            if event_type == "step.delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text" and delta.get("text"):
                                    yielded_any = True
                                    yield str(delta["text"])
                        return
            except GeminiError:
                raise
            except httpx.HTTPError as exc:
                if not yielded_any and attempt < self.max_retries:
                    await asyncio.sleep(1.2 * (attempt + 1))
                    continue
                raise GeminiError(f"Gemini streaming request failed: {exc}") from exc

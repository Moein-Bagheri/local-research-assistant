from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import requests


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]]
    elapsed_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        llm_model: str,
        embedding_model: str,
        timeout: int = 180,
        seed: int = 42,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.seed = seed
        self.session = requests.Session()

    def health(self) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                "Ollama is not reachable. Start it with `ollama serve` and retry."
            ) from exc
        return response.json()

    def model_names(self) -> set[str]:
        payload = self.health()
        return {item.get("name", "") for item in payload.get("models", [])}

    def ensure_models(self) -> None:
        names = self.model_names()
        missing = [
            model
            for model in (self.llm_model, self.embedding_model)
            if model not in names and f"{model}:latest" not in names
        ]
        if missing:
            commands = "\n".join(f"ollama pull {name}" for name in missing)
            raise OllamaError(f"Required Ollama models are missing:\n{commands}")

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.embedding_model, "input": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise OllamaError(f"Embedding request failed: {exc}") from exc
            payload = response.json()
            batch_vectors = payload.get("embeddings")
            if not batch_vectors or len(batch_vectors) != len(batch):
                raise OllamaError("Ollama returned an invalid embedding payload.")
            vectors.extend(batch_vectors)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        format_json: bool = False,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": self.seed,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
            "keep_alive": "10m",
        }
        if tools:
            payload["tools"] = tools
        if format_json:
            payload["format"] = "json"

        started = time.perf_counter()
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Generation request failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        body = response.json()
        message = body.get("message", {})
        return ChatResult(
            content=message.get("content", "").strip(),
            tool_calls=message.get("tool_calls", []) or [],
            elapsed_seconds=elapsed,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
        )


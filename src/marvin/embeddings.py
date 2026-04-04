from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


class EmbeddingBackend(Protocol):
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]: ...


class HashEmbeddingBackend:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = -1.0 if digest[4] % 2 else 1.0
            weight = 1.0 + math.log1p(len(token))
            vector[index] += sign * weight

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class FastEmbedBackend:
    def __init__(self, model_name: str, dimensions: int = 384) -> None:
        from fastembed import TextEmbedding

        self.dimensions = dimensions
        self._model = TextEmbedding(model_name=model_name)

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(vec, dtype=np.float32) for vec in self._model.embed(texts)]


class EmbeddingService:
    def __init__(
        self,
        provider: str = "auto",
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.dimensions = dimensions
        self._backend: EmbeddingBackend | None = None
        self._backend_name: str | None = None

    @property
    def backend_name(self) -> str:
        self._ensure_backend()
        return self._backend_name or "unknown"

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_backend()
        assert self._backend is not None
        return self._backend.embed_texts(texts)

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        if self.provider in {"auto", "fastembed"}:
            try:
                self._backend = FastEmbedBackend(self.model_name, self.dimensions)
                self._backend_name = "fastembed"
                return
            except Exception:
                if self.provider == "fastembed":
                    raise

        self._backend = HashEmbeddingBackend(self.dimensions)
        self._backend_name = "hash"

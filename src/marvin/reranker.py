"""Cross-encoder reranker for Marvin.

Hybrid retrieval (vector + BM25 fused with RRF) is already strong, but RRF is
rank-order only and ignores query-document interaction. A cross-encoder
reranker reads the query and each candidate document jointly and produces a
single relevance logit, which typically lifts top-k precision by 5-15 points
on open-domain QA benchmarks at the cost of a few hundred ms per query.

Design notes:

- The default model is ``BAAI/bge-reranker-v2-m3`` (multilingual, 568M params,
  XLM-RoBERTa backbone, Apache-2.0). We do not wait for upstream ``fastembed``
  support; instead we register the ``onnx-community/bge-reranker-v2-m3-ONNX``
  port (int8 quantized, ~570MB) through :meth:`TextCrossEncoder.add_custom_model`.
  Users can still swap to any natively-supported reranker such as
  ``Xenova/ms-marco-MiniLM-L-6-v2`` or ``BAAI/bge-reranker-base``.
- Backend selection mirrors :mod:`marvin.embeddings`: ``"auto"`` tries
  ``fastembed`` and falls back to a no-op if the model cannot load; ``"none"``
  forces the no-op; ``"fastembed"`` raises on failure.
- The no-op backend returns identity scores so callers can unconditionally
  invoke :meth:`RerankerService.rerank` without branching on availability.
- Reranking is intentionally composed *outside* :class:`MemoryIndex`. The
  index stays a pure retrieval primitive; the service layer decides whether
  to spend compute on reranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import SearchHit

__all__ = [
    "RerankerBackend",
    "NoOpRerankerBackend",
    "FastEmbedRerankerBackend",
    "RerankerService",
    "DEFAULT_RERANK_MODEL",
    "rerank_hits",
]

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

_CUSTOM_FASTEMBED_MODELS: dict[str, dict[str, object]] = {
    # Community ONNX port of bge-reranker-v2-m3. The int8 quantized variant
    # is a single 570MB file which loads cleanly under fastembed's ONNX
    # runtime; the upstream BAAI repo does not ship ONNX weights directly.
    "BAAI/bge-reranker-v2-m3": {
        "hf_repo": "onnx-community/bge-reranker-v2-m3-ONNX",
        "model_file": "onnx/model_quantized.onnx",
        "license": "apache-2.0",
        "size_in_gb": 0.57,
        "description": (
            "BAAI bge-reranker-v2-m3 cross-encoder (int8 quantized ONNX "
            "port from onnx-community)."
        ),
    },
}


class RerankerBackend(Protocol):
    """Score ``(query, document)`` pairs. Higher is more relevant."""

    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class NoOpRerankerBackend:
    """Identity reranker: returns a descending score sequence.

    Used as the fallback when a real reranker is unavailable so the service
    layer can call :meth:`RerankerService.rerank` unconditionally. The
    returned scores preserve input order (first document wins).
    """

    def __init__(self, model_name: str = "noop") -> None:
        self.model_name = model_name

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        n = len(documents)
        return [float(n - i) for i in range(n)]


class FastEmbedRerankerBackend:
    """Cross-encoder reranker backed by ``fastembed.TextCrossEncoder``."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        batch_size: int = 32,
        max_chars: int = 1024,
    ) -> None:
        from fastembed.common.model_description import ModelSource
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars

        spec = _CUSTOM_FASTEMBED_MODELS.get(model_name)
        if spec is not None:
            supported = {
                m["model"] for m in TextCrossEncoder.list_supported_models()
            }
            if model_name not in supported:
                TextCrossEncoder.add_custom_model(
                    model=model_name,
                    sources=ModelSource(hf=str(spec["hf_repo"])),
                    model_file=str(spec["model_file"]),
                    description=str(spec["description"]),
                    license=str(spec["license"]),
                    size_in_gb=float(spec["size_in_gb"]),
                )

        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        truncated = [self._truncate(doc) for doc in documents]
        scores = self._model.rerank(query, truncated, batch_size=self.batch_size)
        return [float(s) for s in scores]

    def _truncate(self, text: str) -> str:
        if self.max_chars and len(text) > self.max_chars:
            return text[: self.max_chars]
        return text


class RerankerService:
    """Lazy-loading cross-encoder reranker with a no-op fallback."""

    def __init__(
        self,
        provider: str = "auto",
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        batch_size: int = 32,
        max_chars: int = 1024,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._backend: RerankerBackend | None = None
        self._backend_name: str | None = None

    @property
    def backend_name(self) -> str:
        self._ensure_backend()
        return self._backend_name or "unknown"

    @property
    def loaded_backend_name(self) -> str | None:
        """Backend name if already loaded; ``None`` if no eager init has happened.

        Use this from health/status endpoints that must not pay a lazy
        model-load cost just to report current state.
        """
        return self._backend_name

    @property
    def is_real(self) -> bool:
        return self.backend_name not in {"noop", "unknown"}

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self._ensure_backend()
        assert self._backend is not None
        return self._backend.score(query, documents)

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        limit: int | None = None,
    ) -> list[SearchHit]:
        """Reorder ``hits`` by cross-encoder score, best first.

        The document shown to the reranker is ``"{title}\\n{excerpt}"``.
        This is a deliberate approximation: ``excerpt`` is already the
        most-salient window from the chunk that matched retrieval, and
        keeping the reranker input short holds CPU cost in check.
        """
        return rerank_hits(
            query,
            hits,
            service=self,
            limit=limit,
        )

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        if self.provider in {"auto", "fastembed"}:
            try:
                self._backend = FastEmbedRerankerBackend(
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    max_chars=self.max_chars,
                )
                self._backend_name = "fastembed"
                return
            except Exception:
                if self.provider == "fastembed":
                    raise

        self._backend = NoOpRerankerBackend(model_name=self.model_name)
        self._backend_name = "noop"


def rerank_hits(
    query: str,
    hits: Sequence[SearchHit],
    *,
    service: RerankerService,
    limit: int | None = None,
) -> list[SearchHit]:
    """Standalone helper: rerank a SearchHit sequence with a service."""
    if not hits:
        return []
    docs = [_hit_to_document(hit) for hit in hits]
    scores = service.score(query, docs)
    # Higher score = more relevant. Preserve original position as stable tiebreaker.
    order = sorted(
        range(len(hits)),
        key=lambda i: (-scores[i], i),
    )
    reordered = [
        hits[i].model_copy(update={"score": round(scores[i], 6)}) for i in order
    ]
    if limit is not None:
        reordered = reordered[:limit]
    return reordered


def _hit_to_document(hit: SearchHit) -> str:
    title = hit.title.strip()
    excerpt = (hit.excerpt or "").strip()
    if title and excerpt:
        return f"{title}\n{excerpt}"
    return title or excerpt

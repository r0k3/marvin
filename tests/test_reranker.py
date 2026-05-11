"""Unit tests for ``marvin.reranker``.

These tests avoid the network. They exercise:

- the no-op backend and identity semantics of ``RerankerService``,
- the service's ability to reorder ``SearchHit`` objects,
- that the bge-reranker-v2-m3 spec is declared for fastembed registration.

Loading the actual 570MB ONNX model is covered by a live smoke test in
``test_reranker_live.py`` which is skipped by default.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from marvin.models import MemoryKind, SearchHit
from marvin.reranker import (
    _CUSTOM_FASTEMBED_MODELS,
    DEFAULT_RERANK_MODEL,
    NoOpRerankerBackend,
    RerankerService,
    _hit_to_document,
    rerank_hits,
)


def _hit(path: str, title: str, excerpt: str = "") -> SearchHit:
    return SearchHit(
        title=title,
        kind=MemoryKind.EPISODIC,
        path=path,
        score=0.0,
        excerpt=excerpt,
    )


class _FakeBackend:
    """Deterministic backend driven by a query->doc->score table."""

    def __init__(self, table: dict[tuple[str, str], float]) -> None:
        self.model_name = "fake"
        self._table = table
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [self._table.get((query, doc), 0.0) for doc in documents]


class TestNoOpBackend:
    def test_identity_preserves_order(self):
        backend = NoOpRerankerBackend()
        scores = backend.score("q", ["a", "b", "c"])
        assert scores == [3.0, 2.0, 1.0]

    def test_empty_documents(self):
        backend = NoOpRerankerBackend()
        assert backend.score("q", []) == []


class TestRerankerServiceFallback:
    def test_provider_none_uses_noop(self):
        svc = RerankerService(provider="none")
        assert svc.backend_name == "noop"
        assert svc.is_real is False

    def test_noop_rerank_preserves_order(self):
        svc = RerankerService(provider="none")
        hits = [_hit("p1", "A"), _hit("p2", "B"), _hit("p3", "C")]
        out = svc.rerank("irrelevant", hits)
        assert [h.path for h in out] == ["p1", "p2", "p3"]

    def test_noop_rerank_respects_limit(self):
        svc = RerankerService(provider="none")
        hits = [_hit("p1", "A"), _hit("p2", "B"), _hit("p3", "C")]
        out = svc.rerank("q", hits, limit=2)
        assert [h.path for h in out] == ["p1", "p2"]

    def test_empty_hits(self):
        svc = RerankerService(provider="none")
        assert svc.rerank("q", []) == []


class TestRerankWithFakeBackend:
    def test_reorders_by_score(self):
        svc = RerankerService(provider="none")
        hits = [
            _hit("p1", "Apples", "red fruit"),
            _hit("p2", "Bears", "forest animal"),
            _hit("p3", "Capitals", "cities"),
        ]
        fake = _FakeBackend(
            {
                ("q", _hit_to_document(hits[0])): 0.1,
                ("q", _hit_to_document(hits[1])): 0.9,
                ("q", _hit_to_document(hits[2])): 0.5,
            }
        )
        svc._backend = fake
        svc._backend_name = "fake"

        out = svc.rerank("q", hits)

        assert [h.path for h in out] == ["p2", "p3", "p1"]
        # Scores are written back onto the hits.
        assert out[0].score == pytest.approx(0.9)
        assert out[1].score == pytest.approx(0.5)
        assert out[2].score == pytest.approx(0.1)

    def test_stable_tiebreaker_preserves_input_order(self):
        svc = RerankerService(provider="none")
        hits = [_hit(f"p{i}", f"T{i}") for i in range(4)]
        # All ties.
        fake = _FakeBackend({("q", _hit_to_document(h)): 1.0 for h in hits})
        svc._backend = fake
        svc._backend_name = "fake"

        out = rerank_hits("q", hits, service=svc)
        assert [h.path for h in out] == [h.path for h in hits]


class TestDocumentFormatting:
    def test_title_plus_excerpt(self):
        hit = _hit("p1", "My Title", "body text")
        assert _hit_to_document(hit) == "My Title\nbody text"

    def test_title_only(self):
        hit = _hit("p1", "Just Title", "")
        assert _hit_to_document(hit) == "Just Title"


class TestCustomModelSpec:
    def test_v2_m3_is_declared(self):
        assert DEFAULT_RERANK_MODEL == "BAAI/bge-reranker-v2-m3"
        spec = _CUSTOM_FASTEMBED_MODELS[DEFAULT_RERANK_MODEL]
        assert "onnx-community" in str(spec["hf_repo"])
        assert str(spec["model_file"]).endswith(".onnx")

    def test_service_scorer_routes_through_backend(self):
        """``RerankerService.score`` must delegate to the bound backend."""
        svc = RerankerService(provider="none")
        fake = _FakeBackend({("q", "d1"): 0.7, ("q", "d2"): 0.1})
        svc._backend = fake
        svc._backend_name = "fake"

        assert svc.score("q", ["d1", "d2"]) == [0.7, 0.1]
        assert fake.calls == [("q", ["d1", "d2"])]

    def test_model_file_env_override(self, monkeypatch):
        """``MARVIN_RERANK_MODEL_FILE`` overrides the registered ONNX file.

        We don't load the model — only verify the override is forwarded
        to fastembed's ``add_custom_model`` call. This lets GPU users
        switch the int8 default to the FP16 variant without code changes.
        """
        from marvin.reranker import FastEmbedRerankerBackend

        captured: dict[str, object] = {}

        class _FakeCrossEncoder:
            @staticmethod
            def list_supported_models():
                return []

            @staticmethod
            def add_custom_model(*, model, sources, model_file, **kwargs):
                captured["model"] = model
                captured["model_file"] = model_file
                captured["hf"] = sources.hf

            def __init__(self, model_name: str) -> None:
                captured["init_model"] = model_name

        class _FakeModelSource:
            def __init__(self, *, hf: str) -> None:
                self.hf = hf

        import sys
        import types

        fake_ce = types.SimpleNamespace(TextCrossEncoder=_FakeCrossEncoder)
        fake_md = types.SimpleNamespace(ModelSource=_FakeModelSource)
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", fake_ce)
        monkeypatch.setitem(sys.modules, "fastembed.common.model_description", fake_md)

        # 1. Default (no env var): use the spec's int8 quantised file.
        monkeypatch.delenv("MARVIN_RERANK_MODEL_FILE", raising=False)
        FastEmbedRerankerBackend(model_name=DEFAULT_RERANK_MODEL)
        assert captured["model_file"] == "onnx/model_quantized.onnx"

        # 2. With the env var set: the FP16 variant is forwarded instead.
        captured.clear()
        monkeypatch.setenv("MARVIN_RERANK_MODEL_FILE", "onnx/model_fp16.onnx")
        FastEmbedRerankerBackend(model_name=DEFAULT_RERANK_MODEL)
        assert captured["model_file"] == "onnx/model_fp16.onnx"

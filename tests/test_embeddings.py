"""Unit tests for ``marvin.embeddings`` provider selection and warnings."""

from __future__ import annotations

import logging

import pytest

from marvin import embeddings as embeddings_module
from marvin.embeddings import EmbeddingService, HashEmbeddingBackend


class TestExplicitHashProvider:
    """``provider="hash"`` is an intentional choice; do not warn."""

    def test_no_warnings(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="marvin.embeddings")
        svc = EmbeddingService(provider="hash", dimensions=8)
        svc.embed_text("hello world")
        assert svc.backend_name == "hash"
        assert caplog.records == []


class TestAutoFallbackEmitsWarnings:
    """``provider="auto"`` + FastEmbed import/load failure must be loud."""

    def test_logs_two_warnings_on_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated fastembed failure")

        monkeypatch.setattr(embeddings_module, "FastEmbedBackend", boom)
        caplog.set_level(logging.WARNING, logger="marvin.embeddings")

        svc = EmbeddingService(provider="auto", dimensions=8)
        svc.embed_text("hello world")

        assert svc.backend_name == "hash"
        assert isinstance(svc._backend, HashEmbeddingBackend)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert "FastEmbed initialisation failed" in warnings[0].getMessage()
        assert "simulated fastembed failure" in warnings[0].getMessage()
        assert "HashEmbeddingBackend" in warnings[1].getMessage()
        assert "BM25-only" in warnings[1].getMessage()

    def test_warning_emitted_only_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated fastembed failure")

        monkeypatch.setattr(embeddings_module, "FastEmbedBackend", boom)
        caplog.set_level(logging.WARNING, logger="marvin.embeddings")

        svc = EmbeddingService(provider="auto", dimensions=8)
        svc.embed_text("first")
        svc.embed_text("second")
        svc.embed_text("third")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2


class TestExplicitFastembedProviderRaises:
    """``provider="fastembed"`` must propagate failures, not silently fall back."""

    def test_re_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated fastembed failure")

        monkeypatch.setattr(embeddings_module, "FastEmbedBackend", boom)
        caplog.set_level(logging.WARNING, logger="marvin.embeddings")

        svc = EmbeddingService(provider="fastembed", dimensions=8)
        with pytest.raises(RuntimeError, match="simulated fastembed failure"):
            svc.embed_text("hello")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == []


class TestLoadedBackendName:
    """``loaded_backend_name`` reports without forcing initialisation."""

    def test_none_before_use(self) -> None:
        svc = EmbeddingService(provider="hash", dimensions=8)
        assert svc.loaded_backend_name is None

    def test_set_after_use(self) -> None:
        svc = EmbeddingService(provider="hash", dimensions=8)
        svc.embed_text("warm up")
        assert svc.loaded_backend_name == "hash"

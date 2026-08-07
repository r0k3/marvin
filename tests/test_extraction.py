"""Entity-extraction model resolution (langextract wants bare model names)."""

from __future__ import annotations

import pytest

from marvin.extraction import _resolve_extract_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in ("MARVIN_EXTRACT_MODEL", "LANGEXTRACT_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_default_is_bare_local_model():
    # langextract's registry matches ^qwen — an ollama/ prefix breaks it.
    assert _resolve_extract_model() == "qwen3.6:35b-a3b-q4_K_M"


def test_litellm_style_override_is_normalized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MARVIN_EXTRACT_MODEL", "ollama/llama3.3:70b")
    assert _resolve_extract_model() == "llama3.3:70b"


def test_remote_override_untouched(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MARVIN_EXTRACT_MODEL", "gpt-5.4")
    assert _resolve_extract_model() == "gpt-5.4"


def test_api_key_switches_default_to_remote(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert _resolve_extract_model() == "gpt-5.4"

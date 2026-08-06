"""Correction detector: high-precision deterministic promotion signals."""

from __future__ import annotations

import pytest

from marvin.promotion import detect_correction


@pytest.mark.parametrize(
    ("text", "expected_signal"),
    [
        ("No, the API uses cursor pagination.", "leading-no"),
        ("wrong: the timeout is 30s not 10s", "leading-no"),
        ("That's not right — we deploy from CI only.", "thats-wrong"),
        ("that's incorrect, look at the config", "thats-wrong"),
        ("Actually, it's PostgreSQL, not MySQL.", "actually-is"),
        ("actually we use uv here", "actually-is"),
        ("I said the staging cluster, not prod.", "i-said"),
        ("I already told you the port is 8421.", "i-said"),
        ("Correction: the vault lives under ~/.marvin_vault.", "explicit-correction"),
        ("We no longer use pip in this repo.", "no-longer"),
        ("we don't use docker compose anymore", "no-longer"),
        ("Stop using the legacy endpoint.", "stop-using"),
        ("The default should be sse, not stdio.", "should-be-not"),
    ],
)
def test_detects_corrections(text: str, expected_signal: str):
    assert expected_signal in detect_correction(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Add validation to the signup form.",
        "Can you refactor the retry logic?",
        "What does the consolidation engine do?",
        "Note that the tests are slow on CI.",  # "not" inside "Note" must not fire
        "run the linter and fix warnings",
        "The knot should be tied twice.",  # 'no' inside a word
    ],
)
def test_ignores_ordinary_prompts(text: str):
    assert detect_correction(text) == []

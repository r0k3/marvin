"""Unit tests for ``marvin.eval.reader``.

The reader's LLM call is stubbed by monkeypatching the module-level
``completion`` symbol, so these tests never touch ollama. They pin the
context format (JSON), the answer parsing (ANSWER: marker), the
abstention clause, and the error-handling contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from marvin.eval.reader import (
    ReaderContextItem,
    ReaderEngine,
    _build_prompt,
    _parse_answer,
    build_reader_context,
)


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _items(n: int) -> list[ReaderContextItem]:
    return [
        ReaderContextItem(
            kind="episodic",
            session_id=f"s{i}",
            date=f"2023/05/{i + 1:02d} (Mon) 10:00",
            body=f"user: hello {i}\nassistant: hi {i}",
        )
        for i in range(n)
    ]


class TestBuildReaderContext:
    def test_emits_json_array_in_rank_order(self):
        ctx = build_reader_context(_items(2))
        parsed = json.loads(ctx)
        assert [e["index"] for e in parsed] == [1, 2]
        assert parsed[0]["type"] == "episodic"
        assert parsed[0]["date"].startswith("2023/05/01")
        assert "hello 0" in parsed[0]["content"]

    def test_truncates_long_entries(self):
        item = ReaderContextItem(kind="episodic", session_id="s", date="", body="x" * 5000)
        ctx = build_reader_context([item], max_entry_chars=100)
        parsed = json.loads(ctx)
        assert parsed[0]["content"].endswith("[…]")
        assert len(parsed[0]["content"]) < 200

    def test_budget_caps_entry_count_but_keeps_first(self):
        # Tiny budget: only the first entry survives (we never drop entry 1).
        ctx = build_reader_context(_items(5), max_context_chars=10)
        parsed = json.loads(ctx)
        assert len(parsed) == 1
        assert parsed[0]["index"] == 1


class TestParseAnswer:
    def test_extracts_after_marker(self):
        assert _parse_answer("NOTES:\n[1] foo\nANSWER: Paris") == "Paris"

    def test_uses_last_marker(self):
        assert _parse_answer("ANSWER: draft\nmore\nANSWER: final") == "final"

    def test_falls_back_to_whole_text_without_marker(self):
        assert _parse_answer("  just an answer  ") == "just an answer"

    def test_case_insensitive_marker(self):
        assert _parse_answer("answer: lower") == "lower"


class TestBuildPrompt:
    def test_contains_abstention_clause_context_and_question(self):
        ctx = build_reader_context(_items(1))
        prompt = _build_prompt("What is X?", ctx, "2024/01/01 (Mon) 09:00")
        assert "I don't have enough information" in prompt
        assert "What is X?" in prompt
        assert "2024/01/01" in prompt
        assert "hello 0" in prompt  # context body made it in
        assert "ANSWER:" in prompt


class TestReaderEngineAnswer:
    def test_parses_answer_and_marks_no_error(self, monkeypatch):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _fake_response("NOTES:\n[1] relevant\nANSWER: 42")

        monkeypatch.setattr("marvin.eval.reader.completion", fake_completion)
        engine = ReaderEngine(model="ollama/test", api_base="http://x")
        result = engine.answer("q?", build_reader_context(_items(1)), question_date="2024/01/01")

        assert result.answer == "42"
        assert result.error is False
        assert captured["model"] == "ollama/test"
        assert captured["temperature"] == 0.0
        assert captured["api_base"] == "http://x"

    def test_ollama_disables_thinking_by_default(self, monkeypatch):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _fake_response("ANSWER: ok")

        monkeypatch.setattr("marvin.eval.reader.completion", fake_completion)
        ReaderEngine().answer("q?", "[]")  # default model is ollama/...
        assert captured.get("think") is False
        assert "/no_think" not in captured["messages"][0]["content"]

    def test_think_true_keeps_thinking(self, monkeypatch):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _fake_response("ANSWER: ok")

        monkeypatch.setattr("marvin.eval.reader.completion", fake_completion)
        ReaderEngine(think=True).answer("q?", "[]")
        assert "think" not in captured

    def test_non_ollama_reader_omits_think(self, monkeypatch):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _fake_response("ANSWER: ok")

        monkeypatch.setattr("marvin.eval.reader.completion", fake_completion)
        ReaderEngine(model="anthropic/claude-sonnet-4-6").answer("q?", "[]")
        assert "think" not in captured

    def test_engine_error_returns_empty_error_result(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("ollama down")

        monkeypatch.setattr("marvin.eval.reader.completion", boom)
        result = ReaderEngine().answer("q?", "[]")
        assert result.error is True
        assert result.answer == ""

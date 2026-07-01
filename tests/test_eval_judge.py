"""Unit tests for ``marvin.eval.judge``.

Verifies the per-type prompt selection matches the official LongMemEval
``get_anscheck_prompt`` dispatch, the ``_abs`` abstention routing, and the
yes/no parsing contract. The LLM call is monkeypatched.
"""

from __future__ import annotations

from types import SimpleNamespace

from marvin.eval.judge import Judge, build_judge_prompt, is_abstention


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class TestIsAbstention:
    def test_detects_abs_in_question_id(self):
        assert is_abstention("abc_abs", "multi-session") is True

    def test_detects_abs_suffix_on_type(self):
        assert is_abstention("abc", "multi-session_abs") is True

    def test_plain_question_is_not_abstention(self):
        assert is_abstention("abc", "multi-session") is False


class TestBuildJudgePrompt:
    def _build(self, qtype: str, qid: str = "q1", answer: str = "ANS") -> str:
        return build_judge_prompt(
            question="Q", answer=answer, hypothesis="H", question_type=qtype, question_id=qid
        )

    def test_temporal_adds_off_by_one_tolerance(self):
        prompt = self._build("temporal-reasoning")
        assert "off-by-one" in prompt

    def test_knowledge_update_tolerates_stale_with_update(self):
        prompt = self._build("knowledge-update")
        assert "previous information along with an updated answer" in prompt

    def test_preference_is_rubric_based(self):
        prompt = self._build("single-session-preference")
        assert "Rubric:" in prompt
        assert "rubric for desired personalized response" in prompt

    def test_multi_session_uses_base_prompt(self):
        prompt = self._build("multi-session")
        assert "contains the correct answer" in prompt
        assert "Rubric:" not in prompt
        assert "off-by-one" not in prompt

    def test_abstention_overrides_base_type(self):
        # Even a temporal question, if it's an _abs id, uses the abstention prompt.
        prompt = self._build("temporal-reasoning", qid="q1_abs")
        assert "unanswerable" in prompt
        assert "Explanation:" in prompt
        assert "off-by-one" not in prompt


class TestJudgeVerdict:
    def test_yes_is_correct(self, monkeypatch):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _fake_response("yes")

        monkeypatch.setattr("marvin.eval.judge.completion", fake_completion)
        verdict = Judge(model="anthropic/claude-sonnet-4-6").judge(
            question="Q",
            answer="A",
            hypothesis="H",
            question_type="multi-session",
            question_id="q1",
        )
        assert verdict is True
        assert captured["max_tokens"] == 10
        assert captured["temperature"] == 0.0

    def test_no_is_incorrect(self, monkeypatch):
        monkeypatch.setattr("marvin.eval.judge.completion", lambda **k: _fake_response("No."))
        verdict = Judge().judge(
            question="Q", answer="A", hypothesis="H", question_type="multi-session", question_id="q"
        )
        assert verdict is False

    def test_uppercase_yes_parsed(self, monkeypatch):
        monkeypatch.setattr("marvin.eval.judge.completion", lambda **k: _fake_response("YES"))
        verdict = Judge().judge(
            question="Q", answer="A", hypothesis="H", question_type="multi-session", question_id="q"
        )
        assert verdict is True

    def test_judge_error_returns_none(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("api down")

        monkeypatch.setattr("marvin.eval.judge.completion", boom)
        verdict = Judge().judge(
            question="Q", answer="A", hypothesis="H", question_type="multi-session", question_id="q"
        )
        assert verdict is None

    def test_ollama_judge_disables_thinking_remote_stays_faithful(self, monkeypatch):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _fake_response("yes")

        monkeypatch.setattr("marvin.eval.judge.completion", fake_completion)

        Judge(model="ollama/qwen3.6:35b-a3b-q4_K_M").judge(
            question="Q", answer="A", hypothesis="H", question_type="multi-session", question_id="q"
        )
        assert captured.get("think") is False
        assert "/no_think" not in captured["messages"][0]["content"]

        # Remote judge: exact official prompt, max_tokens=10, no think kwarg.
        captured.clear()
        Judge(model="anthropic/claude-sonnet-4-6").judge(
            question="Q", answer="A", hypothesis="H", question_type="multi-session", question_id="q"
        )
        assert "think" not in captured
        assert captured["max_tokens"] == 10

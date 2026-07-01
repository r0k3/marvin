"""Unit tests for K-line templates (``marvin.klines``) and the selector.

Covers render/parse round-trip (including the structured trigger dimensions),
the weighted scoring formula with its intent hard-gate, the effectiveness EMA,
and the service-level register / match / record-use / session-wiring behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marvin.config import MarvinSettings
from marvin.klines import (
    KLineTemplate,
    MatchSignal,
    next_effectiveness,
    parse_template_body,
    render_template_body,
    score_template,
)
from marvin.models import MemoryKind
from marvin.service import MarvinService

# ---------------------------------------------------------------------------
# render / parse
# ---------------------------------------------------------------------------


class TestRenderParseRoundTrip:
    def test_full_template_round_trips(self) -> None:
        body = render_template_body(
            intents=("deploy",),
            styles=("terse",),
            entity_types=("service",),
            trigger_phrases=("deploy to staging", "promote build"),
            slots=("service_name",),
            plan=("Run tests", "Apply manifests"),
            failure_modes=("tests are red",),
            intro="Use this when shipping to staging.",
        )
        parsed = parse_template_body(title="Deploy", body=body)
        assert parsed is not None
        assert parsed.intents == ("deploy",)
        assert parsed.styles == ("terse",)
        assert parsed.entity_types == ("service",)
        assert parsed.trigger_phrases == ("deploy to staging", "promote build")
        assert parsed.slots == ("service_name",)
        assert parsed.plan == ("Run tests", "Apply manifests")
        assert parsed.failure_modes == ("tests are red",)
        assert parsed.is_complete()

    def test_empty_sections_omitted(self) -> None:
        body = render_template_body(trigger_phrases=("foo",), plan=("Step 1",))
        assert "## Slots" not in body
        assert "## Failure Modes" not in body
        assert "## Intents" not in body
        assert "## Triggers" in body
        assert "## Plan" in body

    def test_non_template_body_returns_none(self) -> None:
        body = "## Notes\nsome prose, no triggers and no plan"
        assert parse_template_body(title="Generic", body=body) is None

    def test_plan_tolerates_bullet_style(self) -> None:
        body = "## Triggers\n- t1\n\n## Plan\n- step a\n- step b"
        parsed = parse_template_body(title="Mixed", body=body)
        assert parsed is not None
        assert parsed.plan == ("step a", "step b")


# ---------------------------------------------------------------------------
# selector scoring
# ---------------------------------------------------------------------------


class TestScoreTemplate:
    def _kw(self, *triggers: str) -> KLineTemplate:
        return KLineTemplate(title="t", trigger_phrases=triggers, plan=("step",))

    def test_keyword_full_coverage_scores_full_bonus(self) -> None:
        score, matched = score_template(
            self._kw("alpha", "beta"), MatchSignal(text="alpha and beta appear")
        )
        assert score == pytest.approx(0.25)
        assert set(matched) == {"alpha", "beta"}

    def test_keyword_partial_coverage_is_proportional(self) -> None:
        score, matched = score_template(
            self._kw("alpha", "beta", "gamma"), MatchSignal(text="only alpha appears")
        )
        assert score == pytest.approx(0.25 / 3)
        assert matched == ("alpha",)

    def test_no_match_scores_zero(self) -> None:
        score, matched = score_template(self._kw("alpha"), MatchSignal(text="nothing here"))
        assert score == 0.0
        assert matched == ()

    def test_match_is_case_insensitive(self) -> None:
        score, matched = score_template(
            self._kw("Deploy To Staging"), MatchSignal(text="please DEPLOY TO STAGING tonight")
        )
        assert score == pytest.approx(0.25)
        assert matched == ("Deploy To Staging",)

    def test_no_triggers_never_fires(self) -> None:
        score, _ = score_template(KLineTemplate(title="t", plan=("s",)), MatchSignal(text="x"))
        assert score == 0.0

    def test_intent_is_weighted_highest(self) -> None:
        tmpl = KLineTemplate(title="t", intents=("debug",), plan=("s",))
        assert score_template(tmpl, MatchSignal(intent="debug"))[0] == pytest.approx(0.5)

    def test_intent_is_a_hard_gate(self) -> None:
        # A non-matching intent zeroes the score even when every other
        # dimension would otherwise match.
        tmpl = KLineTemplate(
            title="t",
            intents=("debug",),
            styles=("terse",),
            entity_types=("service",),
            trigger_phrases=("alpha",),
            plan=("s",),
        )
        signal = MatchSignal(
            intent="chitchat", styles=("terse",), entity_types=("service",), text="alpha"
        )
        assert score_template(tmpl, signal)[0] == 0.0

    def test_style_and_entity_type_each_add_a_quarter(self) -> None:
        tmpl = KLineTemplate(title="t", styles=("terse",), entity_types=("service",), plan=("s",))
        assert score_template(tmpl, MatchSignal(styles=("terse",)))[0] == pytest.approx(0.25)
        assert score_template(tmpl, MatchSignal(entity_types=("service",)))[0] == pytest.approx(
            0.25
        )

    def test_all_dimensions_combine(self) -> None:
        tmpl = KLineTemplate(
            title="t",
            intents=("debug",),
            styles=("terse",),
            entity_types=("service",),
            trigger_phrases=("timeout",),
            plan=("s",),
        )
        signal = MatchSignal(
            intent="debug", styles=("terse",), entity_types=("service",), text="a timeout occurred"
        )
        score, matched = score_template(tmpl, signal)
        assert score == pytest.approx(1.25)  # 0.5 + 0.25 + 0.25 + 0.25 keyword bonus
        assert matched == ("timeout",)


def test_next_effectiveness_is_an_ema() -> None:
    assert next_effectiveness(0.0, success=True) == pytest.approx(0.3)
    assert next_effectiveness(0.3, success=False) == pytest.approx(0.21)


# ---------------------------------------------------------------------------
# service-level register + match
# ---------------------------------------------------------------------------


def _service(tmp_path: Path) -> MarvinService:
    return MarvinService(
        MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
    )


class TestServiceTemplateAPI:
    def test_register_then_match_round_trip(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            service.register_template(
                title="Deploy to staging",
                trigger_phrases=["deploy to staging", "promote build"],
                plan=["Validate tests", "Apply manifests"],
                slots=["service_name"],
                failure_modes=["tests are red"],
            )
            matches = service.match_template("deploy to staging tonight")
            assert len(matches) == 1
            match = matches[0]
            assert match.template.title == "Deploy to staging"
            assert match.score == pytest.approx(0.125)  # 1 of 2 keywords -> 0.25 * 0.5
            assert match.matched_phrases == ("deploy to staging",)
            assert match.template.plan == ("Validate tests", "Apply manifests")
        finally:
            service.close()

    def test_intent_gated_template_needs_matching_intent(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            service.register_template(
                title="Triage incident",
                intents=["incident"],
                trigger_phrases=["outage"],
                plan=["Page on-call"],
            )
            # Keyword present but no intent supplied -> hard gate drops it.
            assert service.match_template("there is an outage") == []
            # With the intent it fires.
            matches = service.match_template("there is an outage", intent="incident")
            assert [m.template.title for m in matches] == ["Triage incident"]
        finally:
            service.close()

    def test_non_template_procedural_notes_are_ignored(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            service.store_procedure(title="Bootstrap a Python repo", steps=["uv init", "add tests"])
            service.register_template(
                title="Cut a release",
                trigger_phrases=["cut a release", "tag a release"],
                plan=["Update changelog", "Tag commit"],
            )
            matches = service.match_template("time to cut a release")
            assert [m.template.title for m in matches] == ["Cut a release"]
        finally:
            service.close()

    def test_higher_coverage_outranks_lower(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            service.register_template(
                title="A: two-thirds match",
                trigger_phrases=["alpha", "beta", "gamma"],
                plan=["plan A"],
            )
            service.register_template(
                title="B: full match", trigger_phrases=["delta", "epsilon"], plan=["plan B"]
            )
            matches = service.match_template("context mentions delta and epsilon and also alpha")
            assert [m.template.title for m in matches] == ["B: full match", "A: two-thirds match"]
            assert matches[0].score == pytest.approx(0.25)
            assert matches[1].score == pytest.approx(0.25 / 3)
        finally:
            service.close()


class TestAdaptiveUtility:
    def test_record_use_updates_usage_and_effectiveness(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            service.register_template(title="Deploy", trigger_phrases=["deploy"], plan=["go"])
            service.record_template_use("Deploy", success=True)
            note = service.vault.find_note(title="Deploy", kind=MemoryKind.PROCEDURAL)
            assert note is not None
            assert note.metadata.usage_count == 1
            assert note.metadata.effectiveness == pytest.approx(0.3)
            # surfaced on the match result too
            match = service.match_template("deploy now")[0]
            assert match.usage_count == 1
            assert match.effectiveness == pytest.approx(0.3)
        finally:
            service.close()

    def test_effectiveness_breaks_score_ties(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            service.register_template(title="Aye", trigger_phrases=["ship"], plan=["a"])
            service.register_template(title="Bee", trigger_phrases=["ship"], plan=["b"])
            service.record_template_use("Bee", success=True)  # equal score, higher utility
            matches = service.match_template("time to ship")
            assert [m.template.title for m in matches] == ["Bee", "Aye"]
        finally:
            service.close()

    def test_prepare_session_surfaces_matched_template(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            service.register_template(
                title="Debug timeouts",
                trigger_phrases=["timeout"],
                plan=["Check logs", "Reproduce", "Bisect"],
            )
            ctx = service.prepare_session(task="investigating a timeout in the API")
            assert any("Debug timeouts" in line for line in ctx.guidance)
        finally:
            service.close()

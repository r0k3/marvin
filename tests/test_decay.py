"""Unit tests for :mod:`marvin.decay`.

Covers the multiplicative-boost contract documented in the module:

* ``decay_weight=0`` short-circuits to a no-op multiplier.
* ``age=0`` saturates the boost at ``1 + decay_weight``.
* ``age >> half_life`` decays toward ``1.0``.
* Future-dated notes (negative age) are clamped to ``age=0`` rather than
  exploding into a > ``1 + decay_weight`` multiplier.
* Half-life behaves as expected: ``age = half_life`` halves the boost
  contribution.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from marvin.decay import (
    age_days_between,
    freshness_boost,
    parse_note_timestamp,
)


class TestFreshnessBoost:
    def test_zero_weight_is_noop(self) -> None:
        assert freshness_boost(0.0, half_life_days=30.0, decay_weight=0.0) == 1.0
        assert (
            freshness_boost(1000.0, half_life_days=30.0, decay_weight=0.0) == 1.0
        )

    def test_zero_age_saturates(self) -> None:
        boost = freshness_boost(0.0, half_life_days=30.0, decay_weight=0.5)
        assert boost == pytest.approx(1.5)

    def test_half_life_halves_extra(self) -> None:
        boost = freshness_boost(30.0, half_life_days=30.0, decay_weight=0.5)
        assert boost == pytest.approx(1.0 + 0.5 * math.exp(-1.0))

    def test_far_past_decays_to_one(self) -> None:
        boost = freshness_boost(10_000.0, half_life_days=30.0, decay_weight=0.5)
        assert boost == pytest.approx(1.0, abs=1e-9)

    def test_negative_age_clamped_to_max(self) -> None:
        boost = freshness_boost(-5.0, half_life_days=30.0, decay_weight=0.5)
        assert boost == pytest.approx(1.5)

    def test_negative_weight_is_noop(self) -> None:
        assert (
            freshness_boost(0.0, half_life_days=30.0, decay_weight=-1.0) == 1.0
        )


class TestAgeDaysBetween:
    def test_simple_difference(self) -> None:
        now = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
        old = datetime(2025, 1, 5, 12, 0, tzinfo=UTC)
        assert age_days_between(now, old) == pytest.approx(5.0)

    def test_negative_when_future(self) -> None:
        now = datetime(2025, 1, 5, 12, 0, tzinfo=UTC)
        future = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
        assert age_days_between(now, future) == pytest.approx(-5.0)


class TestParseNoteTimestamp:
    def test_iso_round_trip(self) -> None:
        ts = datetime(2025, 1, 10, 12, 30, tzinfo=UTC)
        parsed = parse_note_timestamp(ts.isoformat())
        assert parsed == ts

    def test_none_and_empty(self) -> None:
        assert parse_note_timestamp(None) is None
        assert parse_note_timestamp("") is None

    def test_garbage_returns_none(self) -> None:
        assert parse_note_timestamp("not-a-date") is None

    def test_offset_preserved(self) -> None:
        ts = datetime(
            2025, 1, 10, 12, 30, tzinfo=UTC
        ) - timedelta(hours=3)
        parsed = parse_note_timestamp(ts.isoformat())
        assert parsed == ts

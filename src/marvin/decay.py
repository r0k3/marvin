"""Time-aware retrieval boosting (a.k.a. memory decay).

The hybrid search pipeline ranks notes by relevance, but two notes can
be equally relevant lexically/semantically while one is from yesterday
and the other from two years ago. For long-running chat memory ("what
did I just say about X?", "I told you yesterday that...") the more
recent one is almost always what the user wants. RRF on its own has no
way to express that preference because the rankers are time-blind.

This module supplies a *freshness boost* — not a staleness penalty —
that fits cleanly on top of the existing RRF scores:

    final_score = base_rrf * (1 + decay_weight * freshness)
    freshness = exp(-age_days / half_life_days)            in (0, 1]

Why a multiplicative boost instead of an additive RRF stream or a
multiplicative penalty?

* **Boost, not penalty.** ``exp(-age / h)`` flattens to zero as age
  grows, so the boost factor approaches ``1.0``. An old note keeps its
  full base score; a fresh note gets up to ``1 + decay_weight`` times
  that base. Stale-but-uniquely-relevant notes are never demoted below
  their honest RRF rank.
* **Multiplicative.** The base RRF score has no fixed scale — it's
  ``sum(1/(k+rank))`` over fused streams. Adding a recency-only RRF
  stream would dilute the relevance signal in a regime where retrieval
  quality is already near ceiling (LongMemEval-S hybrid+rerank ~99.6%
  R@5). A multiplicative boost preserves the existing ordering except
  where the magnitude of the boost flips two close rankings.
* **No physical units leak through.** ``freshness`` is bounded in
  ``(0, 1]`` regardless of the absolute timestamp scale, so the same
  ``decay_weight`` works for vaults that span a week or a decade.

The half-life is an explicit choice rather than a property of the data:
choose ~30 days for chat memory dominated by recent context, ~365 days
for a knowledge base where stale-but-true is still useful.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Final

_SECONDS_PER_DAY: Final[float] = 86_400.0


def freshness_boost(
    age_days: float,
    *,
    half_life_days: float,
    decay_weight: float,
) -> float:
    """Multiplier in ``[1.0, 1.0 + decay_weight]`` for a note of given age.

    Returns exactly ``1.0`` for non-positive ``decay_weight`` (no-op),
    and clamps ``age_days`` at 0 so future-dated notes (``updated_at``
    after the query time) get the maximum boost rather than an
    explosive multiplier.
    """
    if decay_weight <= 0.0:
        return 1.0
    age = max(0.0, age_days)
    freshness = math.exp(-age / half_life_days)
    return 1.0 + decay_weight * freshness


def age_days_between(query_time: datetime, note_time: datetime) -> float:
    """Number of days from ``note_time`` to ``query_time``.

    Both timestamps must be timezone-aware (Marvin uses UTC throughout
    via :func:`marvin.models.utc_now`). Result is non-negative when
    ``query_time >= note_time``.
    """
    delta = query_time - note_time
    return delta.total_seconds() / _SECONDS_PER_DAY


def parse_note_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp from the SQLite ``notes.updated_at`` column.

    The schema stores ``updated_at`` as ``isoformat()`` of a
    timezone-aware UTC datetime. Returns ``None`` for missing/empty
    values so callers can skip the boost without raising.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None

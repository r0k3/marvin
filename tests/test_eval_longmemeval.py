"""Unit tests for ``marvin.eval.longmemeval``.

No network access. Embedding uses the deterministic ``hash`` backend so
tests run in well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marvin.embeddings import EmbeddingService
from marvin.eval.longmemeval import (
    ABSTENTION_TYPES,
    LongMemEvalEntry,
    format_summary,
    load_dataset,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_any_at_k,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Pure metric primitives.
# ---------------------------------------------------------------------------


class TestRecallAnyAtK:
    def test_hit_in_top_k(self):
        assert recall_any_at_k(["a", "b", "c"], ["b"], 5) == 1.0

    def test_miss_outside_top_k(self):
        assert recall_any_at_k(["a", "b", "c", "d", "e", "f"], ["f"], 5) == 0.0

    def test_multiple_gold_any_match(self):
        assert recall_any_at_k(["x", "y"], ["nope", "y"], 2) == 1.0

    def test_no_gold(self):
        assert recall_any_at_k(["a", "b"], [], 5) == 0.0

    def test_empty_retrieved(self):
        assert recall_any_at_k([], ["x"], 5) == 0.0


class TestNDCG:
    def test_perfect_first_position(self):
        assert ndcg_at_k(["g", "x", "y"], ["g"], 10) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k(["x", "y"], ["g"], 10) == 0.0

    def test_no_gold_returns_zero(self):
        assert ndcg_at_k(["x", "y"], [], 10) == 0.0

    def test_decay_with_position(self):
        first = ndcg_at_k(["g", "x"], ["g"], 10)
        second = ndcg_at_k(["x", "g"], ["g"], 10)
        assert first > second > 0.0


class TestMRR:
    def test_rank_one(self):
        assert mean_reciprocal_rank(["g", "x"], ["g"]) == 1.0

    def test_rank_three(self):
        assert mean_reciprocal_rank(["a", "b", "g"], ["g"]) == pytest.approx(1 / 3)

    def test_no_match(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["g"]) == 0.0


# ---------------------------------------------------------------------------
# Dataset loader.
# ---------------------------------------------------------------------------


def _write_synthetic_dataset(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def _make_entry(qid: str, qtype: str = "single-session-user") -> dict:
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": f"q for {qid}",
        "answer": "",
        "answer_session_ids": [f"s_{qid}_gold"],
        "haystack_session_ids": [f"s_{qid}_a", f"s_{qid}_gold"],
        "haystack_sessions": [
            [{"role": "user", "content": "filler"}],
            [{"role": "user", "content": "the gold answer is here"}],
        ],
    }


class TestLoadDataset:
    def test_filters_abstention_by_default(self, tmp_path: Path):
        path = tmp_path / "ds.json"
        _write_synthetic_dataset(
            path,
            [
                _make_entry("q1", "single-session-user"),
                _make_entry("q2", "single-session-user_abs"),  # abstention
                _make_entry("q3", "multi-session"),
            ],
        )
        entries = load_dataset(path)
        assert [e.question_id for e in entries] == ["q1", "q3"]

    def test_include_abstention(self, tmp_path: Path):
        path = tmp_path / "ds.json"
        _write_synthetic_dataset(
            path,
            [
                _make_entry("q1", "multi-session"),
                _make_entry("q2", "multi-session_abs"),
            ],
        )
        entries = load_dataset(path, include_abstention=True)
        assert len(entries) == 2

    def test_abstention_set_matches_published(self):
        # Sanity check that we don't accidentally drift from the published
        # set. agentmemory publishes 4 abstention question types.
        assert len(ABSTENTION_TYPES) == 4
        for qt in ABSTENTION_TYPES:
            assert qt.endswith("_abs")


# ---------------------------------------------------------------------------
# End-to-end benchmark on a tiny synthetic haystack.
#
# The protocol is: each question has a unique "needle" word planted in a
# single haystack session. BM25 over the question text must surface that
# session at rank 1.
# ---------------------------------------------------------------------------


def _planted_entry(qid: str, needle: str, distractors: int = 4) -> LongMemEvalEntry:
    sessions: list[list[dict[str, object]]] = []
    session_ids: list[str] = []
    for i in range(distractors):
        session_ids.append(f"{qid}_distractor_{i}")
        sessions.append(
            [
                {"role": "user", "content": "weather chitchat"},
                {"role": "assistant", "content": "today is sunny"},
            ]
        )
    gold_id = f"{qid}_gold"
    session_ids.append(gold_id)
    sessions.append(
        [
            {"role": "user", "content": f"Remember the secret codename {needle}."},
            {"role": "assistant", "content": "Got it."},
        ]
    )
    return LongMemEvalEntry(
        question_id=qid,
        question_type="single-session-user",
        question=f"What was the secret codename {needle}?",
        answer_session_ids=[gold_id],
        haystack_session_ids=session_ids,
        haystack_sessions=sessions,
    )


@pytest.fixture
def hash_embedder() -> EmbeddingService:
    return EmbeddingService(provider="hash", dimensions=64)


class TestRunBenchmark:
    def test_bm25_perfect_recall_on_planted_needle(
        self, hash_embedder: EmbeddingService
    ):
        entries = [
            _planted_entry("q1", needle="zephyrcat"),
            _planted_entry("q2", needle="quantumlemon"),
            _planted_entry("q3", needle="flintdrake"),
        ]
        summary = run_benchmark(
            entries, mode="bm25", embedder=hash_embedder, top_k=5, progress=0
        )
        assert summary.questions == 3
        assert summary.recall_at_5 == 1.0
        assert summary.recall_at_10 == 1.0
        assert summary.mrr == 1.0
        assert summary.ndcg_at_10 == pytest.approx(1.0)
        assert "single-session-user" in summary.per_type
        assert summary.per_type["single-session-user"].count == 3

    def test_hybrid_runs_end_to_end(self, hash_embedder: EmbeddingService):
        entries = [_planted_entry("q1", needle="zephyrcat")]
        summary = run_benchmark(
            entries, mode="hybrid", embedder=hash_embedder, top_k=5, progress=0
        )
        assert summary.questions == 1
        # Hash backend is weak but BM25 dominates RRF here so the gold
        # session must still be retrieved.
        assert summary.recall_at_5 == 1.0

    def test_vector_runs_end_to_end(self, hash_embedder: EmbeddingService):
        entries = [_planted_entry("q1", needle="zephyrcat")]
        summary = run_benchmark(
            entries, mode="vector", embedder=hash_embedder, top_k=5, progress=0
        )
        assert summary.questions == 1
        # Just check the pipeline runs and produces some retrieval.
        assert len(summary.per_question[0].retrieved_session_ids) > 0

    def test_summary_serializes_to_json(self, hash_embedder: EmbeddingService):
        entries = [_planted_entry("q1", needle="zephyrcat")]
        summary = run_benchmark(
            entries, mode="bm25", embedder=hash_embedder, top_k=5, progress=0
        )
        payload = summary.model_dump(mode="json")
        # Round-trips through json without errors.
        rendered = json.dumps(payload)
        assert "recall_at_5" in rendered

    def test_format_summary_contains_key_fields(
        self, hash_embedder: EmbeddingService
    ):
        entries = [_planted_entry("q1", needle="zephyrcat")]
        summary = run_benchmark(
            entries, mode="bm25", embedder=hash_embedder, top_k=5, progress=0
        )
        text = format_summary(summary)
        assert "recall_any@5" in text
        assert "MRR" in text
        assert "single-session-user" in text

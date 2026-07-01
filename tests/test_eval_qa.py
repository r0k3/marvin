"""Tests for QA aggregation in ``marvin.eval.longmemeval``.

These exercise the pure rollup logic (no LLM, no retrieval): that the QA
summary counts verdicts/errors/abstention correctly, and that retrieval
metrics skip abstention (empty-gold) questions while QA accuracy includes
all judged questions.
"""

from __future__ import annotations

from marvin.embeddings import EmbeddingService
from marvin.eval.longmemeval import QuestionResult, _aggregate_qa, _summarize


def _qr(
    qid: str,
    qtype: str,
    gold: list[str],
    *,
    recall: float = 0.0,
    qa_correct: bool | None = None,
) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        question_type=qtype,
        retrieved_session_ids=[],
        gold_session_ids=gold,
        recall_at_5=recall,
        recall_at_10=recall,
        recall_at_20=recall,
        ndcg_at_10=recall,
        mrr=recall,
        latency_ms=1.0,
        qa_correct=qa_correct,
    )


def _scenario() -> list[QuestionResult]:
    return [
        _qr("q1", "multi-session", ["g1"], recall=1.0, qa_correct=True),
        _qr("q2", "multi-session", ["g2"], recall=1.0, qa_correct=False),
        _qr("q3", "temporal-reasoning", ["g3"], recall=0.0, qa_correct=None),  # judge error
        _qr("x_abs", "multi-session_abs", [], qa_correct=True),  # abstention, no gold
    ]


class TestAggregateQA:
    def test_disabled_returns_none(self):
        assert (
            _aggregate_qa(
                results=_scenario(),
                enabled=False,
                reader_model="r",
                judge_model="j",
                reader_top_k=10,
            )
            is None
        )

    def test_counts_verdicts_errors_and_abstention(self):
        qa = _aggregate_qa(
            results=_scenario(),
            enabled=True,
            reader_model="r",
            judge_model="j",
            reader_top_k=10,
        )
        assert qa is not None
        assert qa.questions == 3  # q1, q2, x_abs received a verdict
        assert qa.correct == 2  # q1, x_abs
        assert qa.errors == 1  # q3 had no verdict
        assert qa.abstention_questions == 1
        assert qa.abstention_correct == 1
        assert abs(qa.accuracy - 2 / 3) < 1e-9


class TestSummarizeRetrievalSkipsAbstention:
    def test_retrieval_excludes_abstention_qa_includes_all(self):
        emb = EmbeddingService(provider="hash")
        summary = _summarize(
            mode="hybrid",
            embedder=emb,
            reranker=None,
            rerank_depth=50,
            results=_scenario(),
            elapsed_seconds=1.0,
            qa_enabled=True,
            reader_model="r",
            judge_model="j",
            reader_top_k=10,
        )
        # Headline retrieval is over the 3 non-abstention questions only.
        assert summary.questions == 3
        assert abs(summary.recall_at_5 - 2 / 3) < 1e-9
        # QA accuracy spans all judged questions including abstention.
        assert summary.qa is not None
        assert summary.qa.questions == 3
        assert abs(summary.qa.accuracy - 2 / 3) < 1e-9

    def test_per_type_qa_and_abstention_breakdown(self):
        emb = EmbeddingService(provider="hash")
        summary = _summarize(
            mode="hybrid",
            embedder=emb,
            reranker=None,
            rerank_depth=50,
            results=_scenario(),
            elapsed_seconds=1.0,
            qa_enabled=True,
            reader_model="r",
            judge_model="j",
            reader_top_k=10,
        )
        ms = summary.per_type["multi-session"]
        assert ms.qa_accuracy == 0.5  # q1 correct, q2 wrong
        assert ms.qa_count == 2

        tr = summary.per_type["temporal-reasoning"]
        assert tr.qa_accuracy is None  # only an errored (unverdicted) question
        assert tr.qa_count == 0

        ab = summary.per_type["multi-session_abs"]
        assert ab.qa_accuracy == 1.0
        assert ab.recall_at_5 == 0.0  # no gold -> retrieval metric is 0/NA

    def test_qa_disabled_leaves_summary_qa_none(self):
        emb = EmbeddingService(provider="hash")
        summary = _summarize(
            mode="hybrid",
            embedder=emb,
            reranker=None,
            rerank_depth=50,
            results=_scenario(),
            elapsed_seconds=1.0,
        )
        assert summary.qa is None
        # Even with QA off, retrieval still skips the empty-gold abstention row.
        assert summary.questions == 3

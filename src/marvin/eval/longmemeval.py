"""LongMemEval-S retrieval benchmark for Marvin.

Implements the same protocol used by `agentmemory`'s
``benchmark/longmemeval-bench.ts`` so numbers are comparable:

- Build a fresh in-memory index per question from the question's
  ``haystack_sessions`` (one note per session).
- Query the index with the question text.
- Compute ``recall_any@K`` (does ANY gold session appear in top-K?),
  ``NDCG@10``, and ``MRR`` over the retrieved session ids.
- Filter out abstention question types because they have no gold sessions
  to recall.

Reference:
    Wu et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term
    Interactive Memory.* ICLR 2025. https://arxiv.org/abs/2410.10813
    Cleaned dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from marvin.embeddings import EmbeddingService
from marvin.index import MemoryIndex, chunk_markdown
from marvin.models import MemoryKind, NoteMetadata, NoteRecord, utc_now
from marvin.reranker import RerankerService

from .judge import Judge, is_abstention
from .reader import ReaderContextItem, ReaderEngine, build_reader_context

Mode = Literal["bm25", "vector", "hybrid"]

ABSTENTION_TYPES: frozenset[str] = frozenset(
    {
        "single-session-user_abs",
        "multi-session_abs",
        "knowledge-update_abs",
        "temporal-reasoning_abs",
    }
)


class LongMemEvalEntry(BaseModel):
    """One question from LongMemEval-S.

    The dataset is heterogeneously typed in the wild (``answer`` is sometimes
    a string, an int, or a list). We accept ``Any`` for fields we never act
    on so loading never fails on a stray entry.
    """

    question_id: str
    question_type: str
    question: str
    question_date: str = ""
    answer: object = ""
    answer_session_ids: list[str] = Field(default_factory=list)
    haystack_dates: list[str] = Field(default_factory=list)
    haystack_session_ids: list[str]
    haystack_sessions: list[list[dict[str, object]]]


class QuestionResult(BaseModel):
    """Per-question retrieval result."""

    question_id: str
    question_type: str
    retrieved_session_ids: list[str]
    gold_session_ids: list[str]
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    mrr: float
    latency_ms: float
    hypothesis: str | None = None
    qa_correct: bool | None = None


class TypeBreakdown(BaseModel):
    count: int
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    mrr: float
    qa_accuracy: float | None = None
    qa_count: int = 0


class QASummary(BaseModel):
    """Aggregate QA (reader + judge) accuracy across a run.

    ``questions`` counts only questions that received a judge verdict;
    ``errors`` counts QA-enabled questions where the reader or judge
    produced no verdict. Abstention questions are tracked separately
    because they have no gold evidence and are scored by the dedicated
    abstention judge prompt.
    """

    reader_model: str
    judge_model: str
    reader_top_k: int
    questions: int = 0
    correct: int = 0
    accuracy: float = 0.0
    errors: int = 0
    abstention_questions: int = 0
    abstention_correct: int = 0


class BenchSummary(BaseModel):
    """Aggregate results of a benchmark run."""

    mode: Mode
    embedding_provider: str
    embedding_model: str
    reranker_provider: str | None = None
    reranker_model: str | None = None
    rerank_depth: int | None = None
    questions: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    mrr: float
    median_latency_ms: float
    total_seconds: float
    per_type: dict[str, TypeBreakdown]
    per_question: list[QuestionResult] = Field(default_factory=list)
    qa: QASummary | None = None


# ---------------------------------------------------------------------------
# Metric primitives. Keep these pure so they are trivial to test.
# ---------------------------------------------------------------------------


def recall_any_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Return 1.0 if any gold id appears in the first ``k`` retrieved ids."""
    top_k = set(retrieved[:k])
    return 1.0 if any(g in top_k for g in gold) else 0.0


def _dcg(relevances: Sequence[bool], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel:
            total += 1.0 / math.log2(i + 2)
    return total


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Normalised DCG at ``k`` over binary session-level relevance."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    rels = [r in gold_set for r in retrieved[:k]]
    ideal_count = min(k, len(gold_set))
    ideal = _dcg([True] * ideal_count, k)
    if ideal == 0.0:
        return 0.0
    return _dcg(rels, k) / ideal


def mean_reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    """Reciprocal rank of the first gold session in ``retrieved``."""
    gold_set = set(gold)
    for i, sid in enumerate(retrieved):
        if sid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Dataset loading.
# ---------------------------------------------------------------------------


def load_dataset(path: Path, *, include_abstention: bool = False) -> list[LongMemEvalEntry]:
    """Load LongMemEval-S JSON file into ``LongMemEvalEntry`` objects."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list, got {type(raw).__name__}")

    entries = [LongMemEvalEntry.model_validate(item) for item in raw]
    if include_abstention:
        return entries
    return [e for e in entries if e.question_type not in ABSTENTION_TYPES]


# ---------------------------------------------------------------------------
# Synthetic note construction. One LongMemEval session = one Marvin note.
# ---------------------------------------------------------------------------


def _format_turns(turns: Sequence[dict[str, object]]) -> str:
    parts: list[str] = []
    for turn in turns:
        role = str(turn.get("role", "user"))
        content = str(turn.get("content", ""))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _session_to_note(
    session_id: str,
    turns: Sequence[dict[str, object]],
    *,
    vault_root: Path,
    timestamp: datetime | None = None,
) -> NoteRecord:
    body = _format_turns(turns)
    title = f"Session {session_id}"
    raw_text = f"# {title}\n\n{body}"
    ts = timestamp or utc_now()
    metadata = NoteMetadata(
        kind=MemoryKind.EPISODIC,
        title=title,
        created_at=ts,
        updated_at=ts,
    )
    return NoteRecord(
        path=vault_root / f"{session_id}.md",
        metadata=metadata,
        body=body,
        raw_text=raw_text,
    )


# LongMemEval timestamps look like ``2023/05/30 (Tue) 23:40``. The
# weekday in parens is redundant given the date so we strip it before
# parsing. Returns ``None`` when the format does not match -- the
# benchmark harness then falls back to ``utc_now()`` so a malformed
# entry still runs (just without temporal signal).
_LME_TS_PATTERN = re.compile(r"^(?P<date>\d{4}/\d{2}/\d{2})\s*\([^)]+\)\s*(?P<time>\d{2}:\d{2})$")


def parse_longmemeval_timestamp(value: str | None) -> datetime | None:
    """Parse a LongMemEval ``YYYY/MM/DD (DOW) HH:MM`` string into UTC.

    LongMemEval does not record a timezone; we anchor to UTC. The
    weekday in parens is redundant given the date so we strip it. The
    function returns ``None`` for missing/malformed strings so callers
    can fall back to a default rather than raising.
    """
    if not value:
        return None
    match = _LME_TS_PATTERN.match(value.strip())
    if match is None:
        return None
    raw = f"{match['date']} {match['time']}"
    try:
        naive = datetime.strptime(raw, "%Y/%m/%d %H:%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Single-mode retrieval helpers. They mirror MemoryIndex.hybrid_search but
# expose individual streams so we can measure ablations.
# ---------------------------------------------------------------------------


def _rank_by_path(rows: Sequence[sqlite3.Row], score_fn) -> list[tuple[str, float]]:
    """Aggregate per-chunk rows into per-note ranking using ``max`` pooling.

    ``score_fn`` maps ``(rank, row)`` to a score; we keep the best score
    per relative_path then sort descending.
    """
    best: dict[str, float] = {}
    for rank, row in enumerate(rows, start=1):
        score = score_fn(rank, row)
        path = row["relative_path"]
        prev = best.get(path)
        if prev is None or score > prev:
            best[path] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _bm25_ranking(index: MemoryIndex, query: str, depth: int) -> list[tuple[str, float]]:
    rows = index._fts_hits(query=query, limit=depth, kind=None)
    if not rows:
        return []
    # FTS rank is "lower is better" so invert for max pooling.
    return _rank_by_path(rows, lambda rank, row: -float(row["rank"]))


def _vector_ranking(index: MemoryIndex, query_embedding, depth: int) -> list[tuple[str, float]]:
    rows = index._vector_hits(query_embedding=query_embedding, limit=depth, kind=None)
    if not rows:
        return []
    return _rank_by_path(rows, lambda rank, row: -float(row["distance"]))


def _hybrid_chunk_rows(
    index: MemoryIndex,
    query: str,
    query_embedding,
    depth: int,
) -> list[tuple[sqlite3.Row, float]]:
    """Chunk-level hybrid retrieval: RRF fuse FTS + vector at the chunk level.

    Unlike :meth:`MemoryIndex.hybrid_search`, this does *not* max-pool to
    notes. It returns the top ``depth`` chunk rows with their RRF scores
    so a reranker can operate on the exact snippet that matched.
    """
    vec_rows = index._vector_hits(query_embedding=query_embedding, limit=depth, kind=None)
    fts_rows = index._fts_hits(query=query, limit=depth, kind=None)
    scores: dict[int, float] = defaultdict(float)
    details: dict[int, sqlite3.Row] = {}
    rrf_k = 60.0
    for rank, row in enumerate(vec_rows, start=1):
        cid = row["chunk_id"]
        scores[cid] += 1.0 / (rrf_k + rank)
        details[cid] = row
    for rank, row in enumerate(fts_rows, start=1):
        cid = row["chunk_id"]
        scores[cid] += 1.0 / (rrf_k + rank)
        details[cid] = row
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:depth]
    return [(details[cid], score) for cid, score in ranked]


def _chunk_rows_for_mode(
    index: MemoryIndex,
    mode: Mode,
    query: str,
    query_embedding,
    depth: int,
) -> list[sqlite3.Row]:
    """Top ``depth`` chunk rows for a given mode (pre-aggregation).

    Callers that rerank operate on these rows directly and max-pool to
    sessions afterwards. Passing chunk text (not session bodies) to the
    reranker is essential: the cross-encoder only sees 512 tokens, so
    giving it a naive prefix of a long session usually misses the signal.
    """
    if mode == "hybrid":
        return [row for row, _ in _hybrid_chunk_rows(index, query, query_embedding, depth)]
    if mode == "bm25":
        return index._fts_hits(query=query, limit=depth, kind=None)
    return index._vector_hits(query_embedding=query_embedding, limit=depth, kind=None)


# ---------------------------------------------------------------------------
# Bench runner.
# ---------------------------------------------------------------------------


@contextmanager
def _ephemeral_index(dimensions: int, **index_kwargs: object):
    """In-memory MemoryIndex that gets torn down on exit."""
    index = MemoryIndex(Path(":memory:"), dimensions=dimensions, **index_kwargs)
    try:
        yield index
    finally:
        index.close()


def _run_question(
    entry: LongMemEvalEntry,
    *,
    mode: Mode,
    embedder: EmbeddingService,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    vault_root: Path,
    max_embed_chars: int,
    reranker: RerankerService | None = None,
    rerank_depth: int = 50,
    index_options: dict[str, object] | None = None,
    reader: ReaderEngine | None = None,
    judge: Judge | None = None,
    reader_top_k: int = 10,
) -> QuestionResult:
    import numpy as np

    needs_vectors = mode in {"vector", "hybrid"}
    started = time.perf_counter()

    # Pair each session with its dataset timestamp (zip is short-tolerant
    # rather than ``strict`` because not every release of the dataset
    # carries dates for every haystack entry; missing entries fall back
    # to ``utc_now()`` per ``_session_to_note``).
    haystack_dates = list(entry.haystack_dates)
    while len(haystack_dates) < len(entry.haystack_session_ids):
        haystack_dates.append("")

    query_time = parse_longmemeval_timestamp(entry.question_date)

    with _ephemeral_index(embedder.dimensions, **(index_options or {})) as index:
        # Build chunks for every session up front, then embed in one big
        # batch. fastembed has substantial per-call overhead and
        # superlinear scaling on long inputs, so this is materially faster
        # than embedding session-by-session.
        per_session_chunks: list[tuple[str, NoteRecord, list]] = []
        for session_id, turns, raw_ts in zip(
            entry.haystack_session_ids,
            entry.haystack_sessions,
            haystack_dates,
            strict=True,
        ):
            note = _session_to_note(
                session_id,
                turns,
                vault_root=vault_root,
                timestamp=parse_longmemeval_timestamp(raw_ts),
            )
            chunks = chunk_markdown(note, chunk_size, chunk_overlap)
            per_session_chunks.append((session_id, note, chunks))

        if needs_vectors:
            flat_texts = [
                c.text[:max_embed_chars] for _, _, chunks in per_session_chunks for c in chunks
            ]
            flat_embeds = embedder.embed_texts(flat_texts)
        else:
            flat_embeds = None

        cursor = 0
        for session_id, note, chunks in per_session_chunks:
            n = len(chunks)
            if needs_vectors:
                assert flat_embeds is not None
                embeddings = flat_embeds[cursor : cursor + n]
            else:
                embeddings = [np.zeros(embedder.dimensions, dtype="float32") for _ in chunks]
            cursor += n
            index.upsert_note(
                note,
                relative_path=session_id,
                chunks=chunks,
                embeddings=embeddings,
            )

        query_embedding = embedder.embed_text(entry.question) if needs_vectors else None

        if reranker is not None:
            # Rerank at chunk granularity (the reranker input window is
            # only 512 tokens, so session-body prefixes miss the signal in
            # long conversations). We fetch more chunks than rerank_depth
            # sessions because several chunks may belong to the same
            # session; max-pooling after scoring collapses them back.
            #
            # Note: ``query_time`` is not threaded into the chunk-level
            # rerank path because reranking only re-orders the
            # first-stage chunk pool by query/document interaction;
            # decay is a note-level signal that would only kick in
            # again post-rerank, where it could fight the cross-encoder's
            # ranking. The benchmark therefore exercises decay only on
            # the no-rerank ``hybrid_search`` branch.
            chunk_rows = _chunk_rows_for_mode(
                index=index,
                mode=mode,
                query=entry.question,
                query_embedding=query_embedding,
                depth=max(rerank_depth, top_k * 3),
            )
            retrieved = _rerank_chunks_to_sessions(
                query=entry.question,
                chunk_rows=chunk_rows,
                reranker=reranker,
                top_k=top_k,
            )
        else:
            if mode == "hybrid":
                hits = index.hybrid_search(
                    query=entry.question,
                    query_embedding=query_embedding,
                    limit=top_k,
                    query_time=query_time,
                )
                retrieved = [hit.path for hit in hits]
            elif mode == "bm25":
                ranking = _bm25_ranking(index, entry.question, depth=max(top_k * 5, 20))
                retrieved = [path for path, _ in ranking[:top_k]]
            else:  # vector
                ranking = _vector_ranking(index, query_embedding, depth=max(top_k * 5, 20))
                retrieved = [path for path, _ in ranking[:top_k]]

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    gold = entry.answer_session_ids

    # QA arm: feed the top retrieved sessions to the reader, then grade the
    # answer with the canonical per-type judge. Done after ``elapsed_ms`` so
    # the reported retrieval latency excludes reader/judge LLM time.
    hypothesis: str | None = None
    qa_correct: bool | None = None
    if reader is not None and judge is not None:
        note_by_path = {sid: note for sid, note, _ in per_session_chunks}
        date_by_session = dict(zip(entry.haystack_session_ids, haystack_dates, strict=False))
        items: list[ReaderContextItem] = []
        for rel_path in retrieved[:reader_top_k]:
            note = note_by_path.get(rel_path)
            if note is None:
                continue
            items.append(
                ReaderContextItem(
                    kind=note.metadata.kind.value,
                    session_id=rel_path,
                    date=date_by_session.get(rel_path, ""),
                    body=note.body,
                )
            )
        context = build_reader_context(items)
        reader_result = reader.answer(entry.question, context, question_date=entry.question_date)
        hypothesis = reader_result.answer
        qa_correct = judge.judge(
            question=entry.question,
            answer=str(entry.answer),
            hypothesis=hypothesis,
            question_type=entry.question_type,
            question_id=entry.question_id,
        )

    return QuestionResult(
        question_id=entry.question_id,
        question_type=entry.question_type,
        retrieved_session_ids=retrieved,
        gold_session_ids=gold,
        recall_at_5=recall_any_at_k(retrieved, gold, 5),
        recall_at_10=recall_any_at_k(retrieved, gold, 10),
        recall_at_20=recall_any_at_k(retrieved, gold, 20),
        ndcg_at_10=ndcg_at_k(retrieved, gold, 10),
        mrr=mean_reciprocal_rank(retrieved, gold),
        latency_ms=elapsed_ms,
        hypothesis=hypothesis,
        qa_correct=qa_correct,
    )


def _rerank_chunks_to_sessions(
    *,
    query: str,
    chunk_rows: Sequence[sqlite3.Row],
    reranker: RerankerService,
    top_k: int,
) -> list[str]:
    """Cross-encode ``(query, chunk.text)`` pairs then max-pool by session.

    The reranker assigns a relevance score to each chunk; every chunk's
    score is compared against its owning session's running best. Sessions
    are then ranked by best-chunk score and the top ``top_k`` paths are
    returned.
    """
    if not chunk_rows:
        return []
    docs = [row["text"] for row in chunk_rows]
    scores = reranker.score(query, docs)
    best: dict[str, float] = {}
    for row, score in zip(chunk_rows, scores, strict=True):
        path = row["relative_path"]
        prev = best.get(path)
        if prev is None or score > prev:
            best[path] = score
    ordered = sorted(best.items(), key=lambda x: -x[1])
    return [path for path, _ in ordered[:top_k]]


def run_benchmark(
    entries: Sequence[LongMemEvalEntry],
    *,
    mode: Mode = "hybrid",
    embedder: EmbeddingService | None = None,
    reranker: RerankerService | None = None,
    rerank_depth: int = 50,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    top_k: int = 20,
    max_embed_chars: int = 512,
    progress: int = 0,
    index_options: dict[str, object] | None = None,
    reader: ReaderEngine | None = None,
    judge: Judge | None = None,
    reader_top_k: int = 10,
) -> BenchSummary:
    """Run the benchmark on ``entries`` and return an aggregated summary.

    Args:
        entries: Pre-filtered list of LongMemEval entries.
        mode: One of ``"bm25"``, ``"vector"``, ``"hybrid"``.
        embedder: Embedding service. Required for ``vector`` and ``hybrid``;
            if omitted, a default :class:`EmbeddingService` is constructed.
        reranker: Optional cross-encoder reranker. When provided, the first
            stage retrieves ``rerank_depth`` sessions and the reranker
            reorders them; the final top-K are taken from the reranked list.
        rerank_depth: First-stage retrieval depth when ``reranker`` is set.
        chunk_size, chunk_overlap: Forwarded to :func:`chunk_markdown`.
        top_k: How many retrieved sessions to keep per question.
        max_embed_chars: Truncate each chunk to this many characters before
            embedding. Matches the agentmemory benchmark protocol (512) and
            keeps fastembed's superlinear long-input cost in check; FTS5
            still indexes the full chunk text.
        progress: If > 0, print a one-line progress update every N questions.
    """
    if embedder is None:
        embedder = EmbeddingService()

    started = time.perf_counter()
    vault_root = Path("/tmp/marvin-eval-vault")  # never written to disk
    results: list[QuestionResult] = []

    for i, entry in enumerate(entries, start=1):
        result = _run_question(
            entry,
            mode=mode,
            embedder=embedder,
            reranker=reranker,
            rerank_depth=rerank_depth,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            vault_root=vault_root,
            max_embed_chars=max_embed_chars,
            index_options=index_options,
            reader=reader,
            judge=judge,
            reader_top_k=reader_top_k,
        )
        results.append(result)

        if progress and i % progress == 0:
            ret_so_far = [r for r in results if r.gold_session_ids]
            running = (
                sum(r.recall_at_5 for r in ret_so_far) / len(ret_so_far) if ret_so_far else 0.0
            )
            line = f"  [{i:>4}/{len(entries)}]  running R@5: {running * 100:5.1f}%"
            judged = [r for r in results if r.qa_correct is not None]
            if judged:
                acc = sum(1 for r in judged if r.qa_correct) / len(judged)
                line += f"  QA: {acc * 100:5.1f}%"
            print(line, flush=True)

    elapsed = time.perf_counter() - started
    return _summarize(
        mode=mode,
        embedder=embedder,
        reranker=reranker,
        rerank_depth=rerank_depth,
        results=results,
        elapsed_seconds=elapsed,
        qa_enabled=reader is not None and judge is not None,
        reader_model=reader.model if reader is not None else None,
        judge_model=judge.model if judge is not None else None,
        reader_top_k=reader_top_k,
    )


def _summarize(
    *,
    mode: Mode,
    embedder: EmbeddingService,
    reranker: RerankerService | None,
    rerank_depth: int,
    results: list[QuestionResult],
    elapsed_seconds: float,
    qa_enabled: bool = False,
    reader_model: str | None = None,
    judge_model: str | None = None,
    reader_top_k: int = 10,
) -> BenchSummary:
    reranker_provider = reranker.backend_name if reranker is not None else None
    reranker_model = reranker.model_name if reranker is not None else None
    rerank_depth_value = rerank_depth if reranker is not None else None

    qa_summary = _aggregate_qa(
        results=results,
        enabled=qa_enabled,
        reader_model=reader_model,
        judge_model=judge_model,
        reader_top_k=reader_top_k,
    )

    n = len(results)
    if n == 0:
        empty = BenchSummary(
            mode=mode,
            embedding_provider=embedder.backend_name,
            embedding_model=embedder.model_name,
            reranker_provider=reranker_provider,
            reranker_model=reranker_model,
            rerank_depth=rerank_depth_value,
            questions=0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            recall_at_20=0.0,
            ndcg_at_10=0.0,
            mrr=0.0,
            median_latency_ms=0.0,
            total_seconds=elapsed_seconds,
            per_type={},
            qa=qa_summary,
        )
        return empty

    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)

    per_type = {}
    for qtype, rs in by_type.items():
        # Retrieval metrics are defined only where gold evidence exists;
        # abstention questions (empty gold) are scored on QA accuracy only.
        ret_rs = [r for r in rs if r.gold_session_ids]
        qa_rs = [r for r in rs if r.qa_correct is not None]
        per_type[qtype] = TypeBreakdown(
            count=len(rs),
            recall_at_5=_mean([r.recall_at_5 for r in ret_rs]),
            recall_at_10=_mean([r.recall_at_10 for r in ret_rs]),
            ndcg_at_10=_mean([r.ndcg_at_10 for r in ret_rs]),
            mrr=_mean([r.mrr for r in ret_rs]),
            qa_accuracy=(_mean([1.0 if r.qa_correct else 0.0 for r in qa_rs]) if qa_rs else None),
            qa_count=len(qa_rs),
        )

    # Retrieval aggregates skip abstention so the headline recall numbers
    # stay comparable to the agentmemory protocol (and to runs without QA).
    ret_results = [r for r in results if r.gold_session_ids]

    sorted_lat = sorted(r.latency_ms for r in results)
    median_lat = sorted_lat[n // 2]

    return BenchSummary(
        mode=mode,
        embedding_provider=embedder.backend_name,
        embedding_model=embedder.model_name,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
        rerank_depth=rerank_depth_value,
        questions=len(ret_results),
        recall_at_5=_mean([r.recall_at_5 for r in ret_results]),
        recall_at_10=_mean([r.recall_at_10 for r in ret_results]),
        recall_at_20=_mean([r.recall_at_20 for r in ret_results]),
        ndcg_at_10=_mean([r.ndcg_at_10 for r in ret_results]),
        mrr=_mean([r.mrr for r in ret_results]),
        median_latency_ms=median_lat,
        total_seconds=elapsed_seconds,
        per_type=per_type,
        per_question=results,
        qa=qa_summary,
    )


def _mean(xs: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate_qa(
    *,
    results: Sequence[QuestionResult],
    enabled: bool,
    reader_model: str | None,
    judge_model: str | None,
    reader_top_k: int,
) -> QASummary | None:
    """Roll up reader/judge verdicts into a run-level QA summary."""
    if not enabled:
        return None
    judged = [r for r in results if r.qa_correct is not None]
    correct = sum(1 for r in judged if r.qa_correct)
    errors = sum(1 for r in results if r.qa_correct is None)
    abs_rs = [r for r in results if is_abstention(r.question_id, r.question_type)]
    abs_judged = [r for r in abs_rs if r.qa_correct is not None]
    return QASummary(
        reader_model=reader_model or "-",
        judge_model=judge_model or "-",
        reader_top_k=reader_top_k,
        questions=len(judged),
        correct=correct,
        accuracy=(correct / len(judged) if judged else 0.0),
        errors=errors,
        abstention_questions=len(abs_judged),
        abstention_correct=sum(1 for r in abs_judged if r.qa_correct),
    )


def format_summary(summary: BenchSummary) -> str:
    """Pretty-print a summary as a multi-line string for console output."""
    lines: list[str] = []
    header = summary.mode
    if summary.reranker_provider:
        header = f"{header} + rerank"
    lines.append(f"=== LongMemEval-S Results ({header}) ===")
    lines.append(f"Embedder: {summary.embedding_provider} ({summary.embedding_model})")
    if summary.reranker_provider:
        lines.append(
            f"Reranker: {summary.reranker_provider} ({summary.reranker_model})"
            f" depth={summary.rerank_depth}"
        )
    lines.append(f"Questions: {summary.questions}")
    lines.append(f"recall_any@5:  {summary.recall_at_5 * 100:5.1f}%")
    lines.append(f"recall_any@10: {summary.recall_at_10 * 100:5.1f}%")
    lines.append(f"recall_any@20: {summary.recall_at_20 * 100:5.1f}%")
    lines.append(f"NDCG@10:       {summary.ndcg_at_10 * 100:5.1f}%")
    lines.append(f"MRR:           {summary.mrr * 100:5.1f}%")
    if summary.qa is not None:
        lines.append(
            f"QA accuracy:   {summary.qa.accuracy * 100:5.1f}%  (n={summary.qa.questions})"
        )
    lines.append(
        f"Median latency: {summary.median_latency_ms:.1f}ms  (total {summary.total_seconds:.1f}s)"
    )
    if summary.per_type:
        lines.append("")
        lines.append("By question type:")
        for qtype in sorted(summary.per_type):
            br = summary.per_type[qtype]
            line = (
                f"  {qtype:<32}  R@5 {br.recall_at_5 * 100:5.1f}%  "
                f"R@10 {br.recall_at_10 * 100:5.1f}%  "
                f"NDCG@10 {br.ndcg_at_10 * 100:5.1f}%  "
                f"MRR {br.mrr * 100:5.1f}%  (n={br.count})"
            )
            if br.qa_accuracy is not None:
                line += f"  QA {br.qa_accuracy * 100:5.1f}%"
            lines.append(line)
    if summary.qa is not None:
        q = summary.qa
        lines.append("")
        lines.append(f"QA: reader={q.reader_model} judge={q.judge_model} top_k={q.reader_top_k}")
        lines.append(
            f"  accuracy={q.accuracy * 100:.1f}%  correct={q.correct}/{q.questions}  "
            f"errors={q.errors}"
        )
        if q.abstention_questions:
            lines.append(f"  abstention: {q.abstention_correct}/{q.abstention_questions} correct")
    return "\n".join(lines)

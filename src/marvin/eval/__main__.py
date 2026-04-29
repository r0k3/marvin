"""CLI entrypoint for ``python -m marvin.eval`` (LongMemEval).

Examples:
    # Run hybrid retrieval on the full dataset.
    python -m marvin.eval --dataset path/to/longmemeval_s_cleaned.json

    # Quick BM25 sanity check on the first 10 questions.
    python -m marvin.eval --dataset PATH --mode bm25 --limit 10

    # Force the deterministic hash backend (no network/model download).
    python -m marvin.eval --dataset PATH --embedding-provider hash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from marvin.embeddings import EmbeddingService
from marvin.reranker import DEFAULT_RERANK_MODEL, RerankerService

from ._results import resolve_results_path
from .longmemeval import (
    Mode,
    BenchSummary,
    format_summary,
    load_dataset,
    run_benchmark,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marvin.eval",
        description="LongMemEval-S retrieval benchmark for Marvin.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--mode",
        choices=("bm25", "vector", "hybrid"),
        default="hybrid",
        help="Retrieval mode to evaluate (default: hybrid)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N (post-filter) questions; 0 = all",
    )
    parser.add_argument(
        "--question-type",
        action="append",
        default=None,
        help="Restrict to entries whose ``question_type`` matches. May be "
        "passed multiple times to keep several types. Useful for "
        "ablations on a specific slice (knowledge-update, "
        "temporal-reasoning, ...).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Retrieve K sessions per question (default: 20)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="chunk_markdown chunk size (default: 1200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="chunk_markdown chunk overlap (default: 200)",
    )
    parser.add_argument(
        "--max-embed-chars",
        type=int,
        default=512,
        help="Truncate chunk text to N chars before embedding (default: 512, "
        "matching the agentmemory protocol). FTS5 still indexes the full text.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("auto", "fastembed", "hash"),
        default="auto",
        help="Embedding backend (default: auto = fastembed if available)",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Model name for the embedding backend",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Embedding dimensionality (must match the model)",
    )
    parser.add_argument(
        "--include-abstention",
        action="store_true",
        help="Keep abstention question types (default: excluded, matching the "
        "agentmemory protocol)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank first-stage results with a cross-encoder before top-K",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help=(
            "Reranker model name (default: %(default)s). Any model listed by "
            "``TextCrossEncoder.list_supported_models()`` works, plus "
            f"{DEFAULT_RERANK_MODEL!r} via our custom ONNX registration."
        ),
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=50,
        help="First-stage retrieval depth when reranking (default: 50)",
    )
    parser.add_argument(
        "--rerank-max-chars",
        type=int,
        default=1024,
        help="Truncate each document to N chars before reranking (default: 1024)",
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Disable the K-Lines graph stream entirely (chunk-only RRF)",
    )
    parser.add_argument(
        "--at-ingest",
        action="store_true",
        help="Enable at-ingest regex entity extraction (off by default; "
        "produces a small regression on LongMemEval-S because chat data "
        "is dominated by sentence-starter capitalised noise)",
    )
    parser.add_argument(
        "--kg-ingest-min-length",
        type=int,
        default=3,
        help="Drop at-ingest entities shorter than this many characters "
        "(default: 3, matching MarvinSettings)",
    )
    parser.add_argument(
        "--kg-allow-single-word",
        action="store_true",
        help="Disable the default multi-word-only filter on at-ingest "
        "entities. Single capitalised tokens (Spotify, Java) become "
        "entities too -- useful on curated text, but on chat-style "
        "data introduces heavy sentence-starter noise.",
    )
    parser.add_argument(
        "--kg-fusion-weight",
        type=float,
        default=0.5,
        help="RRF weight for the graph stream relative to the chunk "
        "stream (default 0.5; 1.0 = symmetric fusion)",
    )
    parser.add_argument(
        "--decay",
        action="store_true",
        help="Apply a freshness boost to the final note ranking (uses "
        "haystack/question dates from the dataset). Off by default.",
    )
    parser.add_argument(
        "--decay-half-life-days",
        type=float,
        default=30.0,
        help="Half-life of the freshness boost in days (default: 30). "
        "Smaller values favour very recent notes more aggressively.",
    )
    parser.add_argument(
        "--decay-weight",
        type=float,
        default=0.5,
        help="Maximum freshness multiplier (default: 0.5; an instant-old "
        "note's score is multiplied by 1.5, an infinitely-old note by 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the summary JSON to this path (mutually exclusive with --results-dir)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Write to <results-dir>/<git-short-sha>/<auto-name>.json. The "
        "auto-name encodes mode + key flags (embedder, rerank, limit) so "
        "regression diffs across commits don't clobber each other.",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=25,
        help="Print a one-line progress update every N questions (0 = silent)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"error: dataset not found at {args.dataset}", file=sys.stderr)
        print(
            "  download with: python scripts/download_longmemeval.py",
            file=sys.stderr,
        )
        return 2

    entries = load_dataset(args.dataset, include_abstention=args.include_abstention)
    if args.question_type:
        wanted = set(args.question_type)
        entries = [e for e in entries if e.question_type in wanted]
    if args.limit > 0:
        entries = entries[: args.limit]
    if not entries:
        print("error: no entries to evaluate", file=sys.stderr)
        return 2

    embedder = EmbeddingService(
        provider=args.embedding_provider,
        model_name=args.embedding_model,
        dimensions=args.embedding_dim,
    )

    reranker: RerankerService | None = None
    if args.rerank:
        reranker = RerankerService(
            provider="fastembed",
            model_name=args.rerank_model,
            max_chars=args.rerank_max_chars,
        )
        # Eagerly load so a missing model surfaces before we churn on data.
        _ = reranker.backend_name

    mode_banner = args.mode + (" + rerank" if reranker else "")
    if args.decay:
        mode_banner += " + decay"
    print(
        f"Running LongMemEval-S in {mode_banner} mode on {len(entries)} questions...",
        flush=True,
    )

    index_options: dict[str, object] = {
        "kg_enabled": not args.no_kg,
        "kg_extract_at_ingest": args.at_ingest,
        "kg_ingest_min_length": args.kg_ingest_min_length,
        "kg_ingest_multiword_only": not args.kg_allow_single_word,
        "kg_fusion_weight": args.kg_fusion_weight,
        "decay_enabled": args.decay,
        "decay_half_life_days": args.decay_half_life_days,
        "decay_weight": args.decay_weight,
    }

    summary = run_benchmark(
        entries,
        mode=args.mode,
        embedder=embedder,
        reranker=reranker,
        rerank_depth=args.rerank_depth,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        max_embed_chars=args.max_embed_chars,
        progress=args.progress,
        index_options=index_options,
    )

    print()
    print(format_summary(summary))

    if args.output is not None and args.results_dir is not None:
        print(
            "warning: --output is ignored when --results-dir is set",
            file=sys.stderr,
        )

    output_path: Path | None = None
    if args.results_dir is not None:
        output_path = resolve_results_path(args.results_dir, args)
    elif args.output is not None:
        output_path = args.output

    if output_path is not None:
        _write_summary(summary, output_path)
        print(f"\nSaved JSON to {output_path}")

    return 0


def _write_summary(summary: BenchSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())

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
        "--output",
        type=Path,
        default=None,
        help="Write the summary JSON to this path",
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
    print(
        f"Running LongMemEval-S in {mode_banner} mode on {len(entries)} questions...",
        flush=True,
    )

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
    )

    print()
    print(format_summary(summary))

    if args.output is not None:
        _write_summary(summary, args.output)
        print(f"\nSaved JSON to {args.output}")

    return 0


def _write_summary(summary: BenchSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())

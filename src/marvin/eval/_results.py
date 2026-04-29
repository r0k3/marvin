"""Helpers for writing benchmark results to a versioned location.

The eval CLI accepts ``--results-dir DIR``, which writes the summary
JSON to ``DIR/<short-sha>/<auto-name>.json`` so we can diff a benchmark
across commits without overwriting prior runs. The auto-name encodes
the configuration knobs that change retrieval behaviour (mode, embedder,
reranker, limit, K-Lines flags) so two distinct runs at the same SHA
produce distinct filenames.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

__all__ = [
    "git_short_sha",
    "build_auto_name",
    "resolve_results_path",
]


def git_short_sha(repo_root: Path | None = None) -> str:
    """Return the short SHA of ``HEAD``, with ``-dirty`` if the working tree differs.

    Returns ``"unknown"`` if ``repo_root`` (or its parent) isn't a git
    repository or git is unavailable. Never raises.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    if not sha:
        return "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _slug(value: str) -> str:
    """Turn a model id like ``BAAI/bge-small-en-v1.5`` into ``bge-small-en-v1-5``.

    We strip the org prefix (``BAAI/``) because it's noise; lowercase
    the rest; collapse non-alphanumerics into single dashes. Empty
    input is returned as-is.
    """
    if not value:
        return value
    tail = value.rsplit("/", 1)[-1].lower()
    slug = _SLUG_RE.sub("-", tail).strip("-")
    return slug


def build_auto_name(args: argparse.Namespace) -> str:
    """Build a deterministic filename stem from the eval CLI args.

    The stem encodes:

    * mode (bm25 / vector / hybrid)
    * a slugged embedder model id when it differs from the default
    * ``-rerank`` if reranking is enabled, plus a slugged reranker model
      id when it differs from the default
    * ``-limitN`` when the run is not over the full dataset
    * ``-no-kg`` when the K-Lines stream is disabled
    * ``-at-ingest`` when at-ingest extraction is enabled

    Default values are dropped so a vanilla run yields a short stem
    like ``hybrid`` or ``hybrid-rerank``.
    """
    parts: list[str] = [str(args.mode)]

    embedder = getattr(args, "embedding_model", None)
    if embedder and embedder != "BAAI/bge-small-en-v1.5":
        parts.append(_slug(embedder))

    if getattr(args, "rerank", False):
        parts.append("rerank")
        rerank_model = getattr(args, "rerank_model", None)
        if rerank_model and rerank_model != "BAAI/bge-reranker-v2-m3":
            parts.append(_slug(rerank_model))

    limit = getattr(args, "limit", 0) or 0
    if limit > 0:
        parts.append(f"limit{limit}")

    if getattr(args, "no_kg", False):
        parts.append("no-kg")
    if getattr(args, "at_ingest", False):
        parts.append("at-ingest")
    if getattr(args, "decay", False):
        parts.append("decay")

    return "-".join(parts)


def resolve_results_path(
    results_dir: Path,
    args: argparse.Namespace,
    *,
    sha: str | None = None,
) -> Path:
    """Compute ``results_dir / <sha> / <auto-name>.json``.

    ``sha`` defaults to :func:`git_short_sha` against the *current working
    directory* so the SHA reflects the source code that produced the
    benchmark, not the directory the results are written to. Pass an
    explicit ``sha`` to short-circuit that resolution (useful in tests
    or when the harness runs detached from the source tree).
    """
    resolved_sha = sha if sha is not None else git_short_sha()
    name = build_auto_name(args)
    return results_dir / resolved_sha / f"{name}.json"

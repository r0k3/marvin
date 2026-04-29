"""Tests for the versioned-output helpers in :mod:`marvin.eval._results`."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from marvin.eval._results import (
    build_auto_name,
    git_short_sha,
    resolve_results_path,
)


def _make_args(**overrides: object) -> argparse.Namespace:
    """Build an args namespace with the same fields the CLI parser sets."""
    defaults: dict[str, object] = {
        "mode": "hybrid",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "rerank": False,
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        "limit": 0,
        "no_kg": False,
        "at_ingest": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildAutoName:
    def test_vanilla_hybrid_run_is_short(self):
        assert build_auto_name(_make_args()) == "hybrid"

    def test_bm25_mode(self):
        assert build_auto_name(_make_args(mode="bm25")) == "bm25"

    def test_rerank_appends_suffix(self):
        assert build_auto_name(_make_args(rerank=True)) == "hybrid-rerank"

    def test_non_default_embedder_is_slugged(self):
        name = build_auto_name(
            _make_args(embedding_model="BAAI/bge-base-en-v1.5"),
        )
        assert name == "hybrid-bge-base-en-v1-5"

    def test_non_default_reranker_is_slugged(self):
        name = build_auto_name(
            _make_args(
                rerank=True,
                rerank_model="Xenova/ms-marco-MiniLM-L-6-v2",
            ),
        )
        assert name == "hybrid-rerank-ms-marco-minilm-l-6-v2"

    def test_limit_is_appended(self):
        assert build_auto_name(_make_args(limit=100)) == "hybrid-limit100"

    def test_kg_flags_round_trip(self):
        name = build_auto_name(_make_args(no_kg=True, at_ingest=True))
        assert name == "hybrid-no-kg-at-ingest"

    def test_full_combination(self):
        name = build_auto_name(
            _make_args(
                mode="hybrid",
                embedding_model="BAAI/bge-base-en-v1.5",
                rerank=True,
                rerank_model="Xenova/ms-marco-MiniLM-L-6-v2",
                limit=50,
                no_kg=True,
            ),
        )
        # Order is fixed so two equivalent runs produce the same path.
        assert name == (
            "hybrid-bge-base-en-v1-5-rerank-ms-marco-minilm-l-6-v2-limit50-no-kg"
        )


class TestGitShortSha:
    def test_returns_unknown_outside_repo(self, tmp_path):
        assert git_short_sha(tmp_path) == "unknown"

    def test_returns_short_sha_in_repo(self, tmp_path):
        subprocess.check_call(["git", "init", "-q"], cwd=tmp_path)
        subprocess.check_call(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
        )
        sha = git_short_sha(tmp_path)
        assert sha != "unknown"
        # 7-char short sha; never the dirty marker on a clean tree.
        assert "-dirty" not in sha
        assert len(sha) >= 7

    def test_marks_dirty_when_working_tree_modified(self, tmp_path):
        subprocess.check_call(["git", "init", "-q"], cwd=tmp_path)
        subprocess.check_call(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
        )
        # Add an untracked file to dirty the tree.
        (tmp_path / "scratch.txt").write_text("hello")
        sha = git_short_sha(tmp_path)
        assert sha.endswith("-dirty"), sha


class TestResolveResultsPath:
    def test_path_is_dir_over_sha_over_filename(self, tmp_path):
        args = _make_args(rerank=True)
        path = resolve_results_path(tmp_path, args, sha="deadbee")
        assert path == tmp_path / "deadbee" / "hybrid-rerank.json"

    def test_uses_supplied_sha_verbatim(self, tmp_path):
        args = _make_args()
        # An explicit ``sha`` always wins over discovery.
        path = resolve_results_path(tmp_path, args, sha="abc1234-dirty")
        assert path.parent.name == "abc1234-dirty"
        assert path.name == "hybrid.json"

    def test_falls_back_to_git_when_sha_unset(self, tmp_path, monkeypatch):
        # Force git_short_sha to return a known value via fake subprocess.
        from marvin.eval import _results as mod

        def _fake_check_output(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
            if cmd[:3] == ["git", "rev-parse", "--short"]:
                return "feedf00\n"
            if cmd[:2] == ["git", "status"]:
                return ""
            raise AssertionError(f"unexpected git invocation: {cmd}")

        monkeypatch.setattr(mod.subprocess, "check_output", _fake_check_output)

        # The default (sha=None) should look up the SHA of the CWD's repo,
        # not the directory the results land in. ``tmp_path`` is just where
        # the file is written.
        path = resolve_results_path(tmp_path, _make_args())
        assert path == tmp_path / "feedf00" / "hybrid.json"

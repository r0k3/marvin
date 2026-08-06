"""Project tag derivation from the working directory's git repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

from marvin.project import repo_tag


def _init_repo(path: Path, remote: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    if remote:
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=path, check=True)
    return path


def test_https_remote(tmp_path: Path):
    repo = _init_repo(tmp_path / "clone", "https://github.com/r0k3/marvin.git")
    assert repo_tag(repo) == "project/r0k3-marvin"


def test_ssh_remote(tmp_path: Path):
    repo = _init_repo(tmp_path / "clone", "git@github.com:Some-Org/My.Repo.git")
    assert repo_tag(repo) == "project/some-org-my-repo"


def test_no_remote_falls_back_to_directory_name(tmp_path: Path):
    repo = _init_repo(tmp_path / "Side Project")
    assert repo_tag(repo) == "project/side-project"


def test_outside_a_repo_is_none(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert repo_tag(plain) is None


def test_subdirectory_resolves_to_repo(tmp_path: Path):
    repo = _init_repo(tmp_path / "clone", "https://github.com/r0k3/marvin")
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert repo_tag(sub) == "project/r0k3-marvin"

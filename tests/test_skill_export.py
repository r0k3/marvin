"""`marvin skill export`: compile the vault into a portable skill bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from marvin import cli


@pytest.fixture()
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MARVIN_EMBEDDING_PROVIDER", "hash")
    monkeypatch.delenv("MARVIN_VAULT_PATH", raising=False)
    return tmp_path / "vault"


def run(vault: Path, *argv: str) -> int:
    return cli.main(["--vault-path", str(vault), "--state-dir", str(vault / ".state"), *argv])


class TestExportBundle:
    def test_exports_all_four_kinds(self, vault, tmp_path, capsys):
        run(vault, "remember", "DB", "--predicate", "storage", "--value", "PostgreSQL")
        run(vault, "procedure", "Release checklist", "--step", "run tests", "--step", "tag version")
        run(
            vault,
            "template",
            "register",
            "Debug strategy",
            "--plan",
            "reproduce first",
            "--plan",
            "bisect",
            "--intent",
            "debug",
        )
        run(vault, "reflect", "Small diffs win", "--insight", "Reviewable changes land faster.")
        run(vault, "episode", "Fixed the race", "--summary", "worker double-ack resolved")
        capsys.readouterr()

        out = tmp_path / "bundle"
        assert run(vault, "skill", "export", "--out", str(out)) == 0

        skill = (out / "SKILL.md").read_text()
        assert skill.startswith("---\nname: vault-memory\n")
        assert "description:" in skill and "glossary.md" in skill

        assert "**storage**: PostgreSQL" in (out / "glossary.md").read_text()
        patterns = (out / "patterns.md").read_text()
        assert "Debug strategy (response strategy)" in patterns
        assert "1. reproduce first" in patterns
        assert "Release checklist" in patterns
        assert "Reviewable changes land faster." in (out / "cheatsheet.md").read_text()
        assert "**Fixed the race**" in (out / "history.md").read_text()

    def test_deprecated_facts_are_excluded(self, vault, tmp_path, capsys):
        run(vault, "remember", "DB", "--predicate", "storage", "--value", "MySQL")
        run(vault, "remember", "DB", "--predicate", "storage", "--value", "PostgreSQL")
        capsys.readouterr()
        out = tmp_path / "bundle"
        assert run(vault, "skill", "export", "--out", str(out)) == 0
        glossary = (out / "glossary.md").read_text()
        assert "PostgreSQL" in glossary
        assert "MySQL" not in glossary

    def test_tag_filter_and_derived_name(self, vault, tmp_path, capsys):
        run(
            vault,
            "remember",
            "CI",
            "--predicate",
            "runner",
            "--value",
            "buildkite",
            "--tags",
            "project/acme-rocket",
        )
        run(vault, "remember", "Editor", "--predicate", "choice", "--value", "helix")
        capsys.readouterr()

        out = tmp_path / "scoped"
        assert run(vault, "skill", "export", "--tag", "project/acme-rocket", "--out", str(out)) == 0
        skill = (out / "SKILL.md").read_text()
        assert "name: acme-rocket-memory" in skill
        assert "scope: project/acme-rocket" in skill
        glossary = (out / "glossary.md").read_text()
        assert "buildkite" in glossary
        assert "helix" not in glossary

    def test_budget_truncation_marker(self, vault, tmp_path, capsys):
        for i in range(40):
            run(vault, "remember", f"Concept{i}", "--predicate", "p", "--value", "x" * 80)
        capsys.readouterr()
        out = tmp_path / "small"
        assert run(vault, "skill", "export", "--out", str(out), "--max-chars", "600") == 0
        glossary = (out / "glossary.md").read_text()
        assert "truncated at the character budget" in glossary
        assert len(glossary) < 1200


class TestSkillInstallHosts:
    def test_amp_layout(self, vault, capsys, monkeypatch, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.chdir(project)
        assert run(vault, "skill", "install", "--host", "amp") == 0
        assert (project / ".agents" / "skills" / "marvin-memory" / "SKILL.md").exists()

    def test_grok_shares_claude_paths(self, vault, capsys, monkeypatch, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.chdir(project)
        assert run(vault, "skill", "install", "--host", "grok") == 0
        assert (project / ".claude" / "skills" / "marvin-memory" / "SKILL.md").exists()

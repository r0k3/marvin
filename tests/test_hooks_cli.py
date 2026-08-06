"""Auto-recall hooks: runtime commands, budgets, installer merge semantics."""

from __future__ import annotations

import io
import json
import subprocess
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


def _stdin(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    fake = io.StringIO(text)
    fake.isatty = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr("sys.stdin", fake)


class TestSessionStartHook:
    def test_injects_facts_and_recent(self, vault, capsys, monkeypatch):
        run(vault, "remember", "DB", "--predicate", "storage", "--value", "PostgreSQL")
        run(vault, "episode", "Fixed the race", "--summary", "worker double-ack resolved")
        capsys.readouterr()

        _stdin(monkeypatch, {"source": "startup"})
        assert run(vault, "hook", "session-start") == 0
        out = capsys.readouterr().out
        assert "marvin-memory auto-recall" in out
        assert "fact: DB storage: PostgreSQL" in out
        assert "recent: Fixed the race (episodic)" in out
        assert "use: marvin search" in out

    def test_resume_is_silent(self, vault, capsys, monkeypatch):
        run(vault, "remember", "DB", "--predicate", "storage", "--value", "PostgreSQL")
        capsys.readouterr()
        _stdin(monkeypatch, {"source": "resume"})
        assert run(vault, "hook", "session-start") == 0
        assert capsys.readouterr().out == ""

    def test_budget_is_respected(self, vault, capsys, monkeypatch):
        for i in range(30):
            run(vault, "remember", f"Concept{i}", "--predicate", "p", "--value", "v" * 60)
        capsys.readouterr()
        _stdin(monkeypatch, {"source": "startup"})
        assert run(vault, "hook", "session-start", "--budget-chars", "400") == 0
        out = capsys.readouterr().out
        assert len(out) <= 401  # budget + trailing newline
        assert "use: marvin search" in out  # the use line always survives


class TestUserPromptHook:
    def test_recalls_lexical_matches(self, vault, capsys, monkeypatch):
        run(vault, "remember", "DB", "--predicate", "storage", "--value", "PostgreSQL with asyncpg")
        capsys.readouterr()
        _stdin(monkeypatch, {"prompt": "how do we talk to postgresql from the worker?"})
        assert run(vault, "hook", "user-prompt") == 0
        out = capsys.readouterr().out
        assert "marvin-memory recall:" in out
        assert "DB (semantic)" in out

    def test_correction_nudge(self, vault, capsys, monkeypatch):
        _stdin(monkeypatch, {"prompt": "No, the timeout should be 30s not 10s."})
        assert run(vault, "hook", "user-prompt") == 0
        out = capsys.readouterr().out
        assert "possible correction" in out
        assert "--reason" in out

    def test_short_and_slash_prompts_are_silent(self, vault, capsys, monkeypatch):
        _stdin(monkeypatch, {"prompt": "ok thanks"})
        assert run(vault, "hook", "user-prompt") == 0
        _stdin(monkeypatch, {"prompt": "/compact everything now please"})
        assert run(vault, "hook", "user-prompt") == 0
        assert capsys.readouterr().out == ""

    def test_no_hits_no_output(self, vault, capsys, monkeypatch):
        _stdin(monkeypatch, {"prompt": "completely unrelated zebra question here"})
        assert run(vault, "hook", "user-prompt") == 0
        assert capsys.readouterr().out == ""


class TestAutoProjectTag:
    def test_writes_gain_repo_tag(self, vault, capsys, monkeypatch, tmp_path):
        repo = tmp_path / "workrepo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/acme/rocket.git"],
            cwd=repo,
            check=True,
        )
        monkeypatch.setenv("MARVIN_AUTO_PROJECT_TAG", "true")
        monkeypatch.chdir(repo)
        run(vault, "remember", "CI", "--predicate", "runner", "--value", "buildkite")
        run(vault, "read", "CI")
        out = capsys.readouterr().out
        assert "project/acme-rocket" in out


class TestDeprecationReason:
    def test_reason_recorded_on_replaced_fact(self, vault, capsys):
        run(vault, "remember", "DB", "--predicate", "storage", "--value", "MySQL")
        run(
            vault,
            "remember",
            "DB",
            "--predicate",
            "storage",
            "--value",
            "PostgreSQL",
            "--reason",
            "user correction: we migrated in July",
        )
        capsys.readouterr()
        from marvin.config import MarvinSettings
        from marvin.service import MarvinService

        service = MarvinService(
            MarvinSettings(vault_path=vault, state_dir=vault / ".state", embedding_provider="hash")
        )
        try:
            note = service.get_note("DB")
            deprecated = [f for f in note.metadata.facts if f.deprecated]
            assert len(deprecated) == 1
            assert deprecated[0].deprecated_reason == "user correction: we migrated in July"
        finally:
            service.close()


class TestHooksInstaller:
    def test_install_creates_and_merges(self, vault, capsys, monkeypatch, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        settings_path = project / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps(
                {
                    "model": "opus",
                    "hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "x"}]}]},
                }
            )
        )
        monkeypatch.chdir(project)

        assert run(vault, "hooks", "install", "--host", "claude") == 0
        data = json.loads(settings_path.read_text())
        assert data["model"] == "opus"  # untouched keys survive
        assert data["hooks"]["PreToolUse"]  # unrelated hooks survive
        commands = [
            h["command"]
            for event in ("SessionStart", "UserPromptSubmit")
            for entry in data["hooks"][event]
            for h in entry["hooks"]
        ]
        assert "marvin hook session-start" in commands
        assert "marvin hook user-prompt" in commands

        capsys.readouterr()
        assert run(vault, "hooks", "install", "--host", "claude") == 0
        assert "(already installed)" in capsys.readouterr().out
        data2 = json.loads(settings_path.read_text())
        assert data2["hooks"]["SessionStart"] == data["hooks"]["SessionStart"]  # idempotent

    def test_unwired_host_points_to_show(self, vault, capsys):
        assert run(vault, "hooks", "install", "--host", "codex") == 2
        assert "hooks show --host codex" in capsys.readouterr().out

    def test_show_claude_prints_snippet(self, vault, capsys):
        assert run(vault, "hooks", "show", "--host", "claude") == 0
        out = capsys.readouterr().out
        assert "SessionStart" in out and "marvin hook session-start" in out

"""Tests for `marvin skill show|install` (the bundled agent skill)."""

from __future__ import annotations

from pathlib import Path

import pytest

from marvin import cli


class TestSkillShow:
    def test_prints_frontmatter_and_body(self, capsys):
        assert cli.main(["skill", "show"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("---")
        assert "name: marvin-memory" in out
        assert "description: Use when" in out
        assert "marvin_record_template_use" in out

    def test_does_not_create_a_vault(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert cli.main(["skill", "show"]) == 0
        assert not (tmp_path / "marvin_vault").exists()


class TestSkillInstall:
    def test_installs_to_project_skills_dir(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert cli.main(["skill", "install"]) == 0
        dest = tmp_path / ".claude" / "skills" / "marvin-memory" / "SKILL.md"
        assert dest.exists()
        assert "name: marvin-memory" in dest.read_text(encoding="utf-8")
        out = capsys.readouterr().out
        assert "installed:" in out
        assert str(dest.parent) in out

    def test_install_is_idempotent(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert cli.main(["skill", "install"]) == 0
        assert cli.main(["skill", "install"]) == 0  # second run overwrites cleanly

    def test_target_flag(self, tmp_path, capsys):
        target = tmp_path / "custom-skills"
        assert cli.main(["skill", "install", "--target", str(target)]) == 0
        assert (target / "marvin-memory" / "SKILL.md").exists()

    def test_user_flag(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cli.main(["skill", "install", "--user"]) == 0
        assert (tmp_path / ".claude" / "skills" / "marvin-memory" / "SKILL.md").exists()


class TestSkillQuality:
    """Frontmatter contract checks (agentskills.io specification)."""

    @pytest.fixture()
    def skill_text(self) -> str:
        root = Path(__file__).resolve().parents[1]
        return (root / "src" / "marvin" / "skill" / "SKILL.md").read_text(encoding="utf-8")

    def test_frontmatter_under_1024_chars(self, skill_text):
        frontmatter = skill_text.split("---")[1]
        assert len(frontmatter) <= 1024

    def test_description_is_trigger_only_third_person(self, skill_text):
        frontmatter = skill_text.split("---")[1]
        desc = [line for line in frontmatter.splitlines() if line.startswith("description:")][0]
        assert "Use when" in desc
        assert " I " not in desc and not desc.startswith("description: I ")

    def test_covers_both_surfaces(self, skill_text):
        # Every taught action names the MCP tool and the CLI form.
        for pair in [
            ("marvin_remember_semantic", "marvin remember"),
            ("marvin_store_procedure", "marvin procedure"),
            ("marvin_log_episode", "marvin episode"),
            ("marvin_reflect", "marvin reflect"),
            ("marvin_match_template", "marvin template match"),
            ("marvin_record_template_use", "marvin template used"),
            ("marvin_prepare_session", "marvin session prepare"),
        ]:
            assert pair[0] in skill_text and pair[1] in skill_text, pair

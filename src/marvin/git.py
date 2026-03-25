from __future__ import annotations

from pathlib import Path

from git import Repo


class GitManager:
    """Manages Git-backed agentic worktrees for the Marvin vault."""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self._ensure_repo()

    def _ensure_repo(self) -> None:
        """Initialize the Git repository if it doesn't exist."""
        if not (self.vault_path / ".git").exists():
            repo = Repo.init(self.vault_path)
            # Create an initial commit so branches can be made
            readme_path = self.vault_path / "README.md"
            if not readme_path.exists():
                readme_path.write_text(
                    "# Marvin Memory Vault\n\nAuto-generated Git-backed vault."
                )
            repo.index.add(["README.md"])
            repo.index.commit("chore: init vault")

    @property
    def repo(self) -> Repo:
        return Repo(self.vault_path)

    def current_branch(self) -> str:
        """Returns the active branch name."""
        return self.repo.active_branch.name

    def commit(self, message: str) -> None:
        """Adds all tracked/untracked markdown files and commits them."""
        repo = self.repo
        # Add all markdown files
        repo.git.add(A=True)

        # Check if there are actual changes
        if repo.is_dirty() or repo.untracked_files:
            # We configure author on the fly to avoid global config issues
            with repo.config_writer() as cw:
                if not cw.has_section("user"):
                    cw.set_value("user", "name", "Marvin Brain")
                    cw.set_value("user", "email", "marvin@agent.local")

            repo.index.commit(message)

    def create_worktree(self, branch_name: str) -> str:
        """Creates and checks out a new branch for an agent session."""
        repo = self.repo

        if branch_name in repo.heads:
            repo.heads[branch_name].checkout()
        else:
            new_branch = repo.create_head(branch_name)
            new_branch.checkout()

        return branch_name

    def merge_worktree(
        self, source_branch: str, target_branch: str = "main"
    ) -> dict[str, str]:
        """Merges a worktree branch into the target branch.

        If there are conflicts, they need to be resolved.
        """
        repo = self.repo

        if target_branch not in repo.heads:
            # Maybe main is master
            target_branch = "master" if "master" in repo.heads else target_branch
            if target_branch not in repo.heads:
                target_branch = repo.active_branch.name

        # Checkout target
        repo.heads[target_branch].checkout()

        try:
            # Try a clean merge with --no-ff to visually preserve the worktree history
            repo.git.merge(source_branch, "--no-ff", "-m", f"Merge worktree {source_branch}")
            return {
                "status": "success",
                "message": f"Merged {source_branch} into {target_branch}",
            }
        except Exception as e:
            # Conflict occurred
            # For MVP: Abort merge if conflict. The Brain worker needs to handle this.
            try:
                repo.git.merge("--abort")
            except Exception:
                pass
            return {"status": "conflict", "message": str(e)}

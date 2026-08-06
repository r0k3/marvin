"""Project identity: a stable tag derived from the working directory's repo.

The vault is user/agent-global by design — cross-project memory sharing
is a feature. Tags are how recall narrows when it wants to: every write
made from inside a git repository is auto-tagged ``project/<slug>`` (see
``MARVIN_AUTO_PROJECT_TAG``), with the slug derived from the origin
remote so clones of the same repo share a tag regardless of local path.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REMOTE_RE = re.compile(r"^(?:git@|ssh://git@|https?://)([^/:]+)[:/]+(.+?)(?:\.git)?/?$")


def _clean(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower()
    return slug or "unnamed"


def _git(cwd: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def repo_tag(cwd: Path | None = None) -> str | None:
    """``project/<owner>-<repo>`` tag for the repository containing ``cwd``.

    Derived from the origin remote URL; falls back to the repository's
    top-level directory name when there is no remote. Returns ``None``
    outside a git repository or when git is unavailable.
    """
    base = Path.cwd() if cwd is None else cwd

    url = _git(base, "remote", "get-url", "origin")
    if url:
        match = _REMOTE_RE.match(url.strip())
        if match:
            parts = [p for p in match.group(2).split("/") if p]
            slug = "-".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            return f"project/{_clean(slug)}"

    toplevel = _git(base, "rev-parse", "--show-toplevel")
    if toplevel:
        return f"project/{_clean(Path(toplevel).name)}"
    return None

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_project_tag(monkeypatch: pytest.MonkeyPatch):
    """Keep the suite hermetic (and fast): the repo the tests run from must
    not leak its own ``project/...`` tag — or a git subprocess call — into
    every service write. Dedicated project-tag tests re-enable via env."""
    monkeypatch.setenv("MARVIN_AUTO_PROJECT_TAG", "false")

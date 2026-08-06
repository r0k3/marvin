"""Guard for optional-dependency ("extras") imports.

Modules that need an extra import its symbols under ``try/except
ImportError`` (binding ``None`` when absent) and call :func:`require`
right before first use, so the base install stays lean and failures are
actionable instead of ``TypeError: 'NoneType' object is not callable``.
"""

from __future__ import annotations


def require(feature: str, symbol: object, extra: str) -> None:
    """Raise a helpful error if ``symbol`` is ``None`` (extra not installed)."""
    if symbol is None:
        raise RuntimeError(
            f"{feature} requires the '{extra}' extra. "
            f"Install with: uv tool install 'marvin-memory[{extra}]' "
            f"(pip: pip install 'marvin-memory[{extra}]')"
        )

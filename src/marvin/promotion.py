"""Deterministic promotion signals: which moments deserve a memory write.

The detectors are cheap offline regexes — Marvin never spends an LLM
call deciding *whether* something is worth remembering. When a signal
fires, the *host agent* (which already holds the full conversation in
context) is nudged to do the extraction and the write; Marvin's
soft-deprecation machinery then handles the correction semantics.
"""

from __future__ import annotations

import re

# High-precision cues that a user message corrects something previously
# believed, stored, or done. Names are stable API — hooks and tests
# refer to them. Precision beats recall here: a false nudge on every
# other prompt trains the agent (and the user) to ignore the signal.
_CORRECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "leading-no": re.compile(r"^(no|nope|wrong|incorrect)\b[,.!:; ]", re.IGNORECASE),
    "thats-wrong": re.compile(r"\bthat'?s (wrong|not right|incorrect|not what i)\b", re.IGNORECASE),
    "actually-is": re.compile(
        r"\bactually,? (it'?s|it is|we (use|need|want)|i (use|want|meant)|the)\b",
        re.IGNORECASE,
    ),
    "i-said": re.compile(r"\bi (said|meant|already told you)\b", re.IGNORECASE),
    "explicit-correction": re.compile(r"\b(correction|to correct|let me correct)\b", re.IGNORECASE),
    "no-longer": re.compile(r"\bwe (don'?t|do not|no longer) use\b", re.IGNORECASE),
    "stop-using": re.compile(r"\bstop (using|doing)\b", re.IGNORECASE),
    "should-be-not": re.compile(r"\bshould be\b[^.?!]{0,60}\bnot\b", re.IGNORECASE),
}


def detect_correction(text: str) -> list[str]:
    """Names of the correction cues present in ``text`` (empty list = none)."""
    stripped = text.strip()
    if not stripped:
        return []
    return [name for name, pattern in _CORRECTION_PATTERNS.items() if pattern.search(stripped)]

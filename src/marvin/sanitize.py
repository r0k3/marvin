"""Write-path hygiene: invisible-Unicode stripping and LLM-output scanning.

Two deliberately different policies:

- :func:`strip_invisible` runs on **every** vault write (title and body).
  Invisible code points — zero-width characters, bidi controls, the
  Unicode tag block, filler characters — have no legitimate role in
  memory notes and are the classic channel for smuggling instructions a
  human reviewer cannot see (CVE-2021-42574 and friends). Stripping them
  never changes what a reader perceives.
- :func:`scan_injection` runs **only** on machine-generated text (the
  sleep pass's LLM output) before it is persisted. Agent- and
  user-authored writes are never content-filtered — a security engineer
  must be able to store a note *about* prompt injection — but text an
  LLM produced during consolidation gets no such presumption: a flagged
  item is dropped and logged, because once stored it will be replayed
  into future agent context.
"""

from __future__ import annotations

import re

# Invisible code points stripped from every write.
_INVISIBLE = (
    "​‌‍‎‏"  # zero-width space/joiners, LRM/RLM
    "‪‫‬‭‮"  # bidi embeddings and overrides
    "⁠⁡⁢⁣⁤"  # word joiner, invisible operators
    "⁦⁧⁨⁩"  # bidi isolates
    "﻿"  # BOM / zero-width no-break space
    "ᅟᅠㅤﾠ"  # Hangul fillers
)
_INVISIBLE_RE = re.compile(f"[{_INVISIBLE}]|[\U000e0000-\U000e007f]")  # + Unicode tag block


def strip_invisible(text: str) -> str:
    """Remove invisible/bidi-control code points; visible text is untouched."""
    return _INVISIBLE_RE.sub("", text)


# Injection shapes in machine-generated text. Named so logs and tests can
# refer to them; precision matters less than for the correction detector
# (this only ever filters LLM output, never human/agent-authored notes).
_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction-override": re.compile(
        r"\b(ignore|disregard|forget)\b[^.\n]{0,40}"
        r"\b(previous|prior|above|all|earlier)\b[^.\n]{0,20}\binstructions?\b",
        re.IGNORECASE,
    ),
    "role-reassignment": re.compile(
        r"\byou are now\b|\bnew persona\b|\bact as\b[^.\n]{0,30}\b(admin|root|system)\b",
        re.IGNORECASE,
    ),
    "fake-transcript-prefix": re.compile(r"^\s*(system|assistant|human)\s*:", re.IGNORECASE),
    "chat-template-token": re.compile(
        r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|system\|>|\[INST\]|\[/INST\]"
    ),
    "tool-call-token": re.compile(r"<(function_call|tool_call|invoke)\b", re.IGNORECASE),
}


def scan_injection(text: str) -> list[str]:
    """Names of injection shapes present in ``text`` (empty list = clean)."""
    if not text:
        return []
    return sorted(name for name, pattern in _INJECTION_PATTERNS.items() if pattern.search(text))

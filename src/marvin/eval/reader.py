"""Reader (answer-generation) stage for the LongMemEval-S QA arm.

Marvin's retrieval is effectively saturated on LongMemEval-S (~99.6%
recall_any@5), so the published QA-accuracy frontier (~90-93% on the
benchmark) is gated by the *reader* model, not retrieval. This module
wires a small, local reader on top of the existing hybrid retrieval so
we can report end-to-end QA accuracy under the canonical protocol.

Design choices (all from Wu et al. 2410.10813, kept parsimonious):

- **JSON + Chain-of-Note** is the paper's best reading strategy (Fig 6:
  GPT-4o oracle 0.862 -> 0.924; Llama-3.1-8B 0.710 -> 0.756). We present
  the retrieved memories as a JSON array and ask the model to note each
  entry's relevance before answering. This is prompt-only -- no extra
  infrastructure.
- **Abstention is permitted.** 30 of the 500 questions are unanswerable
  (`_abs`); a reader that never abstains forfeits 6% of the benchmark.
- **Memory entries are kind-labelled** (episodic / semantic / procedural
  / reflective). For the raw-episode arm everything is "episodic"; the
  label is the forward hook for feeding consolidated typed notes to the
  reader without changing this code.

The engine mirrors :class:`marvin.consolidation.ConsolidationEngine`'s
litellm calling convention so the two read alike and share the local
ollama configuration.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence

from litellm import completion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_READER_MODEL = "ollama/qwen3.6:35b-a3b-q4_K_M"

# Total characters of retrieved context handed to the reader, and the cap
# per entry. Retrieval is near-oracle (gold session usually rank 1), so the
# priority is including each retrieved session *whole* -- a too-small
# per-entry cap truncates the answer span out of a long session and forces a
# spurious abstention. ~64k chars ≈ 16k tokens, well within the reader's
# window and still far below the 115k-token full-history baseline.
DEFAULT_MAX_CONTEXT_CHARS = 64_000
DEFAULT_MAX_ENTRY_CHARS = 16_000

_ANSWER_MARKER = "ANSWER:"


class ReaderContextItem(BaseModel):
    """One retrieved memory entry as seen by the reader."""

    kind: str  # episodic | semantic | procedural | reflective
    session_id: str
    date: str  # raw dataset timestamp, e.g. "2023/05/30 (Tue) 23:40"; may be ""
    body: str


class ReaderResult(BaseModel):
    """Parsed reader output."""

    answer: str
    raw_response: str = ""
    error: bool = False


def build_reader_context(
    items: Sequence[ReaderContextItem],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_entry_chars: int = DEFAULT_MAX_ENTRY_CHARS,
) -> str:
    """Render retrieved entries as a JSON array string (rank order preserved).

    Entries are added in rank order until the character budget is
    exhausted; each entry's body is truncated to ``max_entry_chars``.
    Returning a JSON string (rather than free text) is the "JSON" half of
    the paper's JSON+Chain-of-Note strategy.
    """
    rendered: list[dict[str, object]] = []
    total = 0
    for i, item in enumerate(items, start=1):
        body = item.body.strip()
        if len(body) > max_entry_chars:
            body = body[:max_entry_chars] + " […]"
        entry = {
            "index": i,
            "type": item.kind,
            "date": item.date,
            "content": body,
        }
        # Cheap budget check on the serialized entry length.
        cost = len(body) + len(item.date) + len(item.kind) + 40
        if rendered and total + cost > max_context_chars:
            break
        rendered.append(entry)
        total += cost
    return json.dumps(rendered, ensure_ascii=False, indent=2)


def _build_prompt(question: str, context_json: str, question_date: str) -> str:
    today = question_date.strip() or "unknown"
    return f"""You are a precise long-term memory assistant. Answer the QUESTION using \
ONLY the retrieved MEMORY entries below. Today's date is {today}.

MEMORY is a JSON array of retrieved entries. Each entry has a "type"
(episodic = a past conversation; semantic = a distilled fact; procedural =
a learned rule; reflective = a higher-level insight), a "date", and its
"content".

MEMORY:
{context_json}

Instructions:
1. Under a heading "NOTES:", for each RELEVANT entry write one short line
   "[index] <the fact it provides>". Skip irrelevant entries.
2. Reason about the question from those facts. Use the "date" fields for
   any time-based reasoning, and when facts change over time prefer the
   most recent entry.
3. Output the final answer on a single line beginning with "{_ANSWER_MARKER}".
   Keep it short and direct.
4. If MEMORY lacks the information needed, output exactly:
   "{_ANSWER_MARKER} I don't have enough information to answer this question."

QUESTION: {question}
"""


def _parse_answer(text: str) -> str:
    """Extract the final answer following the last ANSWER: marker.

    Falls back to the whole (stripped) response when the marker is absent
    so a model that ignores the format still yields a gradeable hypothesis.
    """
    lower = text.lower()
    idx = lower.rfind(_ANSWER_MARKER.lower())
    if idx == -1:
        return text.strip()
    return text[idx + len(_ANSWER_MARKER) :].strip()


class ReaderEngine:
    """litellm-backed answer generator (local ollama by default)."""

    def __init__(
        self,
        model: str = DEFAULT_READER_MODEL,
        api_base: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        think: bool = False,
        timeout: float = 120.0,
        num_retries: int = 0,
        hard_timeout: float = 130.0,
    ):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hard_timeout = hard_timeout
        # A per-request timeout + retries so a single stalled remote call
        # (e.g. a hung xAI/OpenAI HTTP request) cannot block the whole
        # benchmark indefinitely -- litellm has no default timeout.
        self.timeout = timeout
        self.num_retries = num_retries
        # Reasoning models (e.g. Qwen3) emit hidden reasoning tokens that are
        # stripped from the content but still counted against max_tokens,
        # which on hard questions consumes the whole budget and yields an
        # empty/truncated answer. The ``/no_think`` prompt switch is *ignored*
        # by Qwen3 via ollama; the native ``think=False`` option is what
        # actually disables it (~10x faster, no truncation). We keep the
        # *explicit* Chain-of-Note in the prompt -- that is the paper's actual
        # reasoning mechanism and survives think=False.
        self.think = think

    def answer(
        self,
        question: str,
        context: str,
        *,
        question_date: str = "",
    ) -> ReaderResult:
        """Generate an answer for ``question`` from a pre-built ``context``."""
        prompt = _build_prompt(question, context, question_date)
        extra: dict[str, object] = {}
        # Disable hidden reasoning on local Qwen3 (ollama) -- see __init__.
        if not self.think and self.model.startswith("ollama/"):
            extra["think"] = False

        out: dict[str, object] = {}

        def _call() -> None:
            try:
                resp = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=self.api_base,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    num_retries=self.num_retries,
                    **extra,
                )
                out["raw"] = resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - reported via the error result
                out["err"] = e

        # Hard wall-clock cap via a daemon thread. litellm's own ``timeout`` is
        # unreliable for some providers (it can fire far later than requested),
        # so a single stalled remote call must not block the whole benchmark.
        # On timeout we abandon the daemon thread (it dies with the process)
        # and return an error result so the run continues.
        worker = threading.Thread(target=_call, daemon=True)
        worker.start()
        worker.join(self.hard_timeout)
        if worker.is_alive():
            logger.warning("Reader call exceeded hard timeout %.0fs; abandoning", self.hard_timeout)
            return ReaderResult(answer="", raw_response="", error=True)
        if "err" in out:
            logger.warning("Error during reader generation: %s", out["err"])
            return ReaderResult(answer="", raw_response="", error=True)
        raw = str(out.get("raw", ""))
        return ReaderResult(answer=_parse_answer(raw), raw_response=raw, error=False)

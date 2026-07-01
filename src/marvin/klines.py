"""K-line procedural templates.

A *K-line template* pairs a procedural plan with explicit machinery for selecting
it under context, following Minsky's partial-reactivation principle: multi-
dimensional trigger conditions (intents, styles, entity types, keywords) scored
with fixed weights, intent as a hard gate, plus adaptive utility (a usage count
and an effectiveness EMA) so effective templates are preferred.

The trigger / slot / plan / failure-mode structure lives in the *body* of a
procedural note (Markdown sections, parsed back when matching). The adaptive
utility metadata lives in frontmatter (``usage_count`` / ``effectiveness`` on
:class:`marvin.models.NoteMetadata`), since it mutates as templates are used.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

INTENTS_HEADING = "Intents"
STYLES_HEADING = "Styles"
ENTITY_TYPES_HEADING = "Entity Types"
TRIGGERS_HEADING = "Triggers"
SLOTS_HEADING = "Slots"
PLAN_HEADING = "Plan"
FAILURE_MODES_HEADING = "Failure Modes"

# Scoring weights (paper): intent dominates as the strongest predictor of
# strategy; style and entity-type contribute a quarter each; keyword coverage
# adds a capped bonus so a keyword-only template can still partially activate.
INTENT_WEIGHT = 0.5
STYLE_WEIGHT = 0.25
ENTITY_TYPE_WEIGHT = 0.25
KEYWORD_BONUS = 0.25


@dataclass(frozen=True)
class KLineTemplate:
    """Parsed in-memory form of a K-line template note."""

    title: str
    intents: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    trigger_phrases: tuple[str, ...] = ()
    slots: tuple[str, ...] = ()
    plan: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()

    def is_complete(self) -> bool:
        """Selectable templates need a plan and at least one trigger dimension."""

        has_trigger = bool(self.intents or self.styles or self.entity_types or self.trigger_phrases)
        return has_trigger and bool(self.plan)


@dataclass(frozen=True)
class MatchSignal:
    """The context a template is scored against.

    ``intent`` is a single classified intent for the current turn; ``styles``
    and ``entity_types`` are sets describing it; ``text`` is free text used for
    keyword matching. All are optional -- a bare ``MatchSignal(text=...)``
    degrades to keyword-coverage scoring.
    """

    intent: str = ""
    styles: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    text: str = ""


@dataclass(frozen=True)
class TemplateMatch:
    """One ranked result from :func:`MarvinService.match_template`."""

    template: KLineTemplate
    score: float
    matched_phrases: tuple[str, ...]
    note_path: str
    usage_count: int = 0
    effectiveness: float = 0.0


def score_template(template: KLineTemplate, signal: MatchSignal) -> tuple[float, tuple[str, ...]]:
    """Score a template against a context signal (Minsky's partial reactivation).

    ``score = 0.5*intent + 0.25*style + 0.25*entity_type + keyword-coverage
    bonus``. Intent is a hard gate: if the template names intents and none
    matches the signal, the score is 0 regardless of the other dimensions.
    Returns ``(score, matched_keywords)``.
    """

    # Hard intent gate: a template that specifies intents only fires on a match.
    if template.intents and signal.intent not in template.intents:
        return 0.0, ()

    score = 0.0
    if signal.intent and signal.intent in template.intents:
        score += INTENT_WEIGHT
    if set(signal.styles) & set(template.styles):
        score += STYLE_WEIGHT
    if set(signal.entity_types) & set(template.entity_types):
        score += ENTITY_TYPE_WEIGHT

    matched: tuple[str, ...] = ()
    if template.trigger_phrases:
        needle = signal.text.casefold()
        matched = tuple(p for p in template.trigger_phrases if p.casefold() in needle)
        if matched:
            score += KEYWORD_BONUS * (len(matched) / len(template.trigger_phrases))

    return score, matched


def next_effectiveness(current: float, *, success: bool, alpha: float = 0.3) -> float:
    """ACT-R-style utility update: an exponential moving average of success."""

    target = 1.0 if success else 0.0
    return alpha * target + (1.0 - alpha) * current


def render_template_body(
    *,
    intents: Iterable[str] = (),
    styles: Iterable[str] = (),
    entity_types: Iterable[str] = (),
    trigger_phrases: Iterable[str] = (),
    slots: Iterable[str] = (),
    plan: Iterable[str] = (),
    failure_modes: Iterable[str] = (),
    intro: str = "",
) -> str:
    """Render a K-line template into Markdown body sections.

    Sections are deterministic and parser-friendly. Empty sections are omitted
    so a template with only triggers + plan doesn't leak empty headings.
    """

    parts: list[str] = []
    if intro.strip():
        parts.append(intro.strip())
    parts.append(_render_bullets(INTENTS_HEADING, intents))
    parts.append(_render_bullets(STYLES_HEADING, styles))
    parts.append(_render_bullets(ENTITY_TYPES_HEADING, entity_types))
    parts.append(_render_bullets(TRIGGERS_HEADING, trigger_phrases))
    parts.append(_render_bullets(SLOTS_HEADING, slots))
    parts.append(_render_numbered(PLAN_HEADING, plan))
    parts.append(_render_bullets(FAILURE_MODES_HEADING, failure_modes))
    return "\n\n".join(part for part in parts if part).strip()


def parse_template_body(*, title: str, body: str) -> KLineTemplate | None:
    """Parse the body of a procedural note back into a :class:`KLineTemplate`.

    Returns ``None`` when the body has no trigger dimension and no plan -- i.e.,
    the note is a regular procedural note, not a K-line template -- so
    ``match_template`` can ignore it without raising.
    """

    intents = _extract_bullets(body, INTENTS_HEADING)
    styles = _extract_bullets(body, STYLES_HEADING)
    entity_types = _extract_bullets(body, ENTITY_TYPES_HEADING)
    triggers = _extract_bullets(body, TRIGGERS_HEADING)
    slots = _extract_bullets(body, SLOTS_HEADING)
    plan = _extract_numbered(body, PLAN_HEADING)
    failure_modes = _extract_bullets(body, FAILURE_MODES_HEADING)
    if not (intents or styles or entity_types or triggers) and not plan:
        return None
    return KLineTemplate(
        title=title,
        intents=tuple(intents),
        styles=tuple(styles),
        entity_types=tuple(entity_types),
        trigger_phrases=tuple(triggers),
        slots=tuple(slots),
        plan=tuple(plan),
        failure_modes=tuple(failure_modes),
    )


# ---------------------------------------------------------------------------
# Internal rendering / parsing helpers
# ---------------------------------------------------------------------------


def _render_bullets(heading: str, items: Iterable[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    body = "\n".join(f"- {item}" for item in cleaned)
    return f"## {heading}\n{body}"


def _render_numbered(heading: str, items: Iterable[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    body = "\n".join(f"{i}. {item}" for i, item in enumerate(cleaned, start=1))
    return f"## {heading}\n{body}"


def _section_lines(body: str, heading: str) -> list[str]:
    marker = f"## {heading}"
    if marker not in body:
        return []
    _, _, tail = body.partition(marker)
    out: list[str] = []
    for raw_line in tail.splitlines()[1:]:
        if raw_line.startswith("## "):
            break
        out.append(raw_line.rstrip())
    return out


def _extract_bullets(body: str, heading: str) -> list[str]:
    items: list[str] = []
    for line in _section_lines(body, heading):
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _extract_numbered(body: str, heading: str) -> list[str]:
    items: list[str] = []
    for line in _section_lines(body, heading):
        stripped = line.strip()
        if not stripped:
            continue
        # Strip a leading "<digits>. " prefix if present; tolerate
        # bullets (- ) too so a hand-edited plan still parses.
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
            continue
        prefix_end = 0
        while prefix_end < len(stripped) and stripped[prefix_end].isdigit():
            prefix_end += 1
        if prefix_end > 0 and stripped[prefix_end : prefix_end + 2] == ". ":
            items.append(stripped[prefix_end + 2 :].strip())
        else:
            items.append(stripped)
    return items


__all__ = [
    "ENTITY_TYPES_HEADING",
    "FAILURE_MODES_HEADING",
    "INTENTS_HEADING",
    "KLineTemplate",
    "MatchSignal",
    "PLAN_HEADING",
    "SLOTS_HEADING",
    "STYLES_HEADING",
    "TRIGGERS_HEADING",
    "TemplateMatch",
    "next_effectiveness",
    "parse_template_body",
    "render_template_body",
    "score_template",
]

"""Compile the vault into a portable Agent Skills bundle (`marvin skill export`).

The book-to-skill move applied to memory: a deterministic, zero-LLM
compiler that distills the four memory kinds into a token-budgeted skill
any Agent-Skills-reading host can load — Marvin's "offline mode". Facts
become a glossary, procedures and K-line templates become patterns,
reflective insights become a cheatsheet, and episodes a compact history.
The live system (CLI / MCP) stays the richer interface; the export is a
snapshot for hosts or sessions where no Marvin runs.
"""

from __future__ import annotations

import re

from .klines import parse_template_body
from .models import MemoryKind, NoteRecord
from .service import MarvinService

_TRUNCATED = (
    "\n> (truncated at the character budget — the live vault has more: `marvin search <query>`)"
)


def _cap(sections: list[str], max_chars: int) -> str:
    out: list[str] = []
    used = 0
    for section in sections:
        cost = len(section) + 1
        if used + cost > max_chars:
            out.append(_TRUNCATED)
            break
        out.append(section)
        used += cost
    return ("\n".join(out)).strip() + "\n"


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower()
    return slug or "vault"


def default_bundle_name(tag: str | None) -> str:
    """``project/acme-rocket`` -> ``acme-rocket-memory``; no tag -> ``vault-memory``."""
    if tag:
        return f"{_slug(tag.rsplit('/', 1)[-1])}-memory"
    return "vault-memory"


def build_skill_bundle(
    service: MarvinService,
    *,
    name: str | None = None,
    tag: str | None = None,
    max_chars: int = 8000,
) -> dict[str, str]:
    """Render ``{filename: content}`` for the bundle. Deterministic, no LLM."""
    bundle_name = name or default_bundle_name(tag)

    def keep(note: NoteRecord) -> bool:
        return tag is None or tag in note.metadata.tags

    glossary: list[str] = []
    fact_count = 0
    for note in service.vault.list_notes(MemoryKind.SEMANTIC):
        if not keep(note):
            continue
        active = [fact for fact in note.metadata.facts if not fact.deprecated]
        if not active:
            continue
        lines = [f"### {note.metadata.title}"]
        for fact in sorted(active, key=lambda f: -f.confidence):
            lines.append(
                f"- **{fact.predicate}**: {fact.value}"
                f" _(confidence {fact.confidence:.1f}, {fact.aspect.value})_"
            )
            fact_count += 1
        glossary.append("\n".join(lines) + "\n")

    patterns: list[str] = []
    pattern_count = 0
    for note in service.vault.list_notes(MemoryKind.PROCEDURAL):
        if not keep(note):
            continue
        pattern_count += 1
        template = parse_template_body(title=note.metadata.title, body=note.body)
        if template is not None and template.plan:
            lines = [f"### {note.metadata.title} (response strategy)"]
            triggers: list[str] = []
            if template.intents:
                triggers.append(f"intents: {', '.join(template.intents)}")
            if template.trigger_phrases:
                triggers.append(f"cues: {', '.join(template.trigger_phrases)}")
            if triggers:
                lines.append(f"- When — {'; '.join(triggers)}")
            lines.append("- Plan:")
            lines.extend(f"    {i}. {step}" for i, step in enumerate(template.plan, 1))
            if template.failure_modes:
                lines.append(f"- Avoid: {'; '.join(template.failure_modes)}")
            patterns.append("\n".join(lines) + "\n")
        else:
            patterns.append(f"### {note.metadata.title}\n\n{note.body.strip()}\n")

    cheats: list[str] = [
        f"### {note.metadata.title}\n\n{note.body.strip()}\n"
        for note in service.vault.list_notes(MemoryKind.REFLECTIVE)
        if keep(note)
    ]

    episodes = sorted(
        (note for note in service.vault.list_notes(MemoryKind.EPISODIC) if keep(note)),
        key=lambda note: note.metadata.updated_at,
        reverse=True,
    )
    history = [
        f"- **{note.metadata.title}**"
        f" ({note.metadata.updated_at.date().isoformat()}):"
        f" {(note.body.strip().splitlines() or [''])[0]}"
        for note in episodes
    ]

    files: dict[str, str] = {}
    sections: list[tuple[str, str, str, int]] = []  # (file, title, purpose, count)
    if glossary:
        files["glossary.md"] = f"# Glossary — durable facts\n\n{_cap(glossary, max_chars)}"
        sections.append(("glossary.md", "Glossary", "durable facts and decisions", fact_count))
    if patterns:
        files["patterns.md"] = (
            f"# Patterns — procedures and strategies\n\n{_cap(patterns, max_chars)}"
        )
        sections.append(
            ("patterns.md", "Patterns", "procedures and response strategies", pattern_count)
        )
    if cheats:
        files["cheatsheet.md"] = (
            f"# Cheatsheet — lessons and principles\n\n{_cap(cheats, max_chars)}"
        )
        sections.append(("cheatsheet.md", "Cheatsheet", "cross-cutting lessons", len(cheats)))
    if history:
        files["history.md"] = f"# History — what happened\n\n{_cap(history, max_chars)}"
        sections.append(("history.md", "History", "recent episodes, newest first", len(episodes)))

    scope = f" (scope: {tag})" if tag else ""
    index_lines = [
        f"- `{fname}` — {purpose} ({count} entries)" for fname, _, purpose, count in sections
    ] or ["- (the vault was empty — nothing to compile)"]
    files["SKILL.md"] = (
        "---\n"
        f"name: {bundle_name}\n"
        "description: "
        f"Distilled memory from a Marvin vault{scope}: consult before answering"
        " questions about this user's or project's conventions, decisions,"
        " procedures, or history. Facts in glossary.md, strategies in"
        " patterns.md, lessons in cheatsheet.md.\n"
        "---\n\n"
        f"# {bundle_name}\n\n"
        "A compiled snapshot of a [Marvin](https://github.com/r0k3/marvin)"
        f" memory vault{scope}. Read the file that matches the question:\n\n"
        + "\n".join(index_lines)
        + "\n\nThis bundle is a snapshot. If the live system is available"
        " (`marvin` CLI or `marvin_*` MCP tools), prefer querying it —"
        " `marvin search <query>` — and treat this bundle as fallback.\n"
    )
    return files

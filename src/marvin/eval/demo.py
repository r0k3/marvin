"""Midsummer Night's Dream demo for Marvin.

Runs Marvin's four-memory retrieval over ``examples/demo_vault`` -- a real,
hand-authored vault of *A Midsummer Night's Dream* (episodic scenes,
semantic facts about the fairies, a procedural play-analysis strategy, a
reflective insight on Athenian law) -- and prints an annotated transcript
suitable for the paper's demonstration appendix.

    python -m marvin.eval.demo                      # real semantic embedder
    python -m marvin.eval.demo --embedding-provider hash

The canonical query set (``DEMO_QUERIES``) is reused by
``tests/test_demo_midsummer.py`` so the demo is regression-guarded.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from marvin.config import MarvinSettings
from marvin.models import MemoryKind, SearchHit
from marvin.service import MarvinService

_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


def demo_vault_path() -> Path:
    """Path to the bundled Midsummer demo vault (repo ``examples/demo_vault``)."""
    return Path(__file__).resolve().parents[3] / "examples" / "demo_vault"


# (question, the note title it should surface, the memory kind that note lives in)
DEMO_QUERIES: list[tuple[str, str, MemoryKind]] = [
    (
        "What was Puck's mistake with the love potion?",
        "Love Potion Error",
        MemoryKind.SEMANTIC,
    ),
    (
        "Why did Lysander fall in love with Helena instead of Hermia?",
        "Result of Puck's Mistake",
        MemoryKind.SEMANTIC,
    ),
    (
        "Why are Oberon and Titania in conflict?",
        "Fairy Conflict",
        MemoryKind.SEMANTIC,
    ),
    (
        "What happened to Bottom in the forest?",
        "Bottom's Transformation",
        MemoryKind.EPISODIC,
    ),
    (
        "How should I analyze a play scene with an authority figure?",
        "Play Analysis Strategy",
        MemoryKind.PROCEDURAL,
    ),
    (
        "What does the play say about Athenian law and women?",
        "The Harshness of Athenian Law",
        MemoryKind.REFLECTIVE,
    ),
]


def build_demo_service(*, embedding_provider: str, state_dir: Path) -> MarvinService:
    """Index the demo vault into a throwaway state dir and return the service.

    ``sync`` only reads the vault (it writes to the derived index under
    ``state_dir``), so pointing ``vault_path`` at the bundled demo vault does
    not mutate it.
    """
    service = MarvinService(
        MarvinSettings(
            vault_path=demo_vault_path(),
            state_dir=state_dir,
            embedding_provider=embedding_provider,
        )
    )
    service.sync()
    return service


def _snippet(service: MarvinService, hit: SearchHit, n: int = 110) -> str:
    """First human-readable content line of the retrieved note (summary/fact),
    with wikilink brackets stripped for the transcript."""
    note = service.vault.find_note(title=hit.title, kind=hit.kind)
    if note is None:
        return ""
    for line in note.body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        text = _WIKILINK.sub(r"\1", stripped).replace("[[", "").replace("]]", "")
        text = text.lstrip("-* ").strip()
        if text:
            return text[:n] + ("…" if len(text) > n else "")
    return ""


def run_demo(service: MarvinService, *, top_k: int = 4) -> str:
    """Render an annotated transcript of the demo queries over the vault."""
    notes = service.vault.list_notes()
    counts: dict[str, int] = {}
    for note in notes:
        counts[note.metadata.kind.value] = counts.get(note.metadata.kind.value, 0) + 1

    lines: list[str] = [
        "=== Marvin on A Midsummer Night's Dream ===",
        f"Vault: {len(notes)} notes ("
        + ", ".join(f"{counts[k]} {k}" for k in sorted(counts))
        + ")",
        "",
    ]
    for question, _expected, _kind in DEMO_QUERIES:
        hits = service.search(question, limit=top_k)
        kinds_fired = ", ".join(sorted({h.kind.value for h in hits})) or "(none)"
        lines.append(f'Q: "{question}"')
        lines.append(f"   memory kinds fired: {kinds_fired}")
        for rank, h in enumerate(hits, 1):
            lines.append(f"   {rank}. [{h.kind.value:<10}] {h.title:<26}  {_snippet(service, h)}")
        entities: list[str] = []
        for h in hits:
            for entity in h.links:
                if entity not in entities:
                    entities.append(entity)
        if entities:
            lines.append(f"   linked entities (graph): {', '.join(entities[:10])}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="marvin.eval.demo",
        description="Print the Midsummer Night's Dream demo transcript.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("hash", "fastembed", "auto"),
        default="fastembed",
        help="Embedding backend (default: fastembed for real semantic matching).",
    )
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        service = build_demo_service(
            embedding_provider=args.embedding_provider, state_dir=Path(tmp)
        )
        try:
            print(run_demo(service, top_k=args.top_k))
        finally:
            service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .models import MemoryKind, NoteMetadata, NoteRecord, utc_now

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


def slugify_title(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "", title).strip()
    collapsed = re.sub(r"\s+", " ", cleaned)
    return collapsed or "untitled-memory"


def normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for tag in tags:
        value = tag.strip().lstrip("#")
        if not value:
            continue
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(value)
    return normalized


def normalize_links(links: list[str] | None) -> list[str]:
    if not links:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for link in links:
        value = link.strip()
        if not value:
            continue
        lowered = value.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(value)
    return normalized


def extract_wikilinks(text: str) -> list[str]:
    return normalize_links(WIKILINK_RE.findall(text))


class VaultStore:
    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.vault_path.mkdir(parents=True, exist_ok=True)
        for kind in MemoryKind:
            (self.vault_path / kind.folder_name).mkdir(parents=True, exist_ok=True)

    def note_path(self, kind: MemoryKind, title: str, unique: bool = False) -> Path:
        base = self.vault_path / kind.folder_name
        stem = slugify_title(title)
        candidate = base / f"{stem}.md"
        if not unique or not candidate.exists():
            return candidate
        digest = hashlib.sha1(f"{title}-{utc_now().isoformat()}".encode()).hexdigest()[:8]
        return base / f"{stem}-{digest}.md"

    def write_note(
        self,
        *,
        kind: MemoryKind,
        title: str,
        body: str,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        aliases: list[str] | None = None,
        source: dict[str, str] | None = None,
        existing_path: Path | None = None,
        unique: bool = False,
    ) -> tuple[Path, bool]:
        path = existing_path or self.note_path(kind, title, unique=unique)
        created = not path.exists()

        now = utc_now()
        metadata = None
        if path.exists():
            try:
                metadata = self.read_note(path).metadata
            except Exception:
                metadata = None

        metadata = metadata or NoteMetadata(kind=kind, title=title)
        metadata.kind = kind
        metadata.title = title
        metadata.updated_at = now
        if created:
            metadata.created_at = now
        metadata.tags = normalize_tags(tags or metadata.tags)
        metadata.links = normalize_links((links or []) + extract_wikilinks(body))
        metadata.aliases = normalize_links(aliases or metadata.aliases)
        metadata.source = source or metadata.source

        frontmatter = {
            "id": metadata.id,
            "kind": metadata.kind.value,
            "title": metadata.title,
            "created_at": metadata.created_at.astimezone(UTC).isoformat(),
            "updated_at": metadata.updated_at.astimezone(UTC).isoformat(),
            "tags": metadata.tags,
            "links": metadata.links,
            "aliases": metadata.aliases,
            "source": metadata.source,
        }

        rendered = self._render_note(
            title=title, frontmatter=frontmatter, body=body, links=metadata.links
        )
        path.write_text(rendered, encoding="utf-8")
        return path, created

    def read_note(self, path: Path) -> NoteRecord:
        raw_text = path.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(raw_text)

        title = str(frontmatter.get("title") or self._extract_heading(body) or path.stem)
        kind = self._parse_kind(frontmatter.get("kind") or path.parent.name)
        tags = normalize_tags(frontmatter.get("tags") or [])
        links = normalize_links((frontmatter.get("links") or []) + extract_wikilinks(body))
        aliases = normalize_links(frontmatter.get("aliases") or [])
        source = frontmatter.get("source") or {}

        metadata = NoteMetadata(
            id=str(
                frontmatter.get("id") or hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
            ),
            kind=kind,
            title=title,
            created_at=self._parse_datetime(frontmatter.get("created_at")),
            updated_at=self._parse_datetime(frontmatter.get("updated_at")),
            tags=tags,
            links=links,
            aliases=aliases,
            source=source,
        )
        return NoteRecord(path=path, metadata=metadata, body=body.strip(), raw_text=raw_text)

    def list_notes(self, kind: MemoryKind | None = None) -> list[NoteRecord]:
        roots = [kind] if kind is not None else list(MemoryKind)
        notes: list[NoteRecord] = []
        for memory_kind in roots:
            folder = self.vault_path / memory_kind.folder_name
            for path in sorted(folder.glob("*.md")):
                notes.append(self.read_note(path))
        return notes

    def find_note(self, *, title: str, kind: MemoryKind) -> NoteRecord | None:
        normalized = title.casefold().strip()
        folder = self.vault_path / kind.folder_name
        for path in folder.glob("*.md"):
            record = self.read_note(path)
            candidate_names = [
                record.metadata.title,
                path.stem,
                *record.metadata.aliases,
            ]
            if any(name.casefold().strip() == normalized for name in candidate_names):
                return record
        return None

    def get_note(self, identifier: str) -> NoteRecord | None:
        candidate = self.vault_path / identifier
        if candidate.exists() and candidate.is_file():
            return self.read_note(candidate)

        normalized = identifier.casefold().strip()
        for note in self.list_notes():
            if normalized in {
                note.metadata.title.casefold().strip(),
                note.path.stem.casefold().strip(),
                str(note.path.relative_to(self.vault_path)).casefold().strip(),
            }:
                return note
        return None

    def compose_related_block(self, links: list[str]) -> str:
        normalized = normalize_links(links)
        if not normalized:
            return ""
        bullets = "\n".join(f"- [[{link}]]" for link in normalized)
        return f"\n\n## Related\n{bullets}"

    def _render_note(
        self, *, title: str, frontmatter: dict[str, object], body: str, links: list[str]
    ) -> str:
        rendered_frontmatter = yaml.safe_dump(
            frontmatter, sort_keys=False, allow_unicode=False
        ).strip()
        trimmed_body = body.strip()
        if not trimmed_body.startswith("# "):
            trimmed_body = f"# {title}\n\n{trimmed_body}" if trimmed_body else f"# {title}"

        related_block = ""
        if links and "## Related" not in trimmed_body:
            related_block = self.compose_related_block(links)

        return f"---\n{rendered_frontmatter}\n---\n\n{trimmed_body}{related_block}\n"

    def _split_frontmatter(self, raw_text: str) -> tuple[dict[str, object], str]:
        match = FRONTMATTER_RE.match(raw_text)
        if not match:
            return {}, raw_text
        frontmatter = yaml.safe_load(match.group(1)) or {}
        return frontmatter, raw_text[match.end() :]

    def _extract_heading(self, body: str) -> str | None:
        match = HEADING_RE.search(body)
        return match.group(1).strip() if match else None

    def _parse_kind(self, value: str) -> MemoryKind:
        normalized = value.strip().lower()
        if normalized.endswith("s"):
            normalized = normalized[:-1]
        return MemoryKind(normalized)

    def _parse_datetime(self, value: object) -> datetime:
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return utc_now()
        return utc_now()

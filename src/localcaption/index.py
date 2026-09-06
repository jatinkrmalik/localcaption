"""JSONL search index of previously transcribed videos.

Default location: ``~/.local/share/localcaption/index.jsonl``.
Override with ``LOCALCAPTION_INDEX_PATH``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chapters import Segment, load_segments


def default_index_path() -> Path:
    env = os.environ.get("LOCALCAPTION_INDEX_PATH")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return root / "localcaption" / "index.jsonl"


def upsert_index(entry: dict[str, Any], path: Path | None = None) -> Path:
    """Insert or replace the row whose ``id`` matches *entry*."""
    path = path or default_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") != entry.get("id"):
                rows.append(obj)
    rows.append(entry)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_index(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or default_index_path()
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@dataclass(frozen=True)
class SearchHit:
    id: str
    url: str
    title: str
    start: float | None
    text: str
    score: int
    transcript: str


def search_index(term: str, path: Path | None = None) -> list[SearchHit]:
    """Substring-search every indexed transcript. Higher ``score`` is better."""
    needle = term.strip().lower()
    if not needle:
        return []
    hits: list[SearchHit] = []
    for entry in load_index(path):
        hits.extend(_hits_for_entry(entry, needle))
    hits.sort(key=lambda h: (-h.score, h.id, h.start if h.start is not None else 0.0))
    return hits


def _hits_for_entry(entry: dict[str, Any], needle: str) -> list[SearchHit]:
    transcript = str(entry.get("transcript") or "")
    vid = str(entry.get("id") or "")
    url = str(entry.get("url") or "")
    title = str(entry.get("title") or "")
    title_bonus = title.lower().count(needle)

    hits: list[SearchHit] = []
    for ch in entry.get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        ch_title = str(ch.get("title") or "")
        count = ch_title.lower().count(needle)
        if not count:
            continue
        start = ch.get("start_time")
        hits.append(
            SearchHit(
                id=vid,
                url=url,
                title=title,
                start=None if start is None else float(start),
                text=f"[chapter] {ch_title}",
                score=count + title_bonus + 2,
                transcript=transcript,
            )
        )

    for seg in _entry_segments(transcript):
        count = seg.text.lower().count(needle)
        if not count:
            continue
        hits.append(
            SearchHit(
                id=vid,
                url=url,
                title=title,
                start=seg.start if seg.start >= 0 else None,
                text=seg.text.strip(),
                score=count + title_bonus,
                transcript=transcript,
            )
        )

    if not hits and title_bonus:
        hits.append(
            SearchHit(
                id=vid,
                url=url,
                title=title,
                start=None,
                text=title,
                score=title_bonus,
                transcript=transcript,
            )
        )
    return hits


def _entry_segments(transcript: str) -> list[Segment]:
    if not transcript:
        return []
    path = Path(transcript)
    segs = load_segments(path)
    if segs:
        return segs
    if not path.is_file():
        return []
    # Untimed fallback: one synthetic segment per non-empty line.
    return [
        Segment(start=-1.0, end=-1.0, text=line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]

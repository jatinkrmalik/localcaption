"""YouTube chapter markers: persist metadata and split a transcript by time."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRT_TS = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)

# Longer suffixes first so ".chaptered.md" is not treated as ".md".
_KNOWN_SUFFIXES = (
    ".chapters.json",
    ".chaptered.md",
    ".txt",
    ".json",
    ".srt",
    ".vtt",
    ".md",
)


@dataclass(frozen=True)
class Chapter:
    title: str
    start_time: float
    end_time: float | None = None


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def chapters_from_info(info: dict[str, Any] | None) -> list[Chapter]:
    """Pull ``[{title, start_time, end_time}, ...]`` out of a yt-dlp info dict."""
    if not info:
        return []
    raw = info.get("chapters") or []
    out: list[Chapter] = []
    for i, ch in enumerate(raw):
        if not isinstance(ch, dict):
            continue
        start = ch.get("start_time")
        if start is None:
            continue
        try:
            start_f = float(start)
        except (TypeError, ValueError):
            continue
        title = str(ch.get("title") or "").strip() or f"Chapter {i + 1}"
        end = ch.get("end_time")
        try:
            end_f = None if end is None else float(end)
        except (TypeError, ValueError):
            end_f = None
        out.append(Chapter(title=title, start_time=start_f, end_time=end_f))
    filled: list[Chapter] = []
    for i, ch in enumerate(out):
        end = ch.end_time
        if end is None and i + 1 < len(out):
            end = out[i + 1].start_time
        filled.append(Chapter(title=ch.title, start_time=ch.start_time, end_time=end))
    return filled


def chapters_as_dicts(chapters: list[Chapter]) -> list[dict[str, Any]]:
    return [
        {"title": c.title, "start_time": c.start_time, "end_time": c.end_time}
        for c in chapters
    ]


def format_timestamp(seconds: float) -> str:
    """Render seconds as ``MM:SS`` or ``H:MM:SS``."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def write_chapters_json(path: Path, chapters: list[Chapter]) -> Path:
    path.write_text(
        json.dumps(chapters_as_dicts(chapters), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_segments(transcript_path: Path) -> list[Segment]:
    """Load timed segments from sibling whisper ``.json``, else ``.srt``."""
    base = _without_known_suffix(transcript_path)
    json_path = base.parent / f"{base.name}.json"
    srt_path = base.parent / f"{base.name}.srt"
    try:
        if json_path.is_file():
            segs = _segments_from_whisper_json(json_path)
            if segs:
                return segs
        if srt_path.is_file():
            return _segments_from_srt(srt_path)
    except OSError:
        return []
    return []


def build_chaptered_markdown(
    chapters: list[Chapter],
    segments: list[Segment],
    fallback_text: str = "",
) -> str:
    """Render ``## MM:SS Title`` headings with the transcript sliced underneath."""
    if not chapters:
        return ""

    buckets: list[list[str]] = [[] for _ in chapters]
    if segments:
        for seg in segments:
            buckets[_chapter_index(chapters, seg.start)].append(seg.text)
    elif fallback_text.strip():
        buckets[0].append(fallback_text.strip())

    lines: list[str] = []
    for ch, texts in zip(chapters, buckets, strict=True):
        lines.append(f"## {format_timestamp(ch.start_time)} {ch.title}")
        lines.append("")
        if texts:
            lines.append("\n".join(texts))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_chaptered_md(
    path: Path,
    chapters: list[Chapter],
    segments: list[Segment],
    fallback_text: str = "",
) -> Path:
    path.write_text(
        build_chaptered_markdown(chapters, segments, fallback_text),
        encoding="utf-8",
    )
    return path


def _chapter_index(chapters: list[Chapter], t: float) -> int:
    idx = 0
    for i, ch in enumerate(chapters):
        if ch.start_time <= t:
            idx = i
        else:
            break
    return idx


def _segments_from_whisper_json(path: Path) -> list[Segment]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("transcription") or data.get("segments") or []
    segs: list[Segment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            start, end = _segment_times(item)
        except (TypeError, ValueError):
            continue
        segs.append(Segment(start=start, end=end, text=text))
    return segs


def _segment_times(item: dict[str, Any]) -> tuple[float, float]:
    offsets = item.get("offsets")
    if isinstance(offsets, dict) and "from" in offsets:
        start = float(offsets["from"]) / 1000.0
        end = float(offsets.get("to", offsets["from"])) / 1000.0
        return start, end
    if "start" in item:
        start = float(item["start"])
        return start, float(item.get("end", start))
    ts = item.get("timestamps") if isinstance(item.get("timestamps"), dict) else {}
    return (
        _parse_clock(str(ts.get("from", "00:00:00,000"))),
        _parse_clock(str(ts.get("to", "00:00:00,000"))),
    )


def _parse_clock(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return 0.0


def _without_known_suffix(path: Path) -> Path:
    """Strip a transcript suffix, keeping extra dots in the stem."""
    name = path.name
    lower = name.lower()
    for suffix in _KNOWN_SUFFIXES:
        if lower.endswith(suffix):
            return path.parent / name[: -len(suffix)]
    return path


def _segments_from_srt(path: Path) -> list[Segment]:
    text = path.read_text(encoding="utf-8", errors="replace")
    segs: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if not lines:
            continue
        ts_line = lines[0] if "-->" in lines[0] else (lines[1] if len(lines) > 1 else "")
        m = _SRT_TS.search(ts_line)
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        body_start = 1 if "-->" in lines[0] else 2
        body = " ".join(lines[body_start:]).strip()
        if body:
            segs.append(Segment(start=start, end=end, text=body))
    return segs

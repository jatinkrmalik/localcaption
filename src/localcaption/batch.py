"""Batch transcription: many URLs (or local files) → one summary table.

Run sequentially. whisper.cpp already saturates the machine, so a pool
would only add contention.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import _logging as log
from .pipeline import transcribe_url
from .whisper import DEFAULT_MODEL

# Watch, embed, shorts, live, youtu.be. Used so we can name the output dir
# (and decide resume) without a yt-dlp metadata round-trip.
_YOUTUBE_ID_RE = re.compile(
    r"(?:(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?(?:[^#]*&)?v=|embed/|shorts/|live/|v/)"
    r"|youtu\.be/)([A-Za-z0-9_-]{11})"
)


@dataclass(frozen=True)
class BatchItem:
    """Outcome of one source in a batch run."""

    source: str
    video_id: str
    status: str  # "ok" | "failed" | "skipped"
    duration_s: float | None
    elapsed_s: float
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    """Aggregated result of :func:`transcribe_urls`."""

    items: list[BatchItem]
    wall_clock_s: float

    def exit_code(self) -> int:
        return 1 if any(i.status == "failed" for i in self.items) else 0

    def summary(self) -> str:
        id_w = max((len(i.video_id) for i in self.items), default=0)
        dur_w = max((len(_fmt_hms(i.duration_s)) for i in self.items), default=2)
        lines: list[str] = []
        for item in self.items:
            mark = {"ok": "✅", "failed": "❌", "skipped": "⏭"}[item.status]
            if item.status == "ok":
                tail = _fmt_elapsed(item.elapsed_s)
            elif item.status == "skipped":
                tail = "skipped"
            else:
                tail = item.error or "failed"
            lines.append(
                f"{mark} {item.video_id:<{id_w}}  {_fmt_hms(item.duration_s):>{dur_w}}  {tail}"
            )
        lines.append("--------------------------------")
        n_ok = sum(1 for i in self.items if i.status == "ok")
        n_fail = sum(1 for i in self.items if i.status == "failed")
        n_skip = sum(1 for i in self.items if i.status == "skipped")
        parts = [f"total: {len(self.items)}", f"ok: {n_ok}"]
        if n_skip:
            parts.append(f"skipped: {n_skip}")
        parts.append(f"failed: {n_fail}")
        parts.append(f"{_fmt_elapsed(self.wall_clock_s)} wall-clock")
        lines.append(" · ".join(parts))
        return "\n".join(lines)


def read_url_list(path: Path) -> list[str]:
    """Return sources from *path*: one per line, skipping blanks and ``#`` comments.

    Local paths expand ``~``. Relative local paths are resolved against the
    list file's directory (not the process cwd), so a queue sitting next to
    its media files still works when you pass an absolute ``--batch`` path.
    """
    list_path = Path(path)
    base = list_path.parent
    urls: list[str] = []
    for line in list_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(_normalize_local_source(stripped, base))
    return urls


def video_id_for(source: str) -> str:
    """Stable id used for ``<out>/<id>/<id>.txt`` (resume + isolated output).

    YouTube URLs use the 11-character video id. Local files use a sanitized
    path so two ``episode.mp3`` files in different folders do not share a
    directory. Other URLs use host + full path so two sites (or two shows
    on one CDN) cannot clobber each other.
    """
    if "://" not in source:
        return _safe_id(Path(source).expanduser().as_posix())
    match = _YOUTUBE_ID_RE.search(source)
    if match:
        return match.group(1)
    parsed = urlparse(source)
    host = (parsed.hostname or "url").removeprefix("www.")
    parts = [unquote(p) for p in parsed.path.split("/") if p]
    tail = "_".join(parts) if parts else "video"
    return _safe_id(f"{host}_{tail}")


def transcribe_urls(
    urls: list[str],
    *,
    out_dir: Path,
    whisper_dir: Path,
    model: str = DEFAULT_MODEL,
    language: str = "auto",
    keep_intermediate: bool = False,
) -> BatchResult:
    """Transcribe each source in *urls* sequentially.

    Per-URL output goes to ``out_dir/<videoId>/``. If
    ``<videoId>/<videoId>.txt`` already exists, that source is skipped.
    Failures are recorded and the rest of the list still runs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[BatchItem] = []
    wall0 = time.monotonic()
    for url in urls:
        items.append(
            _transcribe_one(
                url,
                out_dir=out_dir,
                whisper_dir=whisper_dir,
                model=model,
                language=language,
                keep_intermediate=keep_intermediate,
            )
        )
    return BatchResult(items=items, wall_clock_s=time.monotonic() - wall0)


def _transcribe_one(
    url: str,
    *,
    out_dir: Path,
    whisper_dir: Path,
    model: str,
    language: str,
    keep_intermediate: bool,
) -> BatchItem:
    if "://" not in url:
        url = str(Path(url).expanduser())
    video_id = video_id_for(url)
    item_dir = out_dir / video_id
    existing = item_dir / f"{video_id}.txt"
    if existing.is_file():
        log.info(f"skip {video_id}: already exists ({existing})")
        return BatchItem(
            source=url,
            video_id=video_id,
            status="skipped",
            duration_s=None,
            elapsed_s=0.0,
        )

    t0 = time.monotonic()
    try:
        result = transcribe_url(
            url,
            out_dir=item_dir,
            whisper_dir=whisper_dir,
            model=model,
            language=language,
            keep_intermediate=keep_intermediate,
            stem=video_id,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        err = _format_error(exc)
        log.error(f"{video_id}: {err}")
        return BatchItem(
            source=url,
            video_id=video_id,
            status="failed",
            duration_s=None,
            elapsed_s=elapsed,
            error=err,
        )
    return BatchItem(
        source=url,
        video_id=video_id,
        status="ok",
        duration_s=result.duration_s,
        elapsed_s=time.monotonic() - t0,
    )


def _normalize_local_source(source: str, base: Path) -> str:
    """Expand ``~``; make relative local paths relative to *base*."""
    if "://" in source:
        return source
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path)


def _safe_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if not cleaned:
        return "video"
    if len(cleaned) <= 120:
        return cleaned
    # Keep the tail so long macOS temp prefixes do not drop the filename.
    return cleaned[-120:].lstrip("._") or "video"


def _format_error(exc: BaseException) -> str:
    msg = str(exc).splitlines()[0].strip() if str(exc) else ""
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def _fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total}s"
    m, s = divmod(total, 60)
    if m < 60:
        return f"{m}m {s}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if m else f"{h}h"

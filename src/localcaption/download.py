"""Stage 1: download the best available audio stream with yt-dlp.

We use yt-dlp's Python API rather than the CLI so we can deterministically
discover the resulting filename via :py:meth:`YoutubeDL.prepare_filename`.
This avoids the fragility of parsing ``--print after_move:filepath`` output
across yt-dlp versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _logging as log
from .errors import DependencyError, DownloadError

# YouTube periodically blocks specific player clients. The mobile clients
# tend to be the most stable for audio-only retrieval. Order matters: yt-dlp
# tries them left-to-right.
DEFAULT_PLAYER_CLIENTS: tuple[str, ...] = ("android", "ios", "web")


@dataclass(frozen=True)
class DownloadResult:
    """Downloaded audio plus the yt-dlp info dict (chapters, title, ...)."""
    path: Path
    info: dict[str, Any]


def download_audio(
    url: str,
    work_dir: Path,
    *,
    player_clients: tuple[str, ...] = DEFAULT_PLAYER_CLIENTS,
) -> DownloadResult:
    """Download the best audio stream for *url* into *work_dir*.

    Returns the path plus the yt-dlp info dict. Raises :class:`DownloadError`
    if yt-dlp fails to produce a usable file.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise DependencyError(
            "Python package 'yt-dlp' is not installed. "
            "Install with: pip install -e .[dev]   (or)   pip install yt-dlp"
        ) from exc

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
        "restrictfilenames": True,
        "overwrites": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_args": {"youtube": {"player_client": list(player_clients)}},
    }

    log.info(f"yt-dlp: downloading bestaudio for {url}")
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Playlists nest one level deeper. We disabled them above, but be
            # defensive in case the URL is e.g. a single video inside a list.
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            audio_path = Path(ydl.prepare_filename(info))
    except Exception as exc:  # yt-dlp raises a zoo of exception types
        raise DownloadError(f"yt-dlp failed: {exc}") from exc

    # If a post-processor changed the extension, fall back to scanning by id.
    if not audio_path.is_file():
        matches = sorted(work_dir.glob(f"{info.get('id', '*')}.*"))
        if matches:
            audio_path = matches[0]

    if not audio_path.is_file():
        raise DownloadError(
            f"yt-dlp finished but no audio file was found near {audio_path}"
        )

    log.info(f"downloaded audio: {audio_path.name}")
    return DownloadResult(path=audio_path, info=info or {})

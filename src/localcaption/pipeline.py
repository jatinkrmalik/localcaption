"""High-level orchestration: URL → transcript artefacts.

This module is the public Python API. The CLI is a thin wrapper around
:func:`transcribe_url`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from . import _logging as log
from .audio import to_whisper_wav
from .download import download_audio
from .whisper import DEFAULT_MODEL, TranscriptionResult, transcribe


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated result of one URL → transcript run."""
    source_url: str
    audio_path: Path | None
    wav_path: Path | None
    transcripts: TranscriptionResult


def _is_local_file(source: str) -> bool:
    if "://" in source:
        return False
    return Path(source).is_file()


def transcribe_url(
    url: str,
    *,
    out_dir: Path,
    whisper_dir: Path,
    model: str = DEFAULT_MODEL,
    language: str = "auto",
    keep_intermediate: bool = False,
) -> PipelineResult:
    """Run the full pipeline on *url* and return the produced artefacts.

    *url* may be an actual URL or a path to a local video/audio file.

    Parameters
    ----------
    url:
        Any URL `yt-dlp` can resolve, or a local file path.
    out_dir:
        Directory for the final transcript files.
    whisper_dir:
        Path to the whisper.cpp checkout (built and with a ggml model present).
    model:
        whisper.cpp model name (e.g. ``base.en``, ``small.en``, ``large-v3``).
    language:
        ISO language code or ``"auto"`` to let whisper detect it.
    keep_intermediate:
        If True, leave the downloaded audio + 16 kHz WAV in ``out_dir/.work``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path: Path | None = None
    wav_path: Path | None = None
    try:
        if _is_local_file(url):
            audio_path = Path(url).resolve()
        else:
            audio_path = download_audio(url, work_dir)
        wav_path = work_dir / f"{audio_path.stem}.16k.wav"
        to_whisper_wav(audio_path, wav_path)

        out_base = out_dir / audio_path.stem
        transcripts = transcribe(
            wav_path, model, out_base, whisper_dir=whisper_dir, language=language
        )
    finally:
        if not keep_intermediate:
            shutil.rmtree(work_dir, ignore_errors=True)
            audio_path = None
            wav_path = None

    log.info("done")
    return PipelineResult(
        source_url=url,
        audio_path=audio_path,
        wav_path=wav_path,
        transcripts=transcripts,
    )

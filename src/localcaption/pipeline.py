"""High-level orchestration: URL → transcript artefacts.

This module is the public Python API. The CLI is a thin wrapper around
:func:`transcribe_url`.
"""

from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

from . import _logging as log
from .audio import to_whisper_wav
from .download import download_audio
from .summary import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from .summary import write_summary
from .whisper import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    Backend,
    TranscriptionResult,
    get_backend,
    transcribe,
)


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated result of one URL → transcript run."""
    source_url: str
    audio_path: Path | None
    wav_path: Path | None
    transcripts: TranscriptionResult
    duration_s: float | None = None
    summary: Path | None = None


def _is_local_file(source: str) -> bool:
    if "://" in source:
        return False
    return Path(source).is_file()


def transcribe_url(
    url: str,
    *,
    out_dir: Path,
    whisper_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    language: str = "auto",
    keep_intermediate: bool = False,
    stem: str | None = None,
    backend: str | Backend = DEFAULT_BACKEND,
    summary: bool = False,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    summary_prompt: Path | None = None,
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
        Required for the whisper-cpp backend; ignored by faster-whisper.
    model:
        Model name (e.g. ``base.en``, ``small.en``, ``large-v3``).
    language:
        ISO language code or ``"auto"`` to let whisper detect it.
    keep_intermediate:
        If True, leave the downloaded audio + 16 kHz WAV in ``out_dir/.work``.
    stem:
        Basename for transcript files. Defaults to the audio/file stem.
    backend:
        Backend name (``whisper-cpp``, ``faster-whisper``) or a :class:`Backend`.
    summary:
        If True, POST the ``.txt`` transcript to local Ollama and write
        ``<id>.summary.md``. Failures are warnings; they do not raise.
    summary_model:
        Ollama model name (default ``llama3.1:8b``).
    summary_prompt:
        Optional path to a prompt template. ``{transcript}`` is substituted
        if present; otherwise the transcript is appended.
    """
    # Validate named backends before yt-dlp/ffmpeg.
    if isinstance(backend, str):
        get_backend(backend, whisper_dir=whisper_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path: Path | None = None
    wav_path: Path | None = None
    duration_s: float | None = None
    try:
        audio_path = (
            Path(url).resolve() if _is_local_file(url) else download_audio(url, work_dir)
        )
        wav_path = work_dir / f"{audio_path.stem}.16k.wav"
        to_whisper_wav(audio_path, wav_path)
        duration_s = _wav_duration_s(wav_path)

        out_base = out_dir / (stem or audio_path.stem)
        transcripts = transcribe(
            wav_path,
            model,
            out_base,
            whisper_dir=whisper_dir,
            language=language,
            backend=backend,
        )
    finally:
        if not keep_intermediate:
            shutil.rmtree(work_dir, ignore_errors=True)
            audio_path = None
            wav_path = None

    summary_path: Path | None = None
    if summary:
        summary_path = write_summary(
            transcripts.txt,
            model=summary_model,
            prompt_path=summary_prompt,
        )

    log.info("done")
    return PipelineResult(
        source_url=url,
        audio_path=audio_path,
        wav_path=wav_path,
        transcripts=transcripts,
        duration_s=duration_s,
        summary=summary_path,
    )


def _wav_duration_s(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            if not rate:
                return None
            return wf.getnframes() / float(rate)
    except Exception:
        return None

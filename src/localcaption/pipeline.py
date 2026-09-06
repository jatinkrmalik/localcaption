"""High-level orchestration: URL → transcript artefacts.

This module is the public Python API. The CLI is a thin wrapper around
:func:`transcribe_url`.
"""

from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _logging as log
from .audio import to_whisper_wav
from .chapters import (
    Chapter,
    chapters_as_dicts,
    chapters_from_info,
    load_segments,
    write_chaptered_md,
    write_chapters_json,
)
from .download import download_audio
from .index import upsert_index
from .summary import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from .summary import write_summary
from .whisper import DEFAULT_MODEL, TranscriptionResult, transcribe


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated result of one URL → transcript run."""
    source_url: str
    audio_path: Path | None
    wav_path: Path | None
    transcripts: TranscriptionResult
    duration_s: float | None = None
    summary: Path | None = None
    chapters_json: Path | None = None
    chaptered_md: Path | None = None


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
    stem: str | None = None,
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
    model:
        whisper.cpp model name (e.g. ``base.en``, ``small.en``, ``large-v3``).
    language:
        ISO language code or ``"auto"`` to let whisper detect it.
    keep_intermediate:
        If True, leave the downloaded audio + 16 kHz WAV in ``out_dir/.work``.
    stem:
        Basename for transcript files. Defaults to the audio/file stem.
    summary:
        If True, POST the ``.txt`` transcript to local Ollama and write
        ``<id>.summary.md``. Failures are warnings; they do not raise.
    summary_model:
        Ollama model name (default ``llama3.1:8b``).
    summary_prompt:
        Optional path to a prompt template. ``{transcript}`` is substituted
        if present; otherwise the transcript is appended.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path: Path | None = None
    wav_path: Path | None = None
    duration_s: float | None = None
    chapters_json: Path | None = None
    chaptered_md: Path | None = None
    index_entry: dict[str, Any] | None = None
    try:
        if _is_local_file(url):
            audio_path = Path(url).resolve()
            info: dict[str, Any] = {
                "id": audio_path.stem,
                "title": audio_path.name,
                "webpage_url": str(audio_path),
            }
        else:
            downloaded = download_audio(url, work_dir)
            audio_path = downloaded.path
            info = downloaded.info or {}

        wav_path = work_dir / f"{audio_path.stem}.16k.wav"
        to_whisper_wav(audio_path, wav_path)
        duration_s = _wav_duration_s(wav_path)

        out_base = out_dir / (stem or audio_path.stem)
        transcripts = transcribe(
            wav_path, model, out_base, whisper_dir=whisper_dir, language=language
        )

        chapters = chapters_from_info(info)
        if chapters:
            chapters_json, chaptered_md = _write_chapter_artefacts(
                out_base, chapters, transcripts
            )

        index_entry = _index_entry(url, audio_path, info, chapters, transcripts)
    finally:
        if not keep_intermediate:
            shutil.rmtree(work_dir, ignore_errors=True)
            audio_path = None
            wav_path = None

    if index_entry is not None:
        try:
            upsert_index(index_entry)
        except OSError as exc:
            log.warn(f"could not update search index: {exc}")

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
        chapters_json=chapters_json,
        chaptered_md=chaptered_md,
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


def _write_chapter_artefacts(
    out_base: Path,
    chapters: list[Chapter],
    transcripts: TranscriptionResult,
) -> tuple[Path, Path]:
    chapters_json = write_chapters_json(out_base.with_suffix(".chapters.json"), chapters)
    fallback = ""
    if transcripts.txt.exists():
        fallback = transcripts.txt.read_text(encoding="utf-8", errors="replace")
    chaptered_md = write_chaptered_md(
        out_base.with_suffix(".chaptered.md"),
        chapters,
        load_segments(out_base),
        fallback_text=fallback,
    )
    return chapters_json, chaptered_md


def _index_entry(
    url: str,
    audio_path: Path,
    info: dict[str, Any],
    chapters: list[Chapter],
    transcripts: TranscriptionResult,
) -> dict[str, Any]:
    video_id = str(info.get("id") or audio_path.stem)
    title = info.get("title") or video_id
    webpage = info.get("webpage_url") or info.get("original_url") or url
    return {
        "id": video_id,
        "url": webpage,
        "title": title,
        "duration": info.get("duration"),
        "chapters": chapters_as_dicts(chapters),
        "transcript": str(transcripts.txt.resolve()),
    }

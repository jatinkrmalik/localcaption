"""faster-whisper transcription backend (optional extra)."""

from __future__ import annotations

import json
from pathlib import Path

from .. import _logging as log
from ..errors import DependencyError, TranscriptionError
from ..whisper import TranscriptionResult


def require_faster_whisper() -> type:
    """Return ``WhisperModel``, or raise if the optional extra is missing."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise DependencyError(
            "faster-whisper is not installed. "
            "Install it with: pip install 'localcaption[faster]'"
        ) from exc
    return WhisperModel


class FasterWhisperBackend:
    """Transcribe with SYSTRAN/faster-whisper (CTranslate2).

    Requires ``pip install localcaption[faster]``. Models are loaded by
    faster-whisper itself (Hugging Face / CTranslate2), not from a
    whisper.cpp ``models/`` directory.
    """

    def transcribe(
        self,
        wav: Path,
        model: str,
        out_basename: Path,
        *,
        language: str = "auto",
    ) -> TranscriptionResult:
        WhisperModel = require_faster_whisper()

        lang = None if language == "auto" else language
        log.info(f"faster-whisper: model={model} language={language}")
        try:
            fw_model = WhisperModel(model)
            segments, info = fw_model.transcribe(str(wav), language=lang)
            segments = list(segments)
        except Exception as exc:
            raise TranscriptionError(
                f"faster-whisper failed on {wav}: {exc}"
            ) from exc

        out_basename = Path(out_basename)
        out_basename.parent.mkdir(parents=True, exist_ok=True)

        full_text = "".join(getattr(s, "text", "") or "" for s in segments).strip()
        txt_path = out_basename.with_suffix(".txt")
        srt_path = out_basename.with_suffix(".srt")
        vtt_path = out_basename.with_suffix(".vtt")
        json_path = out_basename.with_suffix(".json")

        txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
        srt_path.write_text(_to_srt(segments), encoding="utf-8")
        vtt_path.write_text(_to_vtt(segments), encoding="utf-8")
        payload = {
            "language": getattr(info, "language", None),
            "text": full_text,
            "segments": [
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text,
                }
                for s in segments
            ],
        }
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

        return TranscriptionResult(txt=txt_path, srt=srt_path, vtt=vtt_path, json=json_path)


def _fmt_ts(seconds: float, *, vtt: bool = False) -> str:
    ms = int(round(max(seconds, 0.0) * 1000.0))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _to_srt(segments) -> str:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def _to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        lines.append(
            f"{_fmt_ts(seg.start, vtt=True)} --> {_fmt_ts(seg.end, vtt=True)}"
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

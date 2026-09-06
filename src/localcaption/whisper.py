"""Stage 3: transcribe a 16 kHz mono WAV via a pluggable backend.

The default backend is whisper.cpp. Alternative backends live in
``localcaption.backends`` and are selected with ``--backend`` /
``$LOCALCAPTION_BACKEND``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errors import DependencyError

DEFAULT_MODEL = "small.en"
BACKEND_WHISPER_CPP = "whisper-cpp"
BACKEND_FASTER_WHISPER = "faster-whisper"
DEFAULT_BACKEND = BACKEND_WHISPER_CPP
BACKEND_NAMES = (BACKEND_WHISPER_CPP, BACKEND_FASTER_WHISPER)
SUPPORTED_OUTPUT_FORMATS = ("txt", "srt", "vtt", "json")


@dataclass(frozen=True)
class WhisperPaths:
    """Locations of an installed whisper.cpp checkout."""
    root: Path

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def model_file(self, model_name: str) -> Path:
        return self.models_dir / f"ggml-{model_name}.bin"

    def find_binary(self) -> Path:
        candidates = [
            self.root / "build" / "bin" / "whisper-cli",
            self.root / "build" / "bin" / "main",
            self.root / "main",
        ]
        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                return c
        raise DependencyError(
            f"whisper.cpp binary not found under {self.root}. "
            "Run scripts/setup.sh to build it."
        )


@dataclass(frozen=True)
class TranscriptionResult:
    """Paths to the artefacts emitted by a transcription backend."""
    txt: Path
    srt: Path
    vtt: Path
    json: Path

    def existing(self) -> dict[str, Path]:
        """Return only the outputs that actually exist on disk."""
        return {k: v for k, v in vars(self).items() if isinstance(v, Path) and v.exists()}


class Backend(Protocol):
    """A transcription backend turns a WAV file into transcript artefacts."""

    def transcribe(
        self,
        wav: Path,
        model: str,
        out_basename: Path,
        *,
        language: str = "auto",
    ) -> TranscriptionResult: ...


def output_file(out_basename: Path, suffix: str) -> Path:
    """Append ``suffix`` without pathlib stripping extra dots in the stem.

    ``Path.with_suffix('.txt')`` turns ``lecture.2024`` into ``lecture.txt``.
    whisper.cpp's ``-of`` keeps the full basename, so we must too.
    """
    if not suffix.startswith("."):
        suffix = "." + suffix
    return out_basename.parent / f"{out_basename.name}{suffix}"


def resolve_backend_name(cli_value: str | None = None) -> str:
    """Return the backend name from an explicit value, env, or the default.

    Precedence: *cli_value* (if given) > ``$LOCALCAPTION_BACKEND`` > whisper-cpp.
    """
    if cli_value:
        name = cli_value.strip()
    else:
        name = os.environ.get("LOCALCAPTION_BACKEND", "").strip() or DEFAULT_BACKEND
    if name not in BACKEND_NAMES:
        raise DependencyError(
            f"Unknown transcription backend {name!r}. "
            f"Choose from: {', '.join(BACKEND_NAMES)}"
        )
    return name


def get_backend(name: str, *, whisper_dir: Path | None = None) -> Backend:
    """Instantiate a backend by name."""
    if name == BACKEND_WHISPER_CPP:
        if whisper_dir is None:
            raise DependencyError(
                "whisper-cpp backend requires a whisper.cpp directory "
                "(pass --whisper-dir or set LOCALCAPTION_WHISPER_DIR)"
            )
        from .backends.whisper_cpp import WhisperCppBackend

        return WhisperCppBackend(whisper_dir)
    if name == BACKEND_FASTER_WHISPER:
        from .backends.faster_whisper import FasterWhisperBackend, require_faster_whisper

        require_faster_whisper()
        return FasterWhisperBackend()
    raise DependencyError(
        f"Unknown transcription backend {name!r}. "
        f"Choose from: {', '.join(BACKEND_NAMES)}"
    )


def transcribe(
    wav: Path,
    model: str,
    out_basename: Path,
    *,
    whisper_dir: Path | None = None,
    language: str = "auto",
    backend: str | Backend = DEFAULT_BACKEND,
) -> TranscriptionResult:
    """Transcribe *wav* and emit artefacts at *out_basename*.{txt,srt,vtt,json}.

    *backend* is a backend name (``whisper-cpp``, ``faster-whisper``) or an
    object matching :class:`Backend`. ``whisper_dir`` is required for the
    whisper.cpp backend and ignored otherwise.
    """
    impl: Backend = (
        backend
        if not isinstance(backend, str)
        else get_backend(backend, whisper_dir=whisper_dir)
    )
    return impl.transcribe(wav, model, out_basename, language=language)

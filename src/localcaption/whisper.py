"""Stage 3: run whisper.cpp against a 16 kHz mono WAV.

The whisper.cpp checkout is treated as an external dependency installed by
``scripts/setup.sh``. We auto-detect the binary because the project moved
its executable around historically (``main`` → ``build/bin/main`` →
``build/bin/whisper-cli``).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import _logging as log
from .errors import DependencyError, TranscriptionError

DEFAULT_MODEL = "small.en"
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
    """Paths to the artefacts emitted by whisper.cpp."""
    txt: Path
    srt: Path
    vtt: Path
    json: Path

    def existing(self) -> dict[str, Path]:
        """Return only the outputs that actually exist on disk."""
        return {k: v for k, v in vars(self).items() if isinstance(v, Path) and v.exists()}


def transcribe(
    wav: Path,
    model: str,
    out_basename: Path,
    *,
    whisper_dir: Path,
    language: str = "auto",
) -> TranscriptionResult:
    """Run whisper.cpp on *wav* and emit transcripts at *out_basename*.{txt,srt,vtt,json}."""
    paths = WhisperPaths(whisper_dir)
    binary = paths.find_binary()
    model_path = paths.model_file(model)
    if not model_path.is_file():
        raise DependencyError(
            f"Model file missing: {model_path}\n"
            f"Download it with: bash {paths.models_dir}/download-ggml-model.sh {model}"
        )

    out_basename.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "-m", str(model_path),
        "-f", str(wav),
        "-of", str(out_basename),
        "-otxt", "-osrt", "-ovtt", "-oj",
        "-l", language,
    ]
    log.info(f"whisper.cpp: model={model} language={language}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise TranscriptionError(
            f"whisper.cpp failed (exit {exc.returncode}) on {wav}"
        ) from exc

    return TranscriptionResult(
        txt=out_basename.with_suffix(".txt"),
        srt=out_basename.with_suffix(".srt"),
        vtt=out_basename.with_suffix(".vtt"),
        json=out_basename.with_suffix(".json"),
    )

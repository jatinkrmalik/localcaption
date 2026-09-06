"""whisper.cpp transcription backend."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .. import _logging as log
from ..errors import DependencyError, TranscriptionError
from ..whisper import TranscriptionResult, WhisperPaths, output_file


class WhisperCppBackend:
    """Invoke a local whisper.cpp checkout (``whisper-cli``)."""

    def __init__(self, whisper_dir: Path) -> None:
        self.whisper_dir = Path(whisper_dir)

    def transcribe(
        self,
        wav: Path,
        model: str,
        out_basename: Path,
        *,
        language: str = "auto",
    ) -> TranscriptionResult:
        paths = WhisperPaths(self.whisper_dir)
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
            txt=output_file(out_basename, ".txt"),
            srt=output_file(out_basename, ".srt"),
            vtt=output_file(out_basename, ".vtt"),
            json=output_file(out_basename, ".json"),
        )

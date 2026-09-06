"""Fake-based tests for pluggable transcription backends.

No real whisper.cpp binary or faster-whisper model is loaded.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from localcaption.errors import DependencyError, TranscriptionError
from localcaption.whisper import (
    BACKEND_FASTER_WHISPER,
    BACKEND_WHISPER_CPP,
    DEFAULT_BACKEND,
    TranscriptionResult,
    get_backend,
    resolve_backend_name,
    transcribe,
)


def _touch_executable(path: Path, source: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


def _fake_whisper_checkout(root: Path, *, binary_src: str | None = None) -> Path:
    if binary_src is None:
        binary_src = """\
#!/bin/sh
OF=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-of" ]; then
    shift
    OF=$1
  fi
  shift
done
mkdir -p "$(dirname "$OF")"
for ext in txt srt vtt json; do
  printf ok > "$OF.$ext"
done
"""
    _touch_executable(root / "build" / "bin" / "whisper-cli", binary_src)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "ggml-base.en.bin").write_bytes(b"fake-model")
    return root


class _Seg:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


# --- resolve_backend_name / get_backend ---------------------------------


def test_resolve_backend_default(monkeypatch) -> None:
    monkeypatch.delenv("LOCALCAPTION_BACKEND", raising=False)
    assert resolve_backend_name(None) == DEFAULT_BACKEND == BACKEND_WHISPER_CPP


def test_resolve_backend_env(monkeypatch) -> None:
    monkeypatch.setenv("LOCALCAPTION_BACKEND", "faster-whisper")
    assert resolve_backend_name(None) == BACKEND_FASTER_WHISPER


def test_resolve_backend_flag_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("LOCALCAPTION_BACKEND", "faster-whisper")
    assert resolve_backend_name("whisper-cpp") == BACKEND_WHISPER_CPP


def test_resolve_backend_strips_whitespace(monkeypatch) -> None:
    monkeypatch.setenv("LOCALCAPTION_BACKEND", "  faster-whisper  ")
    assert resolve_backend_name(None) == BACKEND_FASTER_WHISPER


def test_resolve_backend_unknown_raises(monkeypatch) -> None:
    monkeypatch.delenv("LOCALCAPTION_BACKEND", raising=False)
    with pytest.raises(DependencyError, match="Unknown transcription backend"):
        resolve_backend_name("mlx")


def test_get_backend_whisper_cpp_requires_dir() -> None:
    with pytest.raises(DependencyError, match="requires a whisper.cpp directory"):
        get_backend(BACKEND_WHISPER_CPP, whisper_dir=None)


def test_get_backend_unknown() -> None:
    with pytest.raises(DependencyError, match="Unknown transcription backend"):
        get_backend("mlx", whisper_dir=Path("/tmp"))


def test_get_backend_returns_implementations(tmp_path: Path, dummy_faster_whisper) -> None:
    from localcaption.backends.faster_whisper import FasterWhisperBackend
    from localcaption.backends.whisper_cpp import WhisperCppBackend

    cpp = get_backend(BACKEND_WHISPER_CPP, whisper_dir=tmp_path)
    fw = get_backend(BACKEND_FASTER_WHISPER)
    assert isinstance(cpp, WhisperCppBackend)
    assert isinstance(fw, FasterWhisperBackend)


def test_get_backend_faster_whisper_missing_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(DependencyError, match="localcaption\\[faster\\]"):
        get_backend(BACKEND_FASTER_WHISPER)


# --- whisper.cpp ---------------------------------------------------------


def test_whisper_cpp_writes_outputs(tmp_path: Path) -> None:
    from localcaption.backends.whisper_cpp import WhisperCppBackend

    root = _fake_whisper_checkout(tmp_path / "whisper.cpp")
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    out = tmp_path / "out" / "talk.final"

    result = WhisperCppBackend(root).transcribe(wav, "base.en", out, language="en")

    assert result.txt == tmp_path / "out" / "talk.final.txt"
    assert result.txt.read_text() == "ok"
    assert result.srt.exists()
    assert result.vtt.exists()
    assert result.json.exists()
    assert result.existing().keys() == {"txt", "srt", "vtt", "json"}


def test_whisper_cpp_via_transcribe_dispatcher(tmp_path: Path) -> None:
    root = _fake_whisper_checkout(tmp_path / "whisper.cpp")
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    out = tmp_path / "talk"
    result = transcribe(wav, "base.en", out, whisper_dir=root, language="auto")
    assert result.txt.exists()


def test_whisper_cpp_missing_model(tmp_path: Path) -> None:
    from localcaption.backends.whisper_cpp import WhisperCppBackend

    root = tmp_path / "whisper.cpp"
    _touch_executable(root / "build" / "bin" / "whisper-cli")
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    with pytest.raises(DependencyError, match="Model file missing"):
        WhisperCppBackend(root).transcribe(wav, "base.en", tmp_path / "out")


def test_whisper_cpp_nonzero_exit(tmp_path: Path) -> None:
    from localcaption.backends.whisper_cpp import WhisperCppBackend

    root = _fake_whisper_checkout(
        tmp_path / "whisper.cpp",
        binary_src="#!/bin/sh\nexit 3\n",
    )
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    with pytest.raises(TranscriptionError, match="whisper.cpp failed \\(exit 3\\)"):
        WhisperCppBackend(root).transcribe(wav, "base.en", tmp_path / "out")


def test_transcribe_accepts_backend_instance(tmp_path: Path) -> None:
    called: list[tuple] = []

    class FakeBackend:
        def transcribe(self, wav, model, out_basename, *, language="auto"):
            called.append((wav, model, out_basename, language))
            return TranscriptionResult(
                txt=out_basename.with_suffix(".txt"),
                srt=out_basename.with_suffix(".srt"),
                vtt=out_basename.with_suffix(".vtt"),
                json=out_basename.with_suffix(".json"),
            )

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    out = tmp_path / "talk"
    result = transcribe(
        wav, "base.en", out, whisper_dir=tmp_path, language="en", backend=FakeBackend()
    )
    assert called == [(wav, "base.en", out, "en")]
    assert result.txt == out.with_suffix(".txt")


# --- faster-whisper ------------------------------------------------------


def _install_fake_faster_whisper(monkeypatch, *, segments=None, fail: bool = False):
    if segments is None:
        segments = [_Seg(0.0, 1.5, " Hello")]
    captured: dict = {}

    class FakeModel:
        def __init__(self, name, *a, **k):
            if fail:
                raise RuntimeError("boom")
            captured["model"] = name

        def transcribe(self, audio, language=None, **k):
            captured["audio"] = audio
            captured["language"] = language
            info = types.SimpleNamespace(language="en")
            return segments, info

    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)
    return captured


def test_faster_whisper_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    from localcaption.backends.faster_whisper import FasterWhisperBackend

    segs = [_Seg(0.0, 1.5, " Hello"), _Seg(3723.456, 3724.0, "world")]
    captured = _install_fake_faster_whisper(monkeypatch, segments=segs)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    out = tmp_path / "out" / "talk.final"

    result = FasterWhisperBackend().transcribe(wav, "base.en", out, language="auto")
    assert result.txt == tmp_path / "out" / "talk.final.txt"

    assert captured["model"] == "base.en"
    assert captured["audio"] == str(wav)
    assert captured["language"] is None
    assert result.txt.read_text(encoding="utf-8") == "Helloworld\n"
    srt = result.srt.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "01:02:03,456 --> 01:02:04,000" in srt
    vtt = result.vtt.read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    payload = json.loads(result.json.read_text(encoding="utf-8"))
    assert payload["language"] == "en"
    assert payload["text"] == "Helloworld"
    assert len(payload["segments"]) == 2


def test_transcribe_faster_whisper_without_whisper_dir(monkeypatch, tmp_path: Path) -> None:
    _install_fake_faster_whisper(monkeypatch)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    result = transcribe(wav, "base.en", tmp_path / "talk", backend=BACKEND_FASTER_WHISPER)
    assert result.txt.exists()


def test_faster_whisper_passes_language(monkeypatch, tmp_path: Path) -> None:
    from localcaption.backends.faster_whisper import FasterWhisperBackend

    captured = _install_fake_faster_whisper(monkeypatch)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    FasterWhisperBackend().transcribe(wav, "small.en", tmp_path / "talk", language="en")
    assert captured["language"] == "en"
    assert captured["model"] == "small.en"


def test_faster_whisper_missing_extra(monkeypatch, tmp_path: Path) -> None:
    from localcaption.backends.faster_whisper import FasterWhisperBackend

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    with pytest.raises(DependencyError, match="localcaption\\[faster\\]"):
        FasterWhisperBackend().transcribe(wav, "base.en", tmp_path / "talk")


def test_faster_whisper_runtime_error(monkeypatch, tmp_path: Path) -> None:
    from localcaption.backends.faster_whisper import FasterWhisperBackend

    _install_fake_faster_whisper(monkeypatch, fail=True)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    with pytest.raises(TranscriptionError, match="faster-whisper failed"):
        FasterWhisperBackend().transcribe(wav, "base.en", tmp_path / "talk")

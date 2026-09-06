"""Tests for the WhisperPaths binary discovery — no network or models required."""

from __future__ import annotations

from pathlib import Path

import pytest

from localcaption.errors import DependencyError
from localcaption.whisper import DEFAULT_MODEL, WhisperPaths


def _touch_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def test_find_binary_prefers_whisper_cli(tmp_path: Path) -> None:
    paths = WhisperPaths(tmp_path)
    _touch_executable(tmp_path / "build" / "bin" / "whisper-cli")
    _touch_executable(tmp_path / "build" / "bin" / "main")
    assert paths.find_binary().name == "whisper-cli"


def test_find_binary_falls_back_to_main(tmp_path: Path) -> None:
    paths = WhisperPaths(tmp_path)
    _touch_executable(tmp_path / "build" / "bin" / "main")
    assert paths.find_binary().name == "main"


def test_find_binary_legacy_top_level(tmp_path: Path) -> None:
    paths = WhisperPaths(tmp_path)
    _touch_executable(tmp_path / "main")
    assert paths.find_binary().name == "main"


def test_find_binary_missing_raises(tmp_path: Path) -> None:
    paths = WhisperPaths(tmp_path)
    with pytest.raises(DependencyError):
        paths.find_binary()


def test_default_model_is_small_en() -> None:
    assert DEFAULT_MODEL == "small.en"


def test_setup_and_install_scripts_default_to_small_en() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("scripts/setup.sh", "scripts/install.sh"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "${WHISPER_MODEL:-small.en}" in text, rel


def test_model_file_path(tmp_path: Path) -> None:
    paths = WhisperPaths(tmp_path)
    assert paths.model_file("base.en") == tmp_path / "models" / "ggml-base.en.bin"

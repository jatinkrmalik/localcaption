"""Shared test fixtures."""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep backend selection tests independent of the host environment."""
    monkeypatch.delenv("LOCALCAPTION_BACKEND", raising=False)


@pytest.fixture
def dummy_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend ``faster-whisper`` is importable without loading a real model."""
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = type("WhisperModel", (), {})
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)

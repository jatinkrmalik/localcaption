"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep backend selection tests independent of the host environment."""
    monkeypatch.delenv("LOCALCAPTION_BACKEND", raising=False)

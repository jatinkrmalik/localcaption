"""Tests for ``localcaption.models`` — registry, listing, removal.

Notes:
- Network downloads are NOT exercised here (would be flaky and slow).
  The download path is covered by an integration-style test that uses
  a fake HTTP server in tests/test_models_download.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localcaption import models
from localcaption.errors import LocalCaptionError

# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


def test_registry_contains_well_known_models():
    names = {m.name for m in models.known_models()}
    # Sanity: any of these missing means we accidentally broke the registry.
    for required in ("tiny.en", "base.en", "small.en", "large-v3"):
        assert required in names, f"missing well-known model: {required}"


def test_registry_is_sorted_smallest_first():
    sizes = [m.approx_size_mb for m in models.known_models()]
    assert sizes == sorted(sizes), "known_models() must be sorted by size ascending"


def test_get_model_returns_spec():
    spec = models.get_model("base.en")
    assert spec.name == "base.en"
    assert spec.is_english_only
    assert spec.url.endswith("ggml-base.en.bin")
    assert "huggingface.co" in spec.url


def test_get_model_raises_on_unknown():
    with pytest.raises(LocalCaptionError) as exc_info:
        models.get_model("xxx-not-a-model")
    msg = str(exc_info.value)
    # Must give the user actionable info, not just "no".
    assert "Unknown model" in msg
    assert "model list" in msg


def test_english_only_flag():
    assert models.get_model("base.en").is_english_only
    assert not models.get_model("base").is_english_only


# ──────────────────────────────────────────────────────────────────────
# Disk introspection
# ──────────────────────────────────────────────────────────────────────


def _make_fake_whisper_dir(root: Path, installed: list[str]) -> Path:
    """Build a minimal whisper.cpp-shaped tree with the given fake models."""
    whisper = root / "whisper.cpp"
    (whisper / "models").mkdir(parents=True)
    for name in installed:
        # 1 byte per file is enough; tests don't care about real ggml format.
        (whisper / "models" / f"ggml-{name}.bin").write_bytes(b"x")
    return whisper


def test_installed_model_files_empty_when_dir_missing(tmp_path):
    # Should NOT raise — listing just returns an empty mapping.
    assert models.installed_model_files(tmp_path / "no-such-dir") == {}


def test_installed_model_files_finds_existing(tmp_path):
    whisper = _make_fake_whisper_dir(tmp_path, ["base.en", "small.en"])
    found = models.installed_model_files(whisper)
    assert set(found.keys()) == {"base.en", "small.en"}
    for path in found.values():
        assert path.is_file()


def test_list_status_marks_installed_correctly(tmp_path):
    whisper = _make_fake_whisper_dir(tmp_path, ["base.en"])
    rows = models.list_status(whisper)
    by_name = {r.spec.name: r for r in rows}
    assert by_name["base.en"].is_installed is True
    assert by_name["small.en"].is_installed is False


def test_orphaned_models_are_detected(tmp_path):
    whisper = _make_fake_whisper_dir(tmp_path, ["base.en", "custom-finetune"])
    orphans = models.orphaned_installed_models(whisper)
    assert orphans == ["custom-finetune"]


# ──────────────────────────────────────────────────────────────────────
# Removal
# ──────────────────────────────────────────────────────────────────────


def test_remove_model_deletes_file(tmp_path):
    whisper = _make_fake_whisper_dir(tmp_path, ["base.en"])
    target = whisper / "models" / "ggml-base.en.bin"
    assert target.is_file()

    removed = models.remove_model("base.en", whisper)
    assert removed == target
    assert not target.exists()


def test_remove_model_raises_when_missing(tmp_path):
    whisper = _make_fake_whisper_dir(tmp_path, [])  # nothing installed
    with pytest.raises(LocalCaptionError) as exc_info:
        models.remove_model("base.en", whisper)
    assert "not installed" in str(exc_info.value)


# ──────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────


def test_model_path_uses_whisper_dir_layout(tmp_path):
    p = models.model_path(tmp_path, "small.en")
    assert p == tmp_path / "models" / "ggml-small.en.bin"

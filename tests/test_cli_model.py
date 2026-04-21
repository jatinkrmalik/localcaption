"""Tests for the `localcaption model {list,download,rm,info}` CLI plumbing.

These tests exercise the dispatcher and exit codes — the actual download
machinery is tested in test_models_download.py.
"""

from __future__ import annotations

from pathlib import Path

from localcaption import cli, models


def _run(argv: list[str]) -> int:
    """Run the CLI main() with no global side effects, return exit code."""
    return cli.main(argv)


# ──────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────


def test_bare_model_prints_usage(capsys):
    rc = _run(["model"])
    out = capsys.readouterr().out
    assert "subcommands" in out
    assert "list" in out and "download" in out and "rm" in out
    assert rc == 2  # bare invocation is an error per CLI conventions


def test_model_help_exits_zero(capsys):
    rc = _run(["model", "--help"])
    out = capsys.readouterr().out
    assert "subcommands" in out
    assert rc == 0


def test_unknown_subcommand_is_rejected(capsys):
    rc = _run(["model", "frobnicate"])
    err = capsys.readouterr().err
    assert "unknown subcommand" in err
    assert rc == 2


# ──────────────────────────────────────────────────────────────────────
# `model list`
# ──────────────────────────────────────────────────────────────────────


def _make_whisper(tmp_path: Path, installed: list[str]) -> Path:
    whisper = tmp_path / "whisper.cpp"
    (whisper / "models").mkdir(parents=True)
    for name in installed:
        (whisper / "models" / f"ggml-{name}.bin").write_bytes(b"x")
    return whisper


def test_list_includes_all_known_models(tmp_path, capsys):
    whisper = _make_whisper(tmp_path, [])
    rc = _run(["model", "list", "--whisper-dir", str(whisper)])
    out = capsys.readouterr().out

    assert rc == 0
    for spec in models.known_models():
        assert spec.name in out
    assert "not installed" in out


def test_list_marks_installed_models(tmp_path, capsys):
    whisper = _make_whisper(tmp_path, ["base.en"])
    rc = _run(["model", "list", "--whisper-dir", str(whisper)])
    out = capsys.readouterr().out

    assert rc == 0
    # The installed marker should appear at least once
    assert "installed" in out
    # base.en line must be on the same line as the installed marker
    base_line = next(line for line in out.splitlines() if line.lstrip().startswith("base.en "))
    assert "installed" in base_line


def test_list_works_when_whisper_dir_missing(tmp_path, capsys):
    """`model list` should never crash just because whisper.cpp isn't installed.

    This is critical: a brand-new user runs `model list` to figure out what
    to install BEFORE installing whisper.cpp.
    """
    nonexistent = tmp_path / "no-whisper-here"
    rc = _run(["model", "list", "--whisper-dir", str(nonexistent)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "base.en" in out
    assert "not installed" in out


def test_list_surfaces_orphan_models(tmp_path, capsys):
    whisper = _make_whisper(tmp_path, ["base.en", "my-finetune"])
    _run(["model", "list", "--whisper-dir", str(whisper)])
    out = capsys.readouterr().out

    assert "Other models found on disk" in out
    assert "my-finetune" in out


# ──────────────────────────────────────────────────────────────────────
# `model info`
# ──────────────────────────────────────────────────────────────────────


def test_info_shows_metadata_for_known_model(tmp_path, capsys):
    whisper = _make_whisper(tmp_path, [])
    rc = _run(["model", "info", "small.en", "--whisper-dir", str(whisper)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "small.en" in out
    assert "English-only" in out
    assert "huggingface.co" in out
    assert "Installed:    no" in out


def test_info_shows_installed_status_when_present(tmp_path, capsys):
    whisper = _make_whisper(tmp_path, ["small.en"])
    rc = _run(["model", "info", "small.en", "--whisper-dir", str(whisper)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Installed:    yes" in out


def test_info_unknown_model_returns_error(tmp_path, capsys):
    rc = _run(["model", "info", "totally-fake"])
    err = capsys.readouterr().err
    assert "Unknown model" in err
    assert rc == 2


# ──────────────────────────────────────────────────────────────────────
# `model rm`
# ──────────────────────────────────────────────────────────────────────


def test_rm_with_yes_flag_removes(tmp_path, capsys):
    whisper = _make_whisper(tmp_path, ["base.en"])
    target = whisper / "models" / "ggml-base.en.bin"
    assert target.is_file()

    rc = _run(["model", "rm", "base.en", "--whisper-dir", str(whisper), "-y"])
    out = capsys.readouterr().out

    assert rc == 0
    assert not target.exists()
    assert "Removed" in out


def test_rm_missing_model_returns_error(tmp_path, capsys):
    whisper = _make_whisper(tmp_path, [])
    rc = _run(["model", "rm", "base.en", "--whisper-dir", str(whisper), "-y"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "not installed" in err


def test_rm_aliases(tmp_path):
    """Both `rm`, `remove`, and `delete` should work — small affordance."""
    for alias in ("rm", "remove", "delete"):
        whisper = _make_whisper(tmp_path / alias, ["base.en"])
        rc = _run(["model", alias, "base.en", "--whisper-dir", str(whisper), "-y"])
        assert rc == 0, f"alias {alias!r} failed"

"""Tests for the `localcaption doctor` and dispatcher behaviour.

We don't want these to depend on a real whisper.cpp build, so we point the
diagnostic at a fixture directory and assert on the output text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localcaption.cli import _candidate_whisper_dirs, main


def test_top_level_help_no_args(capsys) -> None:
    rc = main([])
    assert rc == 2  # bare invocation should be a non-zero exit
    out = capsys.readouterr().out
    assert "doctor" in out
    assert "transcribe" in out


def test_help_alias_zero_exit(capsys) -> None:
    rc = main(["help"])
    assert rc == 0
    assert "doctor" in capsys.readouterr().out


def test_doctor_runs_without_install(capsys, tmp_path: Path) -> None:
    """`localcaption doctor` must produce a report even if nothing is installed."""
    rc = main(["doctor", "--whisper-dir", str(tmp_path / "missing")])
    out = capsys.readouterr().out
    assert "localcaption" in out
    assert "System tools" in out
    assert "whisper.cpp" in out
    # Missing whisper.cpp directory → non-zero exit
    assert rc == 1


def test_doctor_recognises_built_install(capsys, tmp_path: Path) -> None:
    """A whisper.cpp directory with a binary + model should pass."""
    whisper_dir = tmp_path / "whisper.cpp"
    bin_path = whisper_dir / "build" / "bin" / "whisper-cli"
    bin_path.parent.mkdir(parents=True)
    bin_path.write_text("#!/bin/sh\nexit 0\n")
    bin_path.chmod(0o755)

    models_dir = whisper_dir / "models"
    models_dir.mkdir()
    (models_dir / "ggml-base.en.bin").write_bytes(b"\x00" * 16)

    rc = main(["doctor", "--whisper-dir", str(whisper_dir)])
    out = capsys.readouterr().out
    assert "binary built" in out
    assert "ggml-base.en.bin" in out
    # ffmpeg / yt-dlp may or may not be present in the test env; the doctor
    # exits non-zero only if anything fails. Just assert the whisper section
    # was happy by checking the binary line shows ✅.
    assert "✅" in out
    # Don't assert on `rc` — depends on host env.
    del rc


def test_candidate_dirs_respects_env(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom-whisper"
    monkeypatch.setenv("LOCALCAPTION_WHISPER_DIR", str(custom))
    candidates = _candidate_whisper_dirs()
    assert candidates[0] == custom
    # Also includes the dev-checkout and XDG paths
    assert any("whisper.cpp" in str(c) for c in candidates[1:])


def test_unknown_subcommand_treated_as_url(monkeypatch) -> None:
    """Anything that isn't a known subcommand should fall through to transcribe.

    We don't want to actually transcribe in a unit test, so we patch
    transcribe_url to raise a known exception and assert it was reached.
    """
    sentinel: dict[str, str] = {}

    def fake(url, **kw):
        sentinel["url"] = url
        raise SystemExit(0)

    monkeypatch.setattr("localcaption.cli.transcribe_url", fake)
    with pytest.raises(SystemExit):
        main(["https://example.com/video"])
    assert sentinel["url"] == "https://example.com/video"


# --- doctor --fix --------------------------------------------------------


def test_doctor_does_not_invoke_installer_without_fix(monkeypatch, tmp_path: Path) -> None:
    """The diagnostic command must remain read-only by default.

    Even when there are obvious gaps, plain ``doctor`` should never call into
    the installer. This is the load-bearing safety guarantee.
    """
    def explode(*_a, **_kw):
        raise AssertionError("installer must not be invoked without --fix")

    monkeypatch.setattr("localcaption.installer.install_system_dep", explode)
    monkeypatch.setattr("localcaption.installer.ensure_whisper_cpp", explode)
    monkeypatch.setattr("localcaption.models.download_model", explode)

    rc = main(["doctor", "--whisper-dir", str(tmp_path / "missing")])
    assert rc == 1  # gaps detected, but no installs attempted


def test_doctor_fix_invokes_installer_for_missing_whisper(
    monkeypatch, tmp_path: Path
) -> None:
    """`doctor --fix` should call ensure_whisper_cpp + download_model."""
    whisper_dir = tmp_path / "whisper.cpp"
    calls: dict[str, object] = {}

    def fake_ensure(path):
        calls["whisper_dir"] = path
        # Simulate a successful build so the post-fix re-check passes.
        bin_path = path / "build" / "bin" / "whisper-cli"
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        bin_path.write_text("#!/bin/sh\nexit 0\n")
        bin_path.chmod(0o755)
        (path / "models").mkdir(exist_ok=True)
        return bin_path

    def fake_download(name, whisper_root, **_kw):
        calls["model_name"] = name
        calls["model_root"] = whisper_root
        target = whisper_root / "models" / f"ggml-{name}.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00" * 32)
        return target

    def fake_install_dep(name):
        calls.setdefault("deps_installed", []).append(name)  # type: ignore[union-attr]

    monkeypatch.setattr("localcaption.installer.ensure_whisper_cpp", fake_ensure)
    monkeypatch.setattr("localcaption.installer.install_system_dep", fake_install_dep)
    monkeypatch.setattr("localcaption.models.download_model", fake_download)

    rc = main(["doctor", "--fix", "--whisper-dir", str(whisper_dir),
               "--model", "tiny.en"])

    assert calls["whisper_dir"] == whisper_dir
    assert calls["model_name"] == "tiny.en"
    assert calls["model_root"] == whisper_dir
    # rc depends on host env (yt-dlp/ffmpeg may still be missing in CI), so
    # we only assert that the fix logic was driven, not the final exit code.
    del rc


def test_doctor_fix_aborts_when_install_step_fails(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A failing install step should print the error and exit non-zero."""
    from localcaption.errors import InstallError

    def fake_ensure(_path):
        raise InstallError("simulated build failure")

    monkeypatch.setattr("localcaption.installer.ensure_whisper_cpp", fake_ensure)
    monkeypatch.setattr("localcaption.installer.install_system_dep",
                        lambda _name: None)
    # Should never get to model download if whisper.cpp setup fails.
    monkeypatch.setattr(
        "localcaption.models.download_model",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("download must not be reached")
        ),
    )

    rc = main(["doctor", "--fix", "--whisper-dir", str(tmp_path / "missing")])
    out = capsys.readouterr().out
    assert "simulated build failure" in out
    assert "Fix aborted" in out
    assert rc == 1


def test_doctor_help_advertises_fix_flag(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["doctor", "--help"])
    out = capsys.readouterr().out
    assert "--fix" in out

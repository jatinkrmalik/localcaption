"""Tests for `localcaption.installer`.

We can't actually `git clone` or `cmake --build` in unit tests, so every
external command is mocked at the `subprocess.run` boundary. The goal is
to exercise the *orchestration* logic — argument shape, ordering, skip
conditions, error translation — not the underlying tools.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from localcaption import installer
from localcaption.errors import InstallError

# --- Detection ------------------------------------------------------------


def test_detect_platform_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    assert installer.detect_platform() == "macos"


def test_detect_platform_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    assert installer.detect_platform() == "linux"


def test_detect_platform_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.platform, "system", lambda: "Windows")
    assert installer.detect_platform() == "unsupported"


def test_detect_package_manager_prefers_brew(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.shutil, "which",
                        lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    assert installer.detect_package_manager() == "brew"


def test_detect_package_manager_falls_back_to_apt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.shutil, "which",
                        lambda name: "/usr/bin/apt-get" if name == "apt-get" else None)
    assert installer.detect_package_manager() == "apt"


def test_detect_package_manager_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)
    assert installer.detect_package_manager() is None


# --- _run wrapper ---------------------------------------------------------


def test_run_raises_install_error_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kw: Any) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    with pytest.raises(InstallError, match="exit 1"):
        installer._run(["false"], label="boom")


def test_run_raises_install_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kw: Any) -> None:
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    with pytest.raises(InstallError, match="command not found"):
        installer._run(["nope"], label="missing")


def test_run_passes_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kw: Any) -> None:
        seen["cmd"] = cmd
        seen["cwd"] = kw.get("cwd")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    installer._run(["echo", "hi"], label="echo", cwd=tmp_path)
    assert seen == {"cmd": ["echo", "hi"], "cwd": tmp_path}


# --- install_system_dep ---------------------------------------------------


def test_install_system_dep_skip_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/usr/bin/" + name)
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "_run", lambda cmd, **_kw: calls.append(cmd))
    installer.install_system_dep("ffmpeg")
    assert calls == []  # nothing to do


def test_install_system_dep_unknown_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)
    with pytest.raises(InstallError, match="Don't know how to install"):
        installer.install_system_dep("nonsense-tool")


def test_install_system_dep_no_package_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    with pytest.raises(InstallError, match="No supported package manager"):
        installer.install_system_dep("ffmpeg")


def test_install_system_dep_uses_brew_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    # ffmpeg missing, brew present.
    def which(name: str) -> str | None:
        return "/opt/homebrew/bin/brew" if name == "brew" else None

    monkeypatch.setattr(installer.shutil, "which", which)
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "_run", lambda cmd, **_kw: calls.append(cmd))
    installer.install_system_dep("ffmpeg")
    assert calls == [["brew", "install", "ffmpeg"]]


def test_install_system_dep_uses_apt_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    def which(name: str) -> str | None:
        return "/usr/bin/apt-get" if name == "apt-get" else None

    monkeypatch.setattr(installer.shutil, "which", which)
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "_run", lambda cmd, **_kw: calls.append(cmd))
    installer.install_system_dep("cmake")
    assert calls == [
        ["sudo", "apt-get", "update", "-y"],
        ["sudo", "apt-get", "install", "-y", "cmake"],
    ]


# --- ensure_whisper_cpp ---------------------------------------------------


def _make_built_checkout(root: Path) -> Path:
    """Lay out a fake whisper.cpp tree with an executable binary."""
    binary = root / "build" / "bin" / "whisper-cli"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    (root / "models").mkdir()
    return binary


def test_ensure_whisper_cpp_short_circuits_when_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    whisper_dir = tmp_path / "whisper.cpp"
    whisper_dir.mkdir()
    expected = _make_built_checkout(whisper_dir)

    # No subprocess calls should happen.
    def boom(*_a: Any, **_kw: Any) -> None:
        raise AssertionError("subprocess.run should not be invoked")

    monkeypatch.setattr(installer.subprocess, "run", boom)
    binary = installer.ensure_whisper_cpp(whisper_dir)
    assert binary == expected


def test_ensure_whisper_cpp_clones_and_builds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    whisper_dir = tmp_path / "whisper.cpp"  # does not yet exist
    calls: list[tuple[list[str], Path | None]] = []

    def fake_run(cmd: list[str], *, label: str, cwd: Path | None = None) -> None:
        calls.append((cmd, cwd))
        # Simulate the side-effects of `git clone` and `cmake --build`.
        if cmd[:2] == ["git", "clone"]:
            whisper_dir.mkdir(parents=True)
            (whisper_dir / "CMakeLists.txt").write_text("project(x)\n")
        elif cmd[:2] == ["cmake", "--build"]:
            _make_built_checkout(whisper_dir)

    monkeypatch.setattr(installer, "_run", fake_run)
    monkeypatch.setattr(installer.shutil, "which",
                        lambda name: f"/usr/bin/{name}")  # git + cmake present

    binary = installer.ensure_whisper_cpp(whisper_dir)

    # Confirm sequencing: clone → cmake configure → cmake build.
    assert [c[0][0] for c in calls] == ["git", "cmake", "cmake"]
    assert calls[0][0][:2] == ["git", "clone"]
    assert calls[1][0][:2] == ["cmake", "-B"]
    assert calls[2][0][:2] == ["cmake", "--build"]
    # cmake calls must run inside the checkout.
    assert calls[1][1] == whisper_dir
    assert calls[2][1] == whisper_dir
    assert binary.is_file()


def test_ensure_whisper_cpp_requires_git_when_cloning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    whisper_dir = tmp_path / "whisper.cpp"  # missing
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)
    with pytest.raises(InstallError, match="git is required"):
        installer.ensure_whisper_cpp(whisper_dir)


def test_ensure_whisper_cpp_requires_cmake_when_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    whisper_dir = tmp_path / "whisper.cpp"
    whisper_dir.mkdir()
    (whisper_dir / "CMakeLists.txt").write_text("project(x)\n")
    # No binary, no cmake → should fail with a hint.
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)
    with pytest.raises(InstallError, match="cmake is required"):
        installer.ensure_whisper_cpp(whisper_dir)


def test_ensure_whisper_cpp_rejects_non_directory(
    tmp_path: Path,
) -> None:
    not_a_dir = tmp_path / "whisper.cpp"
    not_a_dir.write_text("oops, this is a file")
    with pytest.raises(InstallError, match="not a directory"):
        installer.ensure_whisper_cpp(not_a_dir)

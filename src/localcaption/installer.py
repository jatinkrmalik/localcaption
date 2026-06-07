"""Self-healing installer for localcaption's external dependencies.

This module exists so ``localcaption doctor --fix`` can put a broken or
missing install back together without shelling out to ``install.sh``. The
shell script remains the bootstrap for first-time users (it's the thing
``curl | bash`` invokes), but the *logic* lives here so we have one place
to test it from and one place to evolve it.

Three concerns, three small entry points:

* :func:`install_system_dep` — install ffmpeg/cmake via brew/apt.
* :func:`ensure_whisper_cpp` — clone + build whisper.cpp at a chosen path.
* :func:`detect_platform` / :func:`detect_package_manager` — used by both
  callers and tests to decide what's even possible on this host.

All failures raise :class:`~localcaption.errors.InstallError` with an
actionable, copy-pasteable message — never a bare ``CalledProcessError``.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from . import _logging as log
from .errors import InstallError
from .whisper import WhisperPaths

Platform = Literal["macos", "linux", "unsupported"]
PackageManager = Literal["brew", "apt"]

# System dependencies we know how to install. Map: dep name → package name
# under each package manager. ``None`` means "package name matches dep name".
_SYSTEM_DEP_PACKAGES: dict[str, dict[PackageManager, str]] = {
    "ffmpeg": {"brew": "ffmpeg", "apt": "ffmpeg"},
    "cmake": {"brew": "cmake", "apt": "cmake"},
    "git": {"brew": "git", "apt": "git"},
}

WHISPER_CPP_REPO = "https://github.com/ggerganov/whisper.cpp"


# --- Detection helpers ----------------------------------------------------


def detect_platform() -> Platform:
    """Return ``"macos"``, ``"linux"``, or ``"unsupported"``."""
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Linux":
        return "linux"
    return "unsupported"


def detect_package_manager() -> PackageManager | None:
    """Return the first package manager we can drive on this host, if any.

    macOS prefers Homebrew, Linux prefers apt. Everything else returns
    ``None`` and the caller is expected to print a manual-install hint.
    """
    if shutil.which("brew"):
        return "brew"
    if shutil.which("apt-get"):
        return "apt"
    return None


# --- Subprocess wrapper ---------------------------------------------------


def _run(cmd: list[str], *, label: str, cwd: Path | None = None) -> None:
    """Run *cmd*, streaming output to the user's terminal.

    On non-zero exit raises :class:`InstallError` with an actionable message
    that includes the failed command and the working directory. We
    deliberately do **not** capture stdout/stderr — for long-running steps
    like ``cmake --build`` the live progress is the UX.
    """
    pretty = " ".join(cmd)
    log.info(f"{label}: {pretty}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise InstallError(
            f"{label} failed: command not found ({cmd[0]}). "
            f"Install it first, then re-run."
        ) from exc
    except subprocess.CalledProcessError as exc:
        loc = f" (in {cwd})" if cwd else ""
        raise InstallError(
            f"{label} failed (exit {exc.returncode}){loc}:\n    {pretty}"
        ) from exc


# --- System deps ----------------------------------------------------------


def install_system_dep(name: str) -> None:
    """Install a system tool (``ffmpeg``, ``cmake``, ``git``) via the host's
    package manager. No-op if the tool is already on ``PATH``.

    Raises :class:`InstallError` if the dependency is unknown to us, the
    platform is unsupported, or no package manager is available.
    """
    if shutil.which(name):
        log.info(f"{name}: already installed")
        return

    if name not in _SYSTEM_DEP_PACKAGES:
        raise InstallError(
            f"Don't know how to install '{name}' automatically. "
            f"Please install it manually and re-run."
        )

    pm = detect_package_manager()
    if pm is None:
        plat = detect_platform()
        hint = (
            "brew install ffmpeg cmake git" if plat == "macos"
            else "sudo apt-get install -y ffmpeg cmake git"
        )
        raise InstallError(
            f"No supported package manager found (need brew or apt-get) "
            f"to install '{name}'.\n"
            f"Install it manually:\n    {hint}"
        )

    pkg = _SYSTEM_DEP_PACKAGES[name][pm]
    if pm == "brew":
        _run(["brew", "install", pkg], label=f"install {name}")
    else:  # apt
        _run(["sudo", "apt-get", "update", "-y"], label="apt update")
        _run(["sudo", "apt-get", "install", "-y", pkg], label=f"install {name}")


# --- whisper.cpp ----------------------------------------------------------


def ensure_whisper_cpp(whisper_dir: Path) -> Path:
    """Make sure a built whisper.cpp checkout lives at *whisper_dir*.

    Steps (each is idempotent — skipped if already done):

    1. ``git clone --depth 1`` the repo if the directory is missing.
    2. ``cmake -B build && cmake --build build`` if the binary isn't found.

    Returns the path to the executable. Raises :class:`InstallError` on
    any failure with a copy-pasteable message.
    """
    whisper_dir = whisper_dir.expanduser()
    paths = WhisperPaths(whisper_dir)

    # 1. Clone if missing.
    if not whisper_dir.exists():
        if not shutil.which("git"):
            raise InstallError(
                "git is required to clone whisper.cpp but isn't installed.\n"
                "    macOS:  brew install git\n"
                "    Linux:  sudo apt-get install -y git"
            )
        whisper_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["git", "clone", "--depth", "1", WHISPER_CPP_REPO, str(whisper_dir)],
            label="clone whisper.cpp",
        )
    elif not whisper_dir.is_dir():
        raise InstallError(
            f"{whisper_dir} exists but is not a directory. "
            f"Move it aside and re-run."
        )
    else:
        log.info(f"whisper.cpp: already cloned at {whisper_dir}")

    # 2. Build if no binary present.
    try:
        binary = paths.find_binary()
        log.info(f"whisper.cpp: already built at {binary}")
        return binary
    except LookupError:
        pass  # not raised by find_binary — guard kept for future-proofing
    except Exception:
        # find_binary raises DependencyError when no binary is found; we
        # treat any "not found" as "needs building" and let the build step
        # surface a real error if something deeper is wrong.
        pass

    if not shutil.which("cmake"):
        raise InstallError(
            "cmake is required to build whisper.cpp but isn't installed.\n"
            "    macOS:  brew install cmake\n"
            "    Linux:  sudo apt-get install -y cmake"
        )

    cmakelists = whisper_dir / "CMakeLists.txt"
    if cmakelists.is_file():
        _run(
            ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
            label="cmake configure",
            cwd=whisper_dir,
        )
        _run(
            ["cmake", "--build", "build", "-j", "--config", "Release"],
            label="cmake build",
            cwd=whisper_dir,
        )
    else:
        # Older whisper.cpp checkouts only ship a Makefile.
        if not shutil.which("make"):
            raise InstallError(
                "Neither CMakeLists.txt nor make found for whisper.cpp build."
            )
        _run(["make", "-j"], label="make whisper.cpp", cwd=whisper_dir)

    # Re-resolve to confirm a binary now exists.
    return paths.find_binary()

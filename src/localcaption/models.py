"""Whisper.cpp model registry, download, listing, and removal.

This module is the single source of truth for which whisper.cpp models
``localcaption`` knows about. It deliberately mirrors the upstream
``download-ggml-model.sh`` from ggerganov/whisper.cpp so any name that
works with the bash script also works here.

The download is implemented with the standard library only (urllib +
hashlib + a small progress callback) so we don't add third-party
dependencies for what is essentially `curl + sha256sum`.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .errors import DependencyError, LocalCaptionError
from .whisper import WhisperPaths

# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

# Sources are kept in lock-step with whisper.cpp's download-ggml-model.sh:
#   https://github.com/ggerganov/whisper.cpp/blob/master/models/download-ggml-model.sh
HF_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata about a single whisper.cpp model."""

    name: str
    approx_size_mb: int   # rounded, for human display only
    description: str

    @property
    def filename(self) -> str:
        return f"ggml-{self.name}.bin"

    @property
    def url(self) -> str:
        return f"{HF_BASE_URL}/{self.filename}"

    @property
    def is_english_only(self) -> bool:
        # whisper.cpp uses the '.en' suffix on English-only checkpoints.
        return ".en" in self.name


# Curated subset that 99% of users will actually want. The full upstream
# script lists ~30 quantised variants; we expose the high-signal ones and
# let advanced users add others via `model add` (out-of-scope for v0.2.0).
_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("tiny.en",         75,    "smallest English-only — fastest, lowest quality"),
    ModelSpec("tiny",            75,    "smallest multilingual — fastest, lowest quality"),
    ModelSpec("base.en",        142,    "small English-only — current install default"),
    ModelSpec("base",           142,    "small multilingual"),
    ModelSpec("small.en",       466,    "good general-purpose English (recommended)"),
    ModelSpec("small",          466,    "good general-purpose multilingual"),
    ModelSpec("medium.en",     1500,    "high-accuracy English — slower"),
    ModelSpec("medium",        1500,    "high-accuracy multilingual — slower"),
    ModelSpec("large-v3",      3100,    "best accuracy — large multilingual model"),
    ModelSpec("large-v3-turbo", 1620,   "near-large accuracy at ~half the size"),
)


def known_models() -> tuple[ModelSpec, ...]:
    """All models the registry knows about, ordered smallest-to-largest."""
    return tuple(sorted(_REGISTRY, key=lambda m: (m.approx_size_mb, m.name)))


def get_model(name: str) -> ModelSpec:
    """Look up a model by name. Raises ``LocalCaptionError`` if unknown."""
    for spec in _REGISTRY:
        if spec.name == name:
            return spec
    valid = ", ".join(m.name for m in known_models())
    raise LocalCaptionError(
        f"Unknown model: {name!r}\n"
        f"Run `localcaption model list` to see all supported models.\n"
        f"Valid names: {valid}"
    )


# ──────────────────────────────────────────────────────────────────────
# Disk introspection
# ──────────────────────────────────────────────────────────────────────


def installed_model_files(whisper_dir: Path) -> dict[str, Path]:
    """Map of installed model name → on-disk path under *whisper_dir*."""
    models_dir = WhisperPaths(whisper_dir).models_dir
    if not models_dir.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(models_dir.glob("ggml-*.bin")):
        # ggml-base.en.bin -> base.en
        name = path.stem.removeprefix("ggml-")
        found[name] = path
    return found


def model_path(whisper_dir: Path, name: str) -> Path:
    """Where a given model would live on disk (whether installed or not)."""
    return WhisperPaths(whisper_dir).model_file(name)


# ──────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────


def _human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 86_400:
        return "?"
    if seconds < 60:
        return f"{int(seconds):d}s"
    if seconds < 3600:
        return f"{int(seconds // 60):d}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600):d}h{int((seconds % 3600) // 60):02d}m"


class _ProgressBar:
    """Tiny self-contained progress bar so we don't add tqdm.

    Renders nothing if stderr is not a TTY (so logs stay clean in CI).
    """

    BAR_WIDTH = 36

    def __init__(self, total: int, label: str):
        self.total = total
        self.label = label
        self.start = time.monotonic()
        self.last_render = 0.0
        self.tty = sys.stderr.isatty()

    def update(self, downloaded: int) -> None:
        now = time.monotonic()
        # throttle: redraw at most every 0.1 s, plus always on completion
        if not self.tty:
            return
        if downloaded < self.total and (now - self.last_render) < 0.1:
            return
        self.last_render = now
        elapsed = max(now - self.start, 0.001)
        speed = downloaded / elapsed
        if self.total > 0:
            ratio = min(downloaded / self.total, 1.0)
            filled = int(self.BAR_WIDTH * ratio)
            bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)
            remaining = (self.total - downloaded) / speed if speed > 0 else 0
            line = (
                f"\r  {self.label} [{bar}] "
                f"{_human_bytes(downloaded)}/{_human_bytes(self.total)} · "
                f"{_human_bytes(speed)}/s · ETA {_format_eta(remaining)}"
            )
        else:
            line = f"\r  {self.label} {_human_bytes(downloaded)} @ {_human_bytes(speed)}/s"
        # pad with spaces to clear any leftover characters from a previous wider line
        sys.stderr.write(line + "   ")
        sys.stderr.flush()

    def finish(self) -> None:
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _open_with_redirects(url: str, timeout: float = 30.0):
    """urlopen with a friendly user-agent (HuggingFace is more reliable with one)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "localcaption-model-downloader/0.2"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def download_model(
    name: str,
    whisper_dir: Path,
    *,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download model *name* into *whisper_dir*'s ``models/`` directory.

    Args:
        name:        Model name (e.g. ``"small.en"``); must be in the registry.
        whisper_dir: Root of the whisper.cpp checkout.
        force:       Re-download even if the file already exists.
        on_progress: Optional ``(downloaded_bytes, total_bytes) -> None`` callback,
                     useful for tests; defaults to a TTY progress bar.

    Returns:
        The absolute path to the downloaded model file.
    """
    spec = get_model(name)  # raises if unknown — fail fast
    target = model_path(whisper_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and not force:
        return target

    # Atomic-ish download: write to .part then rename on success.
    # Avoids leaving a half-downloaded file at the canonical path on Ctrl-C.
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"  Source: {spec.url}")
    print(f"  Target: {target}")

    bar = _ProgressBar(0, f"{spec.name:14s}")

    try:
        with _open_with_redirects(spec.url) as response:
            total = int(response.headers.get("Content-Length", "0"))
            bar.total = total
            sha = hashlib.sha256()
            downloaded = 0
            chunk = 1024 * 256  # 256 KB chunks → smooth UI without too much sys-call overhead

            with tmp.open("wb") as fh:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    sha.update(block)
                    downloaded += len(block)
                    if on_progress is not None:
                        on_progress(downloaded, total)
                    else:
                        bar.update(downloaded)
        bar.finish()
    except urllib.error.HTTPError as exc:
        # Clean up partial file
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise DependencyError(
            f"Download failed for model {name!r}: HTTP {exc.code} {exc.reason}\n"
            f"  URL: {spec.url}\n"
            f"  Try again, or check https://huggingface.co/ggerganov/whisper.cpp"
        ) from exc
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise DependencyError(
            f"Download failed for model {name!r}: {exc.reason}\n"
            f"  URL: {spec.url}\n"
            "  Check your network connection and try again."
        ) from exc
    except KeyboardInterrupt:
        bar.finish()
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    # Sanity-check size (whisper.cpp doesn't publish per-file SHA-256 in a
    # machine-readable way, so this is the strongest check we can make).
    if total > 0 and downloaded != total:
        tmp.unlink(missing_ok=True)
        raise DependencyError(
            f"Truncated download for {name!r}: got {downloaded} bytes, "
            f"expected {total}. Try again."
        )

    # Atomic move into place
    os.replace(tmp, target)
    return target


# ──────────────────────────────────────────────────────────────────────
# Removal
# ──────────────────────────────────────────────────────────────────────


def remove_model(name: str, whisper_dir: Path) -> Path:
    """Delete a model file from disk.

    Returns the path that was removed. Raises if the file isn't present.
    """
    target = model_path(whisper_dir, name)
    if not target.is_file():
        raise DependencyError(
            f"Model {name!r} is not installed at {target}\n"
            f"Run `localcaption model list` to see installed models."
        )
    target.unlink()
    return target


# ──────────────────────────────────────────────────────────────────────
# Listing helpers (consumed by the CLI for nice formatting)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelStatus:
    """One row in the ``model list`` table."""

    spec: ModelSpec
    installed_at: Path | None   # None = not installed

    @property
    def is_installed(self) -> bool:
        return self.installed_at is not None

    @property
    def actual_size_mb(self) -> int | None:
        """Real on-disk size in MB, if installed."""
        if self.installed_at is None:
            return None
        return max(1, self.installed_at.stat().st_size // (1024 * 1024))


def list_status(whisper_dir: Path) -> list[ModelStatus]:
    """Build the full status table: every known model + install state."""
    installed = installed_model_files(whisper_dir)
    return [
        ModelStatus(spec=spec, installed_at=installed.get(spec.name))
        for spec in known_models()
    ]


def orphaned_installed_models(whisper_dir: Path) -> list[str]:
    """Models present on disk but not in our registry (e.g. user added a custom one)."""
    known = {m.name for m in known_models()}
    return sorted(name for name in installed_model_files(whisper_dir) if name not in known)

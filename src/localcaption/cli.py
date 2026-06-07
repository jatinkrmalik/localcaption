"""Command-line entry point.

Exposed as the ``localcaption`` console script via ``pyproject.toml``.

Two invocation styles are supported:

    localcaption <url> [options]      # one-shot transcription (default)
    localcaption doctor               # diagnose your install
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from . import _logging as log
from .errors import LocalCaptionError
from .pipeline import transcribe_url
from .whisper import DEFAULT_MODEL, WhisperPaths

# Subcommands recognised by the dispatcher. Anything else is treated as a URL
# and routed to the implicit "transcribe" command for backwards compatibility.
SUBCOMMANDS = frozenset({"doctor", "transcribe"})


# --- whisper.cpp directory resolution ------------------------------------

def _xdg_data_home() -> Path:
    """Return $XDG_DATA_HOME or its conventional fallback (~/.local/share)."""
    env = os.environ.get("XDG_DATA_HOME")
    return Path(env).expanduser() if env else Path.home() / ".local" / "share"


def _candidate_whisper_dirs() -> list[Path]:
    """Where to look for the whisper.cpp checkout, in priority order.

    1. ``$LOCALCAPTION_WHISPER_DIR`` if set (explicit override).
    2. ``./whisper.cpp`` if running from a dev checkout.
    3. ``$XDG_DATA_HOME/localcaption/whisper.cpp`` (where ``install.sh`` puts it).
    """
    candidates: list[Path] = []
    env = os.environ.get("LOCALCAPTION_WHISPER_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd() / "whisper.cpp")
    candidates.append(_xdg_data_home() / "localcaption" / "whisper.cpp")
    return candidates


def _default_whisper_dir() -> Path:
    """Pick the first existing whisper.cpp directory, or the last candidate.

    The "last candidate" fallback ensures error messages point users at the
    canonical install location rather than the dev-only ``./whisper.cpp``.
    """
    candidates = _candidate_whisper_dirs()
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[-1]


# --- transcribe (default) subcommand -------------------------------------

def _build_transcribe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localcaption",
        description="Fully-local YouTube → transcript using yt-dlp + ffmpeg + whisper.cpp.",
    )
    parser.add_argument("url", help="YouTube URL (or any URL yt-dlp supports)")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"whisper model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path.cwd() / "transcripts",
        help="output directory for transcript files (default: ./transcripts)",
    )
    parser.add_argument(
        "-l", "--language", default="auto",
        help="ISO language code, or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--whisper-dir", type=Path, default=None,
        help="path to a built whisper.cpp checkout "
             "(default: $LOCALCAPTION_WHISPER_DIR, ./whisper.cpp, "
             "or ~/.local/share/localcaption/whisper.cpp)",
    )
    parser.add_argument(
        "--keep-audio", action="store_true",
        help="keep the downloaded audio and intermediate WAV under <out>/.work/",
    )
    parser.add_argument(
        "--no-print", action="store_true",
        help="do not echo the transcript to stdout when finished",
    )
    parser.add_argument(
        "--auto-download", action="store_true",
        help="if the requested model isn't installed, download it without asking",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _ensure_model_available(model: str, whisper_dir: Path, auto: bool) -> bool:
    """If the requested model isn't installed, prompt and download it.

    Returns True if the model is now available, False otherwise.
    Errors are logged but never raised — caller decides what to do.
    """
    from . import models  # local import → keeps `localcaption --help` cheap

    target = models.model_path(whisper_dir, model)
    if target.is_file():
        return True

    # Unknown model? Don't even try to download — fail loud.
    try:
        spec = models.get_model(model)
    except LocalCaptionError as exc:
        log.error(str(exc))
        return False

    print(
        f"\nModel '{spec.name}' is not installed "
        f"(~{_format_size_mb(spec.approx_size_mb)})."
    )

    if auto:
        proceed = True
    elif not sys.stdin.isatty():
        # Non-interactive (CI, piped stdin) without --auto-download → refuse.
        log.error(
            f"Cannot prompt to download {spec.name!r} (stdin is not a TTY).\n"
            f"  Pass --auto-download, or run: localcaption model download {spec.name}"
        )
        return False
    else:
        try:
            reply = input("  Download it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return False
        proceed = reply in {"", "y", "yes"}

    if not proceed:
        print(
            f"  Skipped. Run: localcaption model download {spec.name}\n"
            "  Or use a different model with --model <name>."
        )
        return False

    try:
        models.download_model(spec.name, whisper_dir)
    except LocalCaptionError as exc:
        log.error(str(exc))
        return False
    except KeyboardInterrupt:
        log.error("interrupted; partial download cleaned up.")
        return False
    return True


def _cmd_transcribe(argv: list[str]) -> int:
    args = _build_transcribe_parser().parse_args(argv)
    whisper_dir = args.whisper_dir or _default_whisper_dir()

    # Pre-flight: ensure the requested model is on disk before doing the (slow)
    # download+ffmpeg dance. Cheap if already installed, helpful if not.
    if whisper_dir.is_dir() and not _ensure_model_available(
        args.model, whisper_dir, args.auto_download
    ):
        return 1

    try:
        result = transcribe_url(
            args.url,
            out_dir=args.out,
            whisper_dir=whisper_dir,
            model=args.model,
            language=args.language,
            keep_intermediate=args.keep_audio,
        )
    except LocalCaptionError as exc:
        log.error(str(exc))
        return 1

    log.info("transcript files:")
    for kind, path in result.transcripts.existing().items():
        print(f"  {kind:>4}: {path}")

    if not args.no_print:
        txt = result.transcripts.txt
        if txt.exists():
            print("\n" + "─" * 30 + " transcript " + "─" * 30)
            print(txt.read_text(encoding="utf-8", errors="replace"))

    return 0


# --- doctor subcommand ---------------------------------------------------

def _check(label: str, ok: bool, detail: str = "") -> bool:
    """Print a diagnostic line. Returns ``ok`` for chaining into a final exit code."""
    mark = "✅" if ok else "❌"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {mark} {label}{suffix}")
    return ok


def _run_doctor_diagnostics(whisper_dir: Path) -> tuple[bool, list[str], dict[str, bool]]:
    """Run all diagnostic checks and print results.

    Returns ``(all_ok, fix_hints, gaps)`` where ``gaps`` is a flag-bag the
    ``--fix`` path consumes to decide which install steps to run:

        {
            "ffmpeg":   bool,   # missing system tool
            "cmake":    bool,
            "git":      bool,
            "whisper":  bool,   # whisper.cpp missing or unbuilt
            "model":    bool,   # no ggml-*.bin model present
        }
    """
    gaps = {"ffmpeg": False, "cmake": False, "git": False,
            "whisper": False, "model": False}
    fix_hints: list[str] = []
    all_ok = True

    print("System tools:")
    all_ok &= _check("python", True, sys.version.split()[0])
    ff = shutil.which("ffmpeg")
    all_ok &= _check("ffmpeg", ff is not None, ff or "missing — `brew install ffmpeg`")
    if ff is None:
        gaps["ffmpeg"] = True
    cm = shutil.which("cmake")
    all_ok &= _check("cmake", cm is not None, cm or "missing — needed to build whisper.cpp")
    if cm is None:
        gaps["cmake"] = True
    git = shutil.which("git")
    all_ok &= _check("git", git is not None, git or "missing")
    if git is None:
        gaps["git"] = True

    print("\nPython dependencies:")
    try:
        import yt_dlp  # noqa: F401
        from yt_dlp.version import __version__ as ytdlp_ver
        all_ok &= _check("yt-dlp", True, ytdlp_ver)
    except ImportError:
        all_ok &= _check("yt-dlp", False, "missing — `pip install yt-dlp`")

    print("\nwhisper.cpp:")
    print(f"  searching: {whisper_dir}")

    if whisper_dir.is_dir():
        _check("directory exists", True, str(whisper_dir))
        paths = WhisperPaths(whisper_dir)
        try:
            binary = paths.find_binary()
            all_ok &= _check("binary built", True, str(binary))
        except LocalCaptionError as exc:
            all_ok &= _check("binary built", False, str(exc).splitlines()[0])
            gaps["whisper"] = True
            fix_hints.append(
                "whisper.cpp directory exists but isn't built. To build it:\n"
                f"    cd {whisper_dir}\n"
                "    cmake -B build && cmake --build build -j --config Release\n"
                "    (or just run: localcaption doctor --fix)"
            )

        models_dir = paths.models_dir
        if models_dir.is_dir():
            available = sorted(p.name for p in models_dir.glob("ggml-*.bin"))
            if available:
                _check("models present", True, ", ".join(available))
            else:
                all_ok &= _check("models present", False, f"no ggml-*.bin in {models_dir}")
                gaps["model"] = True
                fix_hints.append(
                    "No whisper models are installed. To list, pick, and download one:\n"
                    "    localcaption model list\n"
                    "    localcaption model download base.en   # ~142 MB; English-only\n"
                    "    localcaption model download small.en  # ~466 MB; better quality\n"
                    "    (or just run: localcaption doctor --fix)"
                )
        else:
            all_ok &= _check("models directory", False, str(models_dir))
            gaps["model"] = True
            fix_hints.append(
                f"models/ subdirectory missing under {whisper_dir} — "
                "your whisper.cpp clone may be incomplete; re-clone it."
            )
    else:
        all_ok &= _check("directory exists", False, str(whisper_dir))
        gaps["whisper"] = True
        gaps["model"] = True
        fix_hints.append(
            "whisper.cpp is not installed. Pick ONE of:\n\n"
            "  Option A — let localcaption install it for you:\n"
            "    localcaption doctor --fix\n\n"
            "  Option B — bootstrap from scratch (also installs localcaption):\n"
            "    curl -fsSL https://raw.githubusercontent.com/jatinkrmalik/"
            "localcaption/main/scripts/install.sh | bash\n\n"
            "  Option C — DIY, anywhere you like:\n"
            "    git clone https://github.com/ggerganov/whisper.cpp \\\n"
            "        ~/.local/share/localcaption/whisper.cpp\n"
            "    cd ~/.local/share/localcaption/whisper.cpp\n"
            "    cmake -B build && cmake --build build -j --config Release\n"
            "    bash models/download-ggml-model.sh base.en\n\n"
            "  Option D — point us at an existing whisper.cpp checkout:\n"
            "    export LOCALCAPTION_WHISPER_DIR=/path/to/your/whisper.cpp\n"
            "    # add that line to your shell rc to make it stick"
        )

    print("\nLookup paths searched:")
    for c in _candidate_whisper_dirs():
        marker = "✓" if c.is_dir() else "·"
        print(f"  {marker} {c}")

    return all_ok, fix_hints, gaps


def _apply_doctor_fix(whisper_dir: Path, gaps: dict[str, bool], model: str) -> bool:
    """Try to repair the gaps found by the diagnostic sweep.

    Returns ``True`` if every attempted fix succeeded, ``False`` otherwise.
    Each step is best-effort: a single failure is reported and the remaining
    steps are skipped (the user can re-run after addressing the root cause).
    """
    from . import installer, models  # local imports keep --help cheap

    print("\nAttempting fixes:\n")

    # 1. System dependencies first (whisper build needs cmake & git).
    for dep in ("git", "cmake", "ffmpeg"):
        if gaps.get(dep):
            print(f"▸ Installing system dependency: {dep}")
            try:
                installer.install_system_dep(dep)
            except LocalCaptionError as exc:
                print(f"  ❌ Could not install {dep}: {exc}")
                return False
            print(f"  ✅ {dep} installed")

    # 2. whisper.cpp clone + build.
    if gaps.get("whisper"):
        print(f"▸ Installing whisper.cpp into {whisper_dir}")
        try:
            binary = installer.ensure_whisper_cpp(whisper_dir)
        except LocalCaptionError as exc:
            print(f"  ❌ Could not install whisper.cpp: {exc}")
            return False
        print(f"  ✅ whisper.cpp ready ({binary})")

    # 3. Default model download (uses the same registry as `model download`).
    if gaps.get("model"):
        print(f"▸ Downloading model: {model}")
        try:
            target = models.download_model(model, whisper_dir)
        except LocalCaptionError as exc:
            print(f"  ❌ Could not download model: {exc}")
            return False
        print(f"  ✅ model downloaded ({target})")

    return True


def _cmd_doctor(argv: list[str]) -> int:
    """Diagnose a localcaption install. With ``--fix``, also try to repair it."""
    parser = argparse.ArgumentParser(
        prog="localcaption doctor",
        description="Diagnose a localcaption install: external tools, "
                    "whisper.cpp build, available models. "
                    "Use --fix to attempt automatic repair.",
    )
    parser.add_argument(
        "--whisper-dir", type=Path, default=None,
        help="check this whisper.cpp directory (default: auto-detect)",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="attempt to install missing dependencies (ffmpeg/cmake), "
             "clone+build whisper.cpp, and download the default model",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"model to download when fixing a missing-model gap "
             f"(default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)

    print(f"localcaption {__version__}\n")
    whisper_dir = args.whisper_dir or _default_whisper_dir()

    all_ok, fix_hints, gaps = _run_doctor_diagnostics(whisper_dir)

    if all_ok:
        print("\nAll checks passed. You're good to go: localcaption <url>")
        return 0

    if not args.fix:
        if fix_hints:
            print("\nHow to fix:\n")
            for hint in fix_hints:
                for line in hint.splitlines():
                    print(f"  {line}" if line else "")
                print()
        print("Some checks failed. See 'How to fix' above, or re-run with --fix.")
        return 1

    # --fix path: try to repair, then re-run diagnostics for verification.
    if not _apply_doctor_fix(whisper_dir, gaps, args.model):
        print("\nFix aborted. Address the error above and re-run.")
        return 1

    print("\n" + "─" * 60)
    print("Re-running diagnostics to verify…\n")
    all_ok_after, _, _ = _run_doctor_diagnostics(whisper_dir)
    if all_ok_after:
        print("\nAll checks passed. You're good to go: localcaption <url>")
        return 0
    print("\nSome checks still failing — see output above.")
    return 1


# --- top-level dispatcher -------------------------------------------------

# ──────────────────────────────────────────────────────────────────────
# `model` subcommand family
# ──────────────────────────────────────────────────────────────────────


def _format_size_mb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def _cmd_model(argv: list[str]) -> int:
    """Dispatch `localcaption model {list,download,rm,info}`."""
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(
            "usage: localcaption model <subcommand> [options]\n\n"
            "subcommands:\n"
            "  list                list every supported model + install status\n"
            "  download <name>     download a model (e.g. small.en)\n"
            "  rm <name>           remove an installed model\n"
            "  info <name>         show details about one model\n"
        )
        return 0 if argv and argv[0] in {"-h", "--help", "help"} else 2

    sub = argv[0]
    rest = argv[1:]
    if sub == "list":
        return _cmd_model_list(rest)
    if sub == "download":
        return _cmd_model_download(rest)
    if sub in {"rm", "remove", "delete"}:
        return _cmd_model_rm(rest)
    if sub == "info":
        return _cmd_model_info(rest)

    print(f"localcaption model: unknown subcommand: {sub}", file=sys.stderr)
    print("Run `localcaption model --help` to see available subcommands.", file=sys.stderr)
    return 2


def _cmd_model_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="localcaption model list")
    parser.add_argument("--whisper-dir", type=Path, default=None,
                        help="override whisper.cpp location")
    args = parser.parse_args(argv)

    # Avoid an immediate crash when whisper.cpp isn't installed yet — we still
    # want to be able to LIST models (so the user can decide what to download).
    whisper_dir = args.whisper_dir or _default_whisper_dir()

    # We don't import models at top of file to keep CLI startup lean
    from . import models

    table = models.list_status(whisper_dir)

    print("Models available for download (from whisper.cpp upstream):\n")
    print(f"  {'Name':<18}{'Size':>10}   Status")
    print(f"  {'-' * 18:<18}{'-' * 10:>10}   {'-' * 11}")
    for row in table:
        size = _format_size_mb(row.spec.approx_size_mb)
        status = "✅ installed" if row.is_installed else "not installed"
        print(f"  {row.spec.name:<18}{size:>10}   {status}")

    orphans = models.orphaned_installed_models(whisper_dir)
    if orphans:
        print("\nOther models found on disk (not in localcaption's registry):")
        for name in orphans:
            print(f"  {name}")
        print("These work fine with `--model <name>`; they just aren't shown above.")

    print(f"\nInstall location: {models.WhisperPaths(whisper_dir).models_dir}")
    print("\nTips:")
    print("  • Multilingual variants (no .en suffix) are required for non-English audio.")
    print("  • small.en is a great default for English podcasts/lectures.")
    print("  • To download:    localcaption model download small.en")
    print("  • To remove:      localcaption model rm small.en")
    return 0


def _cmd_model_info(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="localcaption model info")
    parser.add_argument("name", help="model name (e.g. small.en)")
    parser.add_argument("--whisper-dir", type=Path, default=None,
                        help="override whisper.cpp location")
    args = parser.parse_args(argv)

    from . import models
    try:
        spec = models.get_model(args.name)
    except LocalCaptionError as exc:
        print(f"localcaption: {exc}", file=sys.stderr)
        return 2

    whisper_dir = args.whisper_dir or _default_whisper_dir()
    target = models.model_path(whisper_dir, spec.name)
    installed = target.is_file()

    print(f"Model:        {spec.name}")
    print(f"Description:  {spec.description}")
    print(f"Approx size:  {_format_size_mb(spec.approx_size_mb)}")
    print(f"Language:     {'English-only' if spec.is_english_only else 'multilingual'}")
    print(f"Source URL:   {spec.url}")
    print(f"Local path:   {target}")
    if installed:
        actual_mb = max(1, target.stat().st_size // (1024 * 1024))
        print(f"Installed:    yes ({actual_mb} MB on disk)")
    else:
        print("Installed:    no")
        print(f"\nDownload with:  localcaption model download {spec.name}")
    return 0


def _cmd_model_download(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="localcaption model download")
    parser.add_argument("name", help="model name (e.g. small.en)")
    parser.add_argument("--whisper-dir", type=Path, default=None,
                        help="override whisper.cpp location")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the model is already present")
    args = parser.parse_args(argv)

    from . import models

    whisper_dir = args.whisper_dir or _default_whisper_dir()
    if not whisper_dir.is_dir():
        print(
            f"localcaption: whisper.cpp not found at {whisper_dir}.\n"
            "  Install it first: see `localcaption doctor`.",
            file=sys.stderr,
        )
        return 1

    try:
        spec = models.get_model(args.name)
    except LocalCaptionError as exc:
        print(f"localcaption: {exc}", file=sys.stderr)
        return 2

    print(f"▸ Downloading whisper model: {spec.name} (~{_format_size_mb(spec.approx_size_mb)})")
    try:
        path = models.download_model(spec.name, whisper_dir, force=args.force)
    except LocalCaptionError as exc:
        print(f"\nlocalcaption: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nlocalcaption: interrupted; partial download cleaned up.", file=sys.stderr)
        return 130

    print(f"✅ Done. {path}")
    print(f"   Use it with: localcaption --model {spec.name} <url>")
    return 0


def _cmd_model_rm(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="localcaption model rm")
    parser.add_argument("name", help="model name (e.g. small.en)")
    parser.add_argument("--whisper-dir", type=Path, default=None,
                        help="override whisper.cpp location")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    from . import models

    whisper_dir = args.whisper_dir or _default_whisper_dir()
    target = models.model_path(whisper_dir, args.name)
    if not target.is_file():
        print(
            f"localcaption: model {args.name!r} is not installed at {target}\n"
            "  Run `localcaption model list` to see installed models.",
            file=sys.stderr,
        )
        return 1

    size_mb = max(1, target.stat().st_size // (1024 * 1024))
    print(f"About to remove: {target} ({_format_size_mb(size_mb)})")
    if not args.yes:
        try:
            reply = input("Continue? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return 130
        if reply not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    try:
        models.remove_model(args.name, whisper_dir)
    except LocalCaptionError as exc:
        print(f"localcaption: {exc}", file=sys.stderr)
        return 1

    print("✓ Removed.")
    return 0


def _print_top_level_help() -> None:
    print("""\
usage: localcaption <url> [options]            transcribe a video (default)
       localcaption doctor                     diagnose your install
       localcaption model <subcommand>         list / download / remove models
       localcaption --help                     show transcribe help
       localcaption --version                  print version

Run `localcaption <subcommand> --help` for details on each.""")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Bare invocation → top-level help (exit non-zero, like most CLIs do).
    if not argv:
        _print_top_level_help()
        return 2

    head = argv[0]

    # Allow `localcaption help` as a friendly alias for the top-level help.
    if head in {"help", "--help-all"}:
        _print_top_level_help()
        return 0

    # Explicit subcommands.
    if head == "doctor":
        return _cmd_doctor(argv[1:])
    if head == "model":
        return _cmd_model(argv[1:])
    if head == "transcribe":
        return _cmd_transcribe(argv[1:])

    # Anything else (URL, --help, --version, …) goes to the default transcribe.
    return _cmd_transcribe(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

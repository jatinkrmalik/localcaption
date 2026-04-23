# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`localcaption model` subcommand family** for first-class whisper model
  management. No more shelling into `download-ggml-model.sh`. Subcommands:
  - `localcaption model list` — show every supported model with size +
    install status (works even before whisper.cpp is installed).
  - `localcaption model info <name>` — show metadata for one model.
  - `localcaption model download <name>` — native Python downloader with
    progress bar, atomic writes (`.part` → rename), Ctrl-C safety, and
    truncation detection.
  - `localcaption model rm <name>` — remove an installed model with a
    confirmation prompt (`-y` to skip).
- **Auto-download prompt for the transcribe path.** Running
  `localcaption --model small.en <url>` when `small.en` isn't installed now
  asks whether to download it, instead of failing with a path. Pass
  `--auto-download` to skip the prompt for scripted use.
- New `scripts/uninstall.sh` — idempotent end-to-end uninstaller with
  `--dry-run`, `--yes`, and `--keep-models` flags.
- **`localcaption doctor --fix`** — self-heals a broken or missing install
  end to end: installs missing system tools (`ffmpeg`/`cmake`/`git`) via
  `brew`/`apt`, clones + builds whisper.cpp at the canonical XDG location,
  downloads the requested model, then re-runs the diagnostics for
  verification. Idempotent (no-ops when already installed).
- New internal `localcaption.installer` module wrapping the install
  steps in pure Python (`subprocess`-based, no bash dependency) so the
  same logic runs from `doctor --fix`, the `install.sh` bootstrap, and
  any future entry point.
- New `InstallError` exception type for typed install-step failures.
- Unit tests: full coverage of `installer` (detection, subprocess
  wrapper, brew/apt selection, clone+build sequencing) and four new
  `doctor --fix` cases (read-only-by-default invariant, happy path,
  abort on failure, `--fix` advertised in `--help`). 68 tests total.

### Changed
- `doctor` now prints **actionable, copy-pasteable fix hints** when checks
  fail, including the new `localcaption model download <name>` recipe and
  the one-shot `localcaption doctor --fix` command.
- `doctor` also checks for `cmake` (needed to build whisper.cpp).
- `scripts/install.sh` slimmed from 134 → 91 lines: it now only handles
  the bootstrap (Python + `pipx` + `pipx install localcaption`) and then
  delegates the heavy lifting (whisper.cpp + model + system deps) to
  `localcaption doctor --fix`. Single source of truth.
- README has a new "Managing models" section with a model-picker table.
- The new `model` subcommand supersedes [#1] (switching the install default
  to `small.en`) — users now pick whatever model they want with one command.

### Fixed
- CI no longer prints the "Node.js 20 actions deprecated" warning on every
  run; opted in early to GitHub's Node 24 runtime via
  `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`.

## [0.1.1] - 2026-04-21

### Added
- `scripts/install.sh` — one-line end-user installer that uses `pipx` for an
  isolated install and bootstraps `whisper.cpp` + a default model into
  `~/.local/share/localcaption/whisper.cpp` (XDG-compliant). After install,
  `localcaption` is callable from any directory.
- New `localcaption doctor` subcommand that diagnoses prerequisites,
  whisper.cpp build, and available models. Used by the bug-report template.
- The CLI now searches a third location for whisper.cpp:
  `$XDG_DATA_HOME/localcaption/whisper.cpp` (after the explicit flag, env var,
  and `./whisper.cpp` dev path).

### Changed
- `localcaption` invoked with no arguments now prints top-level help and
  exits with code 2 (was: argparse error). Existing `localcaption <url>`
  usage is unchanged.

[#1]: https://github.com/jatinkrmalik/localcaption/issues/1

## [0.1.0] - 2026-04-21

### Added
- Initial public release as `localcaption`.
- Modular package layout: `download` (yt-dlp), `audio` (ffmpeg), `whisper`
  (whisper.cpp), orchestrated by `pipeline.transcribe_url`.
- `localcaption` console script and `python -m localcaption` entry point.
- One-shot `scripts/setup.sh` that builds whisper.cpp and downloads a model.
- MIT license, contributor docs, security policy, GitHub Actions CI.

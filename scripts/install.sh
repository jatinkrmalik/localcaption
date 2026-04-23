#!/usr/bin/env bash
# localcaption — end-user installer.
#
# Bootstraps the `localcaption` CLI (via pipx) and then delegates the
# heavy lifting (whisper.cpp clone+build, default model download, missing
# system tools) to `localcaption doctor --fix` so we have ONE source of
# truth for "what does a working install look like".
#
# Re-runnable: skips steps that are already done.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/jatinkrmalik/localcaption/main/scripts/install.sh | bash
#
#   # Or, in a local checkout:
#   ./scripts/install.sh
#
# Env vars:
#   WHISPER_MODEL              default: base.en
#   LOCALCAPTION_PACKAGE_SPEC  default: localcaption  (override with a local path or git URL)
#   PREFIX_DATA                default: $XDG_DATA_HOME or $HOME/.local/share

set -euo pipefail

MODEL="${WHISPER_MODEL:-base.en}"
PKG_SPEC="${LOCALCAPTION_PACKAGE_SPEC:-localcaption}"
DATA_DIR="${PREFIX_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}}/localcaption"
WHISPER_DIR="${DATA_DIR}/whisper.cpp"

log()  { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn   ]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[error  ]\033[0m %s\n" "$*" >&2; exit 1; }

# --- 0. Sanity ------------------------------------------------------------
case "$(uname -s)" in
  Darwin|Linux) ;;
  *) die "Unsupported OS: $(uname -s). localcaption is tested on macOS and Linux." ;;
esac

# --- 1. Prereq for *bootstrapping* localcaption itself --------------------
# We only need python3 here. ffmpeg / cmake / git get installed (if missing)
# by `localcaption doctor --fix` later — keeps install logic in one place.
command -v python3 >/dev/null 2>&1 \
  || die "python3 not found. Install Python 3.10+ first, then re-run."

# --- 2. pipx --------------------------------------------------------------
if ! command -v pipx >/dev/null 2>&1; then
  log "pipx not found, attempting install"
  if command -v brew >/dev/null 2>&1; then
    brew install pipx
    pipx ensurepath
  else
    python3 -m pip install --user --upgrade pipx
    python3 -m pipx ensurepath
  fi
  warn "If 'localcaption' isn't on PATH after this script finishes, open a new shell."
fi

# --- 3. Install (or upgrade) the localcaption package ---------------------
if pipx list 2>/dev/null | grep -q "package localcaption "; then
  log "Upgrading existing pipx install of localcaption"
  pipx upgrade localcaption || pipx install --force "${PKG_SPEC}"
else
  log "Installing localcaption via pipx (${PKG_SPEC})"
  pipx install "${PKG_SPEC}"
fi

# --- 4. Hand off to `localcaption doctor --fix` ---------------------------
# This is the single source of truth for "make this install work":
#   • installs missing system tools (ffmpeg, cmake, git) via brew/apt
#   • clones + builds whisper.cpp at WHISPER_DIR
#   • downloads the requested model
mkdir -p "${DATA_DIR}"

if ! command -v localcaption >/dev/null 2>&1; then
  warn "'localcaption' not yet on PATH (pipx may need a shell rehash)."
  warn "Skipping auto-fix. After opening a new shell, run:"
  warn "    LOCALCAPTION_WHISPER_DIR=${WHISPER_DIR} localcaption doctor --fix --model ${MODEL}"
else
  log "Running 'localcaption doctor --fix' to set up whisper.cpp + model"
  LOCALCAPTION_WHISPER_DIR="${WHISPER_DIR}" \
    localcaption doctor --fix --model "${MODEL}" \
    || die "Auto-fix failed. Run 'localcaption doctor' to diagnose, then retry."
fi

# --- 5. Done --------------------------------------------------------------
echo ""
log "Install complete!"
echo ""
echo "  Run:        localcaption <youtube-url>"
echo "  Diagnose:   localcaption doctor"
echo "  Auto-heal:  localcaption doctor --fix"
echo "  whisper.cpp lives at:  ${WHISPER_DIR}"
echo ""
if ! command -v localcaption >/dev/null 2>&1; then
  warn "'localcaption' is not on this shell's PATH yet."
  warn "Open a new terminal or run:  exec \"\$SHELL\""
fi

#!/usr/bin/env bash
# localcaption — end-user installer.
#
# Installs the `localcaption` command system-wide (via pipx) and bootstraps
# everything it needs (whisper.cpp + a default model) so you can immediately
# run `localcaption <youtube-url>` from any directory.
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
#   WHISPER_MODEL   default: base.en   (tiny.en | base.en | small.en | medium.en | large-v3)
#   LOCALCAPTION_PACKAGE_SPEC   default: localcaption  (override e.g. with a local path or git URL)
#   PREFIX_DATA     default: $XDG_DATA_HOME or $HOME/.local/share

set -euo pipefail

MODEL="${WHISPER_MODEL:-base.en}"
PKG_SPEC="${LOCALCAPTION_PACKAGE_SPEC:-localcaption}"
DATA_DIR="${PREFIX_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}}/localcaption"
WHISPER_DIR="${DATA_DIR}/whisper.cpp"

log()  { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn   ]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[error  ]\033[0m %s\n" "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1 — please install it first."; }

# --- 0. Sanity ------------------------------------------------------------
case "$(uname -s)" in
  Darwin|Linux) ;;
  *) die "Unsupported OS: $(uname -s). localcaption is tested on macOS and Linux." ;;
esac

# --- 1. System tools ------------------------------------------------------
log "Checking system tools…"
need_cmd python3
need_cmd git
need_cmd ffmpeg

if ! command -v cmake >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    log "cmake not found, installing via brew"
    brew install cmake
  elif command -v apt-get >/dev/null 2>&1; then
    log "cmake not found, installing via apt"
    sudo apt-get update -y && sudo apt-get install -y cmake
  else
    die "cmake not found and no brew/apt to install it. Please install cmake manually."
  fi
fi

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

# --- 4. whisper.cpp clone + build (into XDG data dir) ---------------------
mkdir -p "${DATA_DIR}"
if [[ ! -d "${WHISPER_DIR}" ]]; then
  log "Cloning whisper.cpp into ${WHISPER_DIR}"
  git clone --depth 1 https://github.com/ggerganov/whisper.cpp "${WHISPER_DIR}"
fi

BIN_PATH=""
for cand in \
  "${WHISPER_DIR}/build/bin/whisper-cli" \
  "${WHISPER_DIR}/build/bin/main" \
  "${WHISPER_DIR}/main"; do
  [[ -x "${cand}" ]] && BIN_PATH="${cand}" && break
done

if [[ -z "${BIN_PATH}" ]]; then
  log "Building whisper.cpp (this may take a minute)"
  pushd "${WHISPER_DIR}" >/dev/null
  if [[ -f CMakeLists.txt ]]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build build -j --config Release
  else
    make -j
  fi
  popd >/dev/null
fi

# --- 5. Download model ----------------------------------------------------
# Prefer `localcaption model download` (single source of truth, atomic writes,
# progress bar, friendly errors). Fall back to whisper.cpp's bash script if
# `localcaption` isn't on PATH yet (rare race; pipx may need a shell rehash).
MODEL_FILE="${WHISPER_DIR}/models/ggml-${MODEL}.bin"
if [[ -f "${MODEL_FILE}" ]]; then
  log "Model already present: ${MODEL_FILE}"
elif command -v localcaption >/dev/null 2>&1; then
  log "Downloading whisper model: ${MODEL} (via 'localcaption model download')"
  LOCALCAPTION_WHISPER_DIR="${WHISPER_DIR}" \
    localcaption model download "${MODEL}"
else
  log "Downloading whisper model: ${MODEL} (via whisper.cpp's bash script)"
  bash "${WHISPER_DIR}/models/download-ggml-model.sh" "${MODEL}"
fi

# --- 6. Done --------------------------------------------------------------
echo ""
log "Install complete!"
echo ""
echo "  Run:        localcaption <youtube-url>"
echo "  Diagnose:   localcaption doctor"
echo "  whisper.cpp lives at:  ${WHISPER_DIR}"
echo ""
if ! command -v localcaption >/dev/null 2>&1; then
  warn "'localcaption' is not on this shell's PATH yet."
  warn "Open a new terminal or run:  exec \"\$SHELL\""
fi

#!/usr/bin/env bash
# One-shot setup for localcaption.
# - creates a venv at ./.venv
# - pip-installs the package in editable mode (with [dev] extras if requested)
# - clones+builds whisper.cpp into ./whisper.cpp
# - downloads a default whisper ggml model
#
# Re-runnable: skips steps that are already done.
#
# Env vars:
#   WHISPER_MODEL   default: small.en  (tiny.en | base.en | small.en | medium.en | large-v3)
#   EXTRAS          default: dev       (dev | "" )

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="${REPO_ROOT}/whisper.cpp"
VENV_DIR="${REPO_ROOT}/.venv"
MODEL="${WHISPER_MODEL:-small.en}"
EXTRAS="${EXTRAS:-dev}"

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn ]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[err  ]\033[0m %s\n" "$*" >&2; exit 1; }

# --- 1. Required CLI tools -------------------------------------------------
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
need_cmd git
need_cmd make
need_cmd python3
need_cmd ffmpeg

if ! command -v cmake >/dev/null 2>&1; then
  warn "cmake not found, attempting brew install cmake"
  command -v brew >/dev/null 2>&1 || die "Please install cmake (e.g. brew install cmake)"
  brew install cmake
fi

# --- 2. Python venv + editable install ------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating virtualenv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
if [[ -n "${EXTRAS}" ]]; then
  pip install --quiet -e "${REPO_ROOT}[${EXTRAS}]"
else
  pip install --quiet -e "${REPO_ROOT}"
fi
log "localcaption installed in editable mode (extras: ${EXTRAS:-<none>})"

# --- 3. whisper.cpp clone + build -----------------------------------------
if [[ ! -d "${WHISPER_DIR}" ]]; then
  log "Cloning whisper.cpp"
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

# --- 4. Download model ----------------------------------------------------
MODEL_FILE="${WHISPER_DIR}/models/ggml-${MODEL}.bin"
if [[ ! -f "${MODEL_FILE}" ]]; then
  log "Downloading whisper model: ${MODEL}"
  bash "${WHISPER_DIR}/models/download-ggml-model.sh" "${MODEL}"
else
  log "Model already present: ${MODEL_FILE}"
fi

log "Setup complete."
log "Activate venv:   source .venv/bin/activate"
log "Run:             localcaption <youtube-url>"

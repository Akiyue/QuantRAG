#!/usr/bin/env bash
#
# Install llama-cpp-python with CUDA, picking whichever route the machine
# actually favours.
#
#   bash scripts/install_llamacpp.sh
#
# The prebuilt CUDA wheels are 1.3-1.8 GB because they ship kernels for every
# CUDA architecture. On a slow link that is hours, and pip cannot resume a
# broken download - it starts over.
#
# Building from source pulls ~70 MB and compiles kernels for one architecture,
# so on anything short of a fast connection it wins outright. It needs nvcc.
#
# Falls back to a resumable wheel download when nvcc is missing.
#
set -euo pipefail

# Ada Lovelace (RTX 5000 Ada, 4090, L40S) = 8.9. Ampere (A100) = 8.0,
# (A6000, 3090) = 8.6. Hopper (H100) = 9.0. Building one architecture instead
# of a dozen is where the size and time saving comes from.
ARCH="${CUDA_ARCH:-89}"
CUDA_TAG="${CUDA_TAG:-cu124}"
VERSION="${LLAMA_VERSION:-0.3.34}"
CACHE="${CACHE_DIR:-$HOME/.cache/quantrag}"

info() { printf '\033[32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "${CONDA_PREFIX:-}${VIRTUAL_ENV:-}" ]] \
  || die "activate your environment first: conda activate <env>"

if command -v nvcc >/dev/null 2>&1; then
  info "nvcc found: $(nvcc --version | tail -1)"
  info "building from source for sm_${ARCH} (~70 MB download, 15-25 min compile)"

  command -v cmake >/dev/null 2>&1 \
    || die "cmake missing: conda install -c conda-forge cmake ninja -y"

  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${ARCH}" \
    pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir \
    || die "build failed. If nvcc cannot find the host compiler, try:
  conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
Or fall back to the wheel route:
  NO_NVCC=1 bash scripts/install_llamacpp.sh"
else
  warn "no nvcc. Falling back to the prebuilt wheel - it is large."
  URL="https://github.com/abetlen/llama-cpp-python/releases/download/v${VERSION}-${CUDA_TAG}/llama_cpp_python-${VERSION}-py3-none-manylinux_2_35_x86_64.whl"
  mkdir -p "$CACHE"
  WHEEL="$CACHE/$(basename "$URL")"

  info "resumable download -> $WHEEL"
  info "safe to interrupt and re-run; it continues where it stopped"
  # -c is the whole point: pip restarts from zero on a dropped connection,
  # wget picks up from the last byte.
  wget -c --tries=0 --read-timeout=30 -O "$WHEEL" "$URL" \
    || die "download failed. Re-run this script to resume."

  info "installing"
  pip install --no-deps --force-reinstall "$WHEEL"
  pip install "diskcache>=5.6.1" "jinja2>=2.11.3" "numpy>=1.20.0" "typing-extensions>=4.5.0"
fi

info "verifying"
python "$(dirname "${BASH_SOURCE[0]}")/check_gpu.py"

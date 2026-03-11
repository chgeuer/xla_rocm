#!/usr/bin/env bash
#
# xla_rocm CUDA setup script
#
# Gets Nx + EXLA + Bumblebee running on NVIDIA GPUs (CUDA 12.8+)
# by building XLA from source, targeting Blackwell (sm_120) and earlier.
#
# Prerequisites:
#   - Arch Linux with CUDA 12.8+ and cuDNN installed
#   - NVIDIA GPU visible to nvidia-smi
#   - Elixir 1.17+ / OTP 26+ with mix in PATH
#   - clang, make, gcc
#
# Usage:
#   cd your_elixir_project
#   ./scripts/setup_cuda.sh
#
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[xla_cuda]${NC} $*"; }
ok()    { echo -e "${GREEN}[xla_cuda]${NC} $*"; }
warn()  { echo -e "${YELLOW}[xla_cuda]${NC} $*"; }
fail()  { echo -e "${RED}[xla_cuda]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

command -v elixir  >/dev/null || fail "elixir not found in PATH"
command -v mix     >/dev/null || fail "mix not found in PATH"
command -v clang   >/dev/null || fail "clang not found — install with: sudo pacman -S clang"
command -v make    >/dev/null || fail "make not found"

# Check for NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    [ -n "$GPU_NAME" ] && info "GPU: $GPU_NAME"
else
    warn "nvidia-smi not found. Continuing anyway..."
fi

# Check CUDA toolkit
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "unknown")
    info "CUDA version: $CUDA_VERSION"
else
    fail "nvcc not found. Install CUDA toolkit: sudo pacman -S cuda"
fi

# ---------------------------------------------------------------------------
# 2. Install system packages (Arch Linux)
# ---------------------------------------------------------------------------
info "Checking CUDA packages..."

PACKAGES=(
    bazelisk
    cuda
    cudnn
)

MISSING=()
for pkg in "${PACKAGES[@]}"; do
    pacman -Qi "$pkg" &>/dev/null || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    info "Installing: ${MISSING[*]}"
    sudo pacman -S --noconfirm --needed "${MISSING[@]}"
else
    ok "All CUDA packages already installed"
fi

# Remove bazel if installed (conflicts with bazelisk)
if pacman -Qi bazel &>/dev/null 2>&1; then
    warn "Removing bazel (conflicts with bazelisk)..."
    sudo pacman -Rns --noconfirm bazel || true
fi

# ---------------------------------------------------------------------------
# 3. Detect GPU compute capability
# ---------------------------------------------------------------------------
COMPUTE_CAP=""
if command -v nvidia-smi &>/dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || echo "")
fi

if [ -n "$COMPUTE_CAP" ]; then
    info "GPU compute capability: sm_${COMPUTE_CAP/./}"
else
    warn "Could not detect compute capability. Defaulting to sm_89 (Ada Lovelace)."
    COMPUTE_CAP="8.9"
fi

# ---------------------------------------------------------------------------
# 4. Verify CUDA libraries
# ---------------------------------------------------------------------------
info "Verifying CUDA libraries..."

CUDA_DIR="/opt/cuda"
[ -d "$CUDA_DIR" ] || CUDA_DIR="/usr/local/cuda"
[ -d "$CUDA_DIR" ] || fail "CUDA directory not found at /opt/cuda or /usr/local/cuda"

REQUIRED_LIBS=(cudart cublas cublasLt cufft curand cusolver cusparse cudnn)
MISSING_LIBS=()
for lib in "${REQUIRED_LIBS[@]}"; do
    if ! find "$CUDA_DIR/lib64" "$CUDA_DIR/lib" -maxdepth 1 -name "lib${lib}.so*" 2>/dev/null | grep -q .; then
        MISSING_LIBS+=("$lib")
    fi
done

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    warn "Missing CUDA libraries: ${MISSING_LIBS[*]}"
    warn "The build may fail. Try: sudo pacman -S cuda cudnn"
else
    ok "All CUDA libraries found"
fi

# ---------------------------------------------------------------------------
# 5. Fetch Elixir deps (triggers XLA source clone)
# ---------------------------------------------------------------------------
info "Fetching Elixir dependencies..."
mix deps.get

# ---------------------------------------------------------------------------
# 6. Trigger XLA source clone (first compile attempt clones XLA)
# ---------------------------------------------------------------------------
info "Compiling deps (this triggers the XLA source clone)..."
XLA_BUILD=true XLA_TARGET=cuda mix deps.compile xla --force 2>&1 | tail -1 || true

# ---------------------------------------------------------------------------
# 7. Patch the cloned XLA C++ source (if needed)
# ---------------------------------------------------------------------------
XLA_DIR=$(find "$HOME/.cache/xla_build" -maxdepth 1 -name 'xla-*' -type d 2>/dev/null | head -1)
[ -n "$XLA_DIR" ] || fail "XLA source not found in ~/.cache/xla_build/"
info "XLA source at $XLA_DIR"

# 7a. Restore .bazelversion (the Makefile deletes it, bazelisk needs it)
echo "7.4.1" > "$XLA_DIR/.bazelversion"
ok "Set .bazelversion to 7.4.1"

# 7b. Patch fine.hpp: add std::exception catch for better error messages
FINE_HPP="deps/fine/c_include/fine.hpp"
if grep -q 'catch (const std::exception &error)' "$FINE_HPP" 2>/dev/null; then
    ok "fine.hpp already patched"
else
    sed -i '/catch (const std::runtime_error &error) {/,/error.what());/{
        /error.what());/a\
  } catch (const std::exception \&error) {\
    return raise_error_with_message(env, __private__::atoms::ElixirRuntimeError,\
                                    error.what());
    }' "$FINE_HPP"
    ok "Added std::exception catch to fine.hpp"
fi

# ---------------------------------------------------------------------------
# 8. Build XLA + EXLA
# ---------------------------------------------------------------------------
info "Building XLA from source for CUDA (this takes 20-40 minutes)..."

export XLA_BUILD=true
export XLA_TARGET=cuda
export TF_CUDA_COMPUTE_CAPABILITIES="$COMPUTE_CAP"
export CUDA_DIR
export BUILD_FLAGS="--copt=-Wno-error=incompatible-pointer-types-discards-qualifiers"
export ELIXIR_ERL_OPTIONS="${ELIXIR_ERL_OPTIONS:-} +sssdio 128"
export CC=clang
export CXX=clang++

mix deps.clean xla --build
mix deps.compile xla --force

info "Building EXLA NIF..."
mix deps.clean exla --build
mix deps.compile exla

info "Compiling project..."
mix compile

# ---------------------------------------------------------------------------
# 9. Verify
# ---------------------------------------------------------------------------
info "Verifying..."

mix run -e '
platforms = EXLA.NIF.get_supported_platforms()
IO.inspect(platforms, label: "Platforms")

if Map.has_key?(platforms, :cuda) do
  t = Nx.tensor([1.0, 2.0, 3.0])
  result = Nx.multiply(t, t)
  IO.inspect(result, label: "GPU result")
  IO.puts("\n✅ CUDA GPU is working!")
else
  IO.puts("\n⚠️  CUDA platform not detected. GPU ops will run on CPU.")
  IO.puts("Check: nvidia-smi, nvcc --version")
end
'

ok "Setup complete!"
echo ""
echo "  Run your project with:  GPU_TARGET=cuda mix run -e 'your_code_here'"
echo "  Switch backend:         GPU_TARGET=host mix run ...  (CPU fallback)"
echo ""

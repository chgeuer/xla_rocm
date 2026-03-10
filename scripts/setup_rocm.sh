#!/usr/bin/env bash
#
# xla_rocm setup script
#
# Gets Nx + EXLA + Bumblebee running on AMD ROCm GPUs (ROCm 7.2+)
# by patching the XLA source for:
#   - ROCm 7.x SOVERSION detection (libamdhip64.so.7, librocblas.so.5)
#   - gfx1150/gfx1151 (Strix/Strix Halo APUs) in supported GPU list
#   - rocprim traits API change in ROCm 7.2
#   - Linker fix to preserve ROCm platform static initializer
#   - Clang host compiler for ROCm crosstool
#
# Prerequisites:
#   - Arch Linux with ROCm 7.2+ installed (at minimum: hsa-rocr, rocminfo)
#   - Elixir 1.17+ / OTP 26+ with mix in PATH
#   - clang, make, gcc
#
# Usage:
#   cd your_elixir_project
#   curl -sSL https://raw.githubusercontent.com/chgeuer/xla_rocm/master/scripts/setup_rocm.sh | bash
#   # or
#   ./scripts/setup_rocm.sh
#
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[xla_rocm]${NC} $*"; }
ok()    { echo -e "${GREEN}[xla_rocm]${NC} $*"; }
warn()  { echo -e "${YELLOW}[xla_rocm]${NC} $*"; }
fail()  { echo -e "${RED}[xla_rocm]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

command -v elixir  >/dev/null || fail "elixir not found in PATH"
command -v mix     >/dev/null || fail "mix not found in PATH"
command -v clang   >/dev/null || fail "clang not found — install with: sudo pacman -S clang"
command -v make    >/dev/null || fail "make not found"

ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "")
[ -n "$ROCM_VERSION" ] || fail "ROCm not found at /opt/rocm. Install ROCm 7.2+ first."
info "ROCm version: $ROCM_VERSION"

GPU_ARCH=$(rocminfo 2>/dev/null | grep -A2 'Agent [0-9]' | grep 'Name:.*gfx' | head -1 | grep -oP 'gfx\d+' || echo "")
[ -n "$GPU_ARCH" ] || warn "No AMD GPU detected by rocminfo. Continuing anyway..."
[ -n "$GPU_ARCH" ] && info "GPU architecture: $GPU_ARCH"

# ---------------------------------------------------------------------------
# 2. Install system packages (Arch Linux)
# ---------------------------------------------------------------------------
info "Installing ROCm packages and build tools..."

PACKAGES=(
    bazelisk
    rocm-hip-runtime  # hip-runtime-amd, rocm-llvm, rocm-cmake
    rocblas miopen-hip rocsolver hipfft hipsparse
    rccl rocrand hipblas hipsolver hiprand hipcub
)

MISSING=()
for pkg in "${PACKAGES[@]}"; do
    pacman -Qi "$pkg" &>/dev/null || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    info "Installing: ${MISSING[*]}"
    sudo pacman -S --noconfirm --needed "${MISSING[@]}"
else
    ok "All ROCm packages already installed"
fi

# Remove bazel if installed (conflicts with bazelisk)
if pacman -Qi bazel &>/dev/null 2>&1; then
    warn "Removing bazel (conflicts with bazelisk)..."
    sudo pacman -Rns --noconfirm bazel || true
fi

# ---------------------------------------------------------------------------
# 3. Fix known Arch Linux packaging issues
# ---------------------------------------------------------------------------
if [ -L /opt/rocm/lib/llvm/bin/flang ]; then
    TARGET=$(readlink /opt/rocm/lib/llvm/bin/flang)
    if [ "$TARGET" = "flang" ]; then
        info "Fixing circular symlink: /opt/rocm/lib/llvm/bin/flang"
        sudo rm /opt/rocm/lib/llvm/bin/flang
    fi
fi

# ---------------------------------------------------------------------------
# 4. Verify ROCm library detection
# ---------------------------------------------------------------------------
info "Verifying ROCm libraries..."

REQUIRED_LIBS=(amdhip64 rocblas MIOpen hipsparse hipfft hipsolver hiprand hipblas rocsolver rccl roctracer64 hipblaslt hipcub)
MISSING_LIBS=()
for lib in "${REQUIRED_LIBS[@]}"; do
    if ! find /opt/rocm/lib -maxdepth 1 -name "lib${lib}.so*" 2>/dev/null | grep -q .; then
        # Try alternate casing
        if ! find /opt/rocm/include -maxdepth 1 -iname "${lib}*" 2>/dev/null | grep -q .; then
            MISSING_LIBS+=("$lib")
        fi
    fi
done

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    warn "Missing ROCm libraries: ${MISSING_LIBS[*]}"
    warn "The build may fail. Try: sudo pacman -S <package-name>"
else
    ok "All ROCm libraries found"
fi

# ---------------------------------------------------------------------------
# 5. Fetch Elixir deps (triggers XLA source clone)
# ---------------------------------------------------------------------------
info "Fetching Elixir dependencies..."
mix deps.get

# ---------------------------------------------------------------------------
# 6. Patch Elixir-side files
# ---------------------------------------------------------------------------
info "Patching Elixir deps..."

# Patch xla.ex: add gfx1150/gfx1151 to AMDGPU targets
XLA_EX="deps/xla/lib/xla.ex"
if grep -q 'gfx1150' "$XLA_EX" 2>/dev/null; then
    ok "xla.ex already patched"
else
    sed -i 's/gfx1100,gfx1200/gfx1100,gfx1150,gfx1151,gfx1200/' "$XLA_EX"
    ok "Added gfx1150,gfx1151 to TF_ROCM_AMDGPU_TARGETS"
fi

# Patch extension/BUILD: use direct rocm:all_runtime dep
BUILD_FILE="deps/xla/extension/BUILD"
if grep -q 'rocm:all_runtime' "$BUILD_FILE" 2>/dev/null; then
    ok "extension/BUILD already patched"
else
    sed -i 's|//xla/stream_executor:rocm_platform|//xla/stream_executor/rocm:all_runtime|' "$BUILD_FILE"
    ok "Changed BUILD dep to rocm:all_runtime (preserves alwayslink)"
fi

# Patch fine.hpp: add std::exception catch for better error messages
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
# 7. Trigger XLA source clone (first compile attempt clones XLA)
# ---------------------------------------------------------------------------
info "Compiling deps (this triggers the XLA source clone)..."
# This will likely fail at the bazel build step, which is expected —
# we need to patch the cloned XLA source first.
XLA_BUILD=true XLA_TARGET=rocm mix deps.compile xla --force 2>&1 | tail -1 || true

# ---------------------------------------------------------------------------
# 8. Patch the cloned XLA C++ source
# ---------------------------------------------------------------------------
XLA_DIR=$(find "$HOME/.cache/xla_build" -maxdepth 1 -name 'xla-*' -type d 2>/dev/null | head -1)
[ -n "$XLA_DIR" ] || fail "XLA source not found in ~/.cache/xla_build/"
info "Patching XLA source at $XLA_DIR"

# 8a. Restore .bazelversion (the Makefile deletes it, bazelisk needs it)
echo "7.4.1" > "$XLA_DIR/.bazelversion"
ok "Set .bazelversion to 7.4.1"

# 8b. Add gfx1150/gfx1151 to kSupportedGfxVersions
DESC_H="$XLA_DIR/xla/stream_executor/device_description.h"
if grep -q 'gfx1150' "$DESC_H" 2>/dev/null; then
    ok "device_description.h already patched"
else
    sed -i 's/"gfx1101", "gfx1200"/"gfx1101", "gfx1150", "gfx1151", "gfx1200"/' "$DESC_H"
    ok "Added gfx1150/gfx1151 to kSupportedGfxVersions"
fi

# 8c. Fix SOVERSION for ROCm 7.x
ROCM_BZL="$XLA_DIR/third_party/gpus/rocm_configure.bzl"
if grep -q 'rocm_version_number) >= 70000' "$ROCM_BZL" 2>/dev/null; then
    ok "rocm_configure.bzl already patched"
else
    sed -i \
        's/"6" if int(rocm_config.rocm_version_number) >= 60000 else "5"/"7" if int(rocm_config.rocm_version_number) >= 70000 else ("6" if int(rocm_config.rocm_version_number) >= 60000 else "5")/g' \
        "$ROCM_BZL"
    sed -i \
        's/"4" if int(rocm_config.rocm_version_number) >= 60000 else "3"/"5" if int(rocm_config.rocm_version_number) >= 70000 else ("4" if int(rocm_config.rocm_version_number) >= 60000 else "3")/g' \
        "$ROCM_BZL"
    ok "Fixed SOVERSION detection for ROCm 7.x"
fi

# 8d. Patch cub_sort_kernel_rocm.cu.cc for ROCm 7.2 rocprim API
CUB_FILE="$XLA_DIR/xla/stream_executor/rocm/cub_sort_kernel_rocm.cu.cc"
if grep -q 'TF_ROCM_VERSION >= 70200' "$CUB_FILE" 2>/dev/null; then
    ok "cub_sort_kernel_rocm.cu.cc already patched"
else
    python3 - "$CUB_FILE" << 'PYEOF'
import sys
f = sys.argv[1]
with open(f) as fh:
    content = fh.read()

old = """// Required for sorting Eigen::half and bfloat16.
namespace rocprim {
namespace detail {

#if (TF_ROCM_VERSION >= 50200)
template <>
struct float_bit_mask<Eigen::half> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7C00;
  static constexpr uint16_t mantissa = 0x03FF;
  using bit_type = uint16_t;
};

template <>
struct float_bit_mask<tsl::bfloat16> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7F80;
  static constexpr uint16_t mantissa = 0x007F;
  using bit_type = uint16_t;
};
#endif  // TF_ROCM_VERSION >= 50200
template <>
struct radix_key_codec_base<Eigen::half>
    : radix_key_codec_floating<Eigen::half, uint16_t> {};
template <>
struct radix_key_codec_base<tsl::bfloat16>
    : radix_key_codec_floating<tsl::bfloat16, uint16_t> {};
};  // namespace detail
};  // namespace rocprim"""

new = """// Required for sorting Eigen::half and bfloat16.
// ROCm 7.2+ uses a new traits-based API in rocprim.
#if (TF_ROCM_VERSION >= 70200)
template <>
struct rocprim::traits::define<Eigen::half> {
  using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
  using number_format = rocprim::traits::number_format::values<
      rocprim::traits::number_format::kind::floating_point_type>;
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7C00, 0x03FF>;
};

template <>
struct rocprim::traits::define<tsl::bfloat16> {
  using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
  using number_format = rocprim::traits::number_format::values<
      rocprim::traits::number_format::kind::floating_point_type>;
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
};
#else
namespace rocprim {
namespace detail {

#if (TF_ROCM_VERSION >= 50200)
template <>
struct float_bit_mask<Eigen::half> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7C00;
  static constexpr uint16_t mantissa = 0x03FF;
  using bit_type = uint16_t;
};

template <>
struct float_bit_mask<tsl::bfloat16> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7F80;
  static constexpr uint16_t mantissa = 0x007F;
  using bit_type = uint16_t;
};
#endif  // TF_ROCM_VERSION >= 50200
template <>
struct radix_key_codec_base<Eigen::half>
    : radix_key_codec_floating<Eigen::half, uint16_t> {};
template <>
struct radix_key_codec_base<tsl::bfloat16>
    : radix_key_codec_floating<tsl::bfloat16, uint16_t> {};
};  // namespace detail
};  // namespace rocprim
#endif  // TF_ROCM_VERSION >= 70200"""

if old in content:
    content = content.replace(old, new)
    with open(f, "w") as fh:
        fh.write(content)
    print("Patched cub_sort_kernel_rocm.cu.cc")
else:
    print("WARNING: Could not find expected code in cub_sort_kernel_rocm.cu.cc")
    print("The file may have already been patched or the XLA version differs.")
PYEOF
fi

# ---------------------------------------------------------------------------
# 9. Build XLA + EXLA
# ---------------------------------------------------------------------------
info "Building XLA from source for ROCm (this takes 20-40 minutes)..."

export XLA_BUILD=true
export XLA_TARGET=rocm
export TF_ROCM_CLANG=1
export CLANG_COMPILER_PATH=/usr/bin/clang
export BUILD_FLAGS="--copt=-Wno-error=incompatible-pointer-types-discards-qualifiers --linkopt=-Wl,--no-gc-sections"
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
# 10. Verify
# ---------------------------------------------------------------------------
info "Verifying..."

mix run -e '
platforms = EXLA.NIF.get_supported_platforms()
IO.inspect(platforms, label: "Platforms")

if Map.has_key?(platforms, :rocm) do
  t = Nx.tensor([1.0, 2.0, 3.0])
  result = Nx.multiply(t, t)
  IO.inspect(result, label: "GPU result")
  IO.puts("\n✅ ROCm GPU is working!")
else
  IO.puts("\n⚠️  ROCm platform not detected. GPU ops will run on CPU.")
  IO.puts("Check: rocminfo, dmesg | grep amdgpu")
end
'

ok "Setup complete!"
echo ""
echo "  Run your project with:  mix run -e 'your_code_here'"
echo "  Switch backend:         GPU_TARGET=host mix run ...  (CPU fallback)"
echo ""

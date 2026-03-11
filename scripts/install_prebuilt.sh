#!/usr/bin/env bash
#
# Quick installer for consumer projects that use the pre-built XLA archive.
#
# Instead of the hour-long Bazel build, this downloads the pre-built archive
# from GitHub releases and compiles only the small EXLA NIF wrapper (~1 min).
#
# Usage (from your project directory):
#   curl -sSL https://raw.githubusercontent.com/chgeuer/xla_rocm/master/scripts/install_prebuilt.sh | bash
#   # or
#   /path/to/xla_rocm/scripts/install_prebuilt.sh
#
# Environment variables:
#   XLA_ROCM_VERSION  — Release tag (default: auto-detect from deps)
#   XLA_ROCM_TARGET   — Target platform: rocm, cuda (default: rocm)
#   XLA_ARCHIVE_URL   — Override archive URL entirely
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
# 1. Detect settings
# ---------------------------------------------------------------------------
TARGET="${XLA_ROCM_TARGET:-rocm}"
ARCH=$(uname -m)          # x86_64, aarch64
OS="linux"
ABI="gnu"

# Try to detect XLA version from mix.lock or mix.exs
XLA_VERSION="${XLA_ROCM_VERSION:-}"
if [ -z "$XLA_VERSION" ]; then
    if [ -f mix.lock ]; then
        XLA_VERSION=$(grep -oP '"xla": {:hex, :xla, "\K[^"]+' mix.lock 2>/dev/null || echo "")
    fi
    if [ -z "$XLA_VERSION" ] && [ -f mix.exs ]; then
        XLA_VERSION=$(grep -oP '{:xla, "~> \K[^"]+' mix.exs 2>/dev/null || echo "")
    fi
    [ -n "$XLA_VERSION" ] || fail "Could not detect XLA version. Set XLA_ROCM_VERSION=0.9.1"
fi

# The pre-built archive was compiled against specific versions.
# Warn if the consumer project uses incompatible versions.
EXPECTED_XLA="0.9.1"
EXPECTED_EXLA="0.10.0"
if [ -f mix.lock ]; then
    ACTUAL_XLA=$(grep -oP '"xla": {:hex, :xla, "\K[^"]+' mix.lock 2>/dev/null || echo "")
    ACTUAL_EXLA=$(grep -oP '"exla": {:hex, :exla, "\K[^"]+' mix.lock 2>/dev/null || echo "")
    if [ -n "$ACTUAL_XLA" ] && [ "$ACTUAL_XLA" != "$EXPECTED_XLA" ]; then
        warn "mix.lock has xla $ACTUAL_XLA but the pre-built archive was compiled with xla $EXPECTED_XLA"
        warn "This will likely cause 'undefined symbol' errors. Pin deps in mix.exs:"
        warn '  {:nx, "~> 0.10.0"}, {:exla, "~> 0.10.0"}'
        warn "Then run: mix deps.get"
        fail "Version mismatch — fix mix.exs deps first"
    fi
    if [ -n "$ACTUAL_EXLA" ] && [ "$ACTUAL_EXLA" != "$EXPECTED_EXLA" ]; then
        warn "mix.lock has exla $ACTUAL_EXLA but the pre-built archive expects exla $EXPECTED_EXLA"
        warn "Pin deps in mix.exs: {:exla, \"~> 0.10.0\"}"
        fail "Version mismatch — fix mix.exs deps first"
    fi
fi

ARCHIVE_NAME="xla_extension-${XLA_VERSION}-${ARCH}-${OS}-${ABI}-${TARGET}.tar.gz"
RELEASE_TAG="v${XLA_VERSION}-${TARGET}"

if [ -n "${XLA_ARCHIVE_URL:-}" ]; then
    ARCHIVE_URL="$XLA_ARCHIVE_URL"
else
    ARCHIVE_URL="https://github.com/chgeuer/xla_rocm/releases/download/${RELEASE_TAG}/${ARCHIVE_NAME}"
fi

CACHE_DIR="${XLA_CACHE_DIR:-$HOME/.cache/xla}"
ARCHIVE_PATH="${CACHE_DIR}/${XLA_VERSION}/prebuilt/${ARCHIVE_NAME}"

info "XLA version: $XLA_VERSION"
info "Target: $TARGET ($ARCH-$OS-$ABI)"
info "Archive: $ARCHIVE_NAME"

# ---------------------------------------------------------------------------
# 2. Download (or reuse cached) archive
# ---------------------------------------------------------------------------
if [ -f "$ARCHIVE_PATH" ]; then
    ok "Archive already cached at $ARCHIVE_PATH"
else
    info "Downloading from $ARCHIVE_URL ..."
    mkdir -p "$(dirname "$ARCHIVE_PATH")"

    if command -v curl &>/dev/null; then
        curl -fSL -o "$ARCHIVE_PATH" "$ARCHIVE_URL" || fail "Download failed. Is the release published?"
    elif command -v wget &>/dev/null; then
        wget -q -O "$ARCHIVE_PATH" "$ARCHIVE_URL" || fail "Download failed. Is the release published?"
    else
        fail "Neither curl nor wget found"
    fi

    ok "Downloaded $(du -h "$ARCHIVE_PATH" | cut -f1) archive"
fi

# ---------------------------------------------------------------------------
# 3. Verify archive
# ---------------------------------------------------------------------------
info "Verifying archive..."
tar tzf "$ARCHIVE_PATH" | head -3 | grep -q 'xla_extension/' || fail "Archive doesn't contain xla_extension/ — wrong file?"
ok "Archive looks valid"

# ---------------------------------------------------------------------------
# 4. Install Elixir deps and compile with prebuilt archive
# ---------------------------------------------------------------------------
info "Fetching Elixir dependencies..."
mix deps.get

# Patch fine.hpp if needed (for better error messages)
FINE_HPP="deps/fine/c_include/fine.hpp"
if [ -f "$FINE_HPP" ] && ! grep -q 'catch (const std::exception &error)' "$FINE_HPP" 2>/dev/null; then
    info "Patching fine.hpp..."
    sed -i '/catch (const std::runtime_error &error) {/,/error.what());/{
        /error.what());/a\
  } catch (const std::exception \&error) {\
    return raise_error_with_message(env, __private__::atoms::ElixirRuntimeError,\
                                    error.what());
    }' "$FINE_HPP"
    ok "Patched fine.hpp"
fi

info "Compiling with prebuilt XLA archive (~1 minute)..."

export XLA_ARCHIVE_PATH="$ARCHIVE_PATH"
export XLA_TARGET="$TARGET"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export ELIXIR_ERL_OPTIONS="${ELIXIR_ERL_OPTIONS:-} +sssdio 128"

mix deps.compile xla --force
mix deps.compile exla --force
mix compile

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
info "Verifying..."

PLATFORM_KEY="$TARGET"
mix run -e "
platforms = EXLA.NIF.get_supported_platforms()
IO.inspect(platforms, label: \"Platforms\")

if Map.has_key?(platforms, :$PLATFORM_KEY) do
  IO.puts(\"✅ #{String.upcase(\"$TARGET\")} platform detected!\")
else
  IO.puts(\"⚠️  :$PLATFORM_KEY not in platforms. Available: #{inspect(Map.keys(platforms))}\")
end
"

ok "Done! Pre-built XLA archive installed."
echo ""
echo "  The archive is cached at: $ARCHIVE_PATH"
echo "  Future 'mix deps.compile' will reuse it (no re-download)."
echo ""

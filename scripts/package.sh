#!/usr/bin/env bash
#
# Package the locally-built XLA archive for distribution.
#
# After running setup_rocm.sh or setup_cuda.sh, this script:
#   1. Finds the built archive in ~/.cache/xla/
#   2. Computes SHA256 checksums
#   3. Copies everything to dist/ ready for GitHub release
#
# Usage:
#   ./scripts/package.sh           # package all built archives
#   ./scripts/package.sh rocm      # package only ROCm archive
#   ./scripts/package.sh cuda      # package only CUDA archive
#
set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BLUE}[package]${NC} $*"; }
ok()   { echo -e "${GREEN}[package]${NC} $*"; }
fail() { echo -e "${RED}[package]${NC} $*"; exit 1; }

XLA_VERSION=$(grep -oP '{:xla, "~> \K[^"]+' mix.exs || echo "0.9")
CACHE_DIR="${XLA_CACHE_DIR:-$HOME/.cache/xla}"
DIST_DIR="dist"
FILTER="${1:-all}"

mkdir -p "$DIST_DIR"

found=0

for archive in "$CACHE_DIR"/"$XLA_VERSION"*/build/xla_extension-*.tar.gz \
               "$CACHE_DIR"/"$XLA_VERSION"*/download/xla_extension-*.tar.gz; do
    [ -f "$archive" ] || continue

    filename=$(basename "$archive")

    # Apply filter
    case "$FILTER" in
        rocm) echo "$filename" | grep -q 'rocm' || continue ;;
        cuda) echo "$filename" | grep -q 'cuda' || continue ;;
        cpu)  echo "$filename" | grep -q 'cpu'  || continue ;;
        all)  ;;
        *)    fail "Unknown filter: $FILTER (use: rocm, cuda, cpu, all)" ;;
    esac

    # Skip the upstream CPU download — only package our builds
    if echo "$archive" | grep -q '/download/' && echo "$filename" | grep -q 'cpu'; then
        continue
    fi

    info "Packaging $filename ($(du -h "$archive" | cut -f1))"

    cp "$archive" "$DIST_DIR/$filename"
    sha256sum "$DIST_DIR/$filename" | awk '{print $1 "  " $2}' | sed "s|$DIST_DIR/||" > "$DIST_DIR/$filename.sha256"

    checksum=$(cat "$DIST_DIR/$filename.sha256" | awk '{print $1}')
    ok "$filename → $checksum"
    last_filename="$filename"

    found=$((found + 1))
done

if [ "$found" -eq 0 ]; then
    fail "No archives found. Run setup_rocm.sh or setup_cuda.sh first."
fi

echo ""
ok "Packaged $found archive(s) in $DIST_DIR/"
echo ""
echo "  To publish as a GitHub release:"
echo "    gh release create v${XLA_VERSION}-rocm --title 'XLA ${XLA_VERSION} with ROCm patches' $DIST_DIR/*"
echo ""
echo "  Consumer projects install with:"
echo "    export XLA_ARCHIVE_URL=https://github.com/chgeuer/xla_rocm/releases/download/v${XLA_VERSION}-rocm/$last_filename"
echo "    mix deps.get && mix deps.compile"
echo ""

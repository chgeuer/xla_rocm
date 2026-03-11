# Building CUDA XLA in Docker

The XLA CUDA build requires a specific toolchain range (glibc ≤ 2.39, GCC ≤ 14, Clang ≤ 20, CUDA ≤ 12.8). Arch Linux's rolling-release packages are typically ahead of this range. This document describes how to build the CUDA XLA archive in a Docker container with a compatible toolchain.

## Prerequisites

- Docker (no GPU runtime needed — the build only compiles, doesn't run on GPU)
- ~10 GB disk for the builder image + bazel cache
- ~30–60 min build time

## Build

### 1. Build the Docker image

The image is based on `nvidia/cuda:12.8.0-devel-ubuntu24.04` with Erlang, Elixir, and Bazelisk added.

```bash
cd /path/to/xla_rocm
docker build -t xla-cuda-builder -f docker/Dockerfile.cuda .
```

This takes ~2 min (cached after first build).

### 2. Run the XLA build

```bash
mkdir -p dist

docker run --rm \
  -v "$PWD":/src:ro \
  -v "$PWD/dist":/output \
  -v xla-cuda-cache:/root/.cache \
  xla-cuda-builder bash -c '
set -euo pipefail
cp -r /src/. /build/ && cd /build

# Remove any host-specific LOCAL_CUDA_PATH patches (container uses hermetic CUDA)
sed -i "/LOCAL_CUDA_PATH/d" deps/xla/lib/xla.ex
sed -i "/LOCAL_CUDNN_PATH/d" deps/xla/lib/xla.ex

mix deps.get
XLA_BUILD=true XLA_TARGET=cuda mix deps.compile xla --force 2>&1 | tail -1 || true

# Find XLA source and apply infeed staging patch
XLA_DIR=$(find ~/.cache/xla_build -maxdepth 1 -name "xla-*" -type d | head -1)
echo "7.4.1" > "$XLA_DIR/.bazelversion"

python3 - "$XLA_DIR/xla/service/gpu/infeed_manager.cc" << PYEOF
import sys
f = sys.argv[1]
with open(f) as fh:
    c = fh.read()
if "HostMemoryAllocate" in c:
    print("Already patched"); sys.exit(0)
c = c.replace("#include <cstdint>", "#include <cstdint>\n#include <cstring>")
old = "  TF_RETURN_IF_ERROR(stream->Memcpy(buffer.memory_ptr(), source, size));\n\n  return std::move(buffer);"
new = """  auto pinned = executor->HostMemoryAllocate(size);
  if (pinned.ok()) {
    std::memcpy((*pinned)->opaque(), source, size);
    TF_RETURN_IF_ERROR(stream->Memcpy(buffer.memory_ptr(), (*pinned)->opaque(), size));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  } else {
    TF_RETURN_IF_ERROR(stream->Memcpy(buffer.memory_ptr(), source, size));
  }
  return std::move(buffer);"""
c = c.replace(old, new)
with open(f, "w") as fh:
    fh.write(c)
print("Patched infeed_manager.cc")
PYEOF

XLA_BUILD=true XLA_TARGET=cuda CC=clang CXX=clang++ mix deps.compile xla --force

ARCHIVE=$(find ~/.cache/xla -name "*cuda*.tar.gz" -path "*/build/*" | head -1)
cp "$ARCHIVE" /output/xla_extension-0.9.1-x86_64-linux-gnu-cuda.tar.gz
cd /output && sha256sum *.tar.gz
'
```

Build takes 30–60 min. The bazel cache is stored in a Docker volume (`xla-cuda-cache`) and survives between runs — incremental rebuilds are much faster.

### 3. Verify the output

```bash
ls -lh dist/xla_extension-0.9.1-x86_64-linux-gnu-cuda.tar.gz
# ~210 MB archive
```

## Using the CUDA archive in a consumer project (e.g. basileus)

The CUDA archive requires setup on the host to work alongside a ROCm build:

### Extract the CUDA binary

```bash
cd /path/to/basileus
mkdir -p _build/cuda/lib/exla/priv/xla_extension
tar xzf /path/to/xla_rocm/dist/xla_extension-0.9.1-x86_64-linux-gnu-cuda.tar.gz \
  -C _build/cuda/lib/exla/priv/

# Copy the host-compiled libexla.so into the CUDA priv dir
# (the NIF wrapper is platform-independent; only libxla_extension.so differs)
cp deps/exla/cache/libexla.so _build/cuda/lib/exla/priv/libexla.so
```

### Extract CUDA runtime libraries

The CUDA archive was built against hermetic CUDA 12.8 libraries. The host system may have CUDA 13.x with different SONAMEs. Extract the matching 12.8 runtime libs from the Docker build cache:

```bash
mkdir -p _build/cuda_libs

docker run --rm \
  -v xla-cuda-cache:/cache:ro \
  -v "$PWD/_build/cuda_libs":/output \
  ubuntu:24.04 bash -c '
BAZEL="/cache/bazel/_bazel_root/*/external"
for dir in cuda_cudart cuda_cublas cuda_cufft cuda_cusolver cuda_cusparse \
           cuda_nvcc cuda_cupti cuda_nvjitlink cuda_nvrtc; do
  LIB=$(find $BAZEL -path "*/$dir/lib" -type d 2>/dev/null | head -1)
  [ -d "$LIB" ] && cp -L "$LIB"/*.so* /output/ 2>/dev/null
done
for dir in $(find $BAZEL -maxdepth 1 -name "cuda_nccl*" -type d 2>/dev/null); do
  [ -d "$dir/lib" ] && cp -L "$dir/lib"/*.so* /output/ 2>/dev/null
done
'
```

### Compile the CUDA build variant

```bash
XLA_TARGET=cuda MIX_BUILD_PATH=_build/cuda mix compile
```

### Start a CUDA worker node

```elixir
# From iex on the main (ROCm) node:
cuda_libs = Path.join(File.cwd!(), "_build/cuda_libs")
Basileus.Cluster.LocalWorker.start_link(
  name: "cuda_worker",
  gpu_target: "cuda",
  build_path: "_build/cuda",
  extra_env: [{"LD_LIBRARY_PATH", cuda_libs}]
)
```

Or to auto-start on boot, add to `application.ex`'s children list (see [basileus clustering docs](../../basileus/docs/plans/clustering.md)).

## What the Dockerfile contains

```
nvidia/cuda:12.8.0-devel-ubuntu24.04
├── build-essential, clang, git, curl, python3
├── erlang, elixir (from Ubuntu apt)
├── bazelisk (GitHub release binary)
└── mix local.hex + rebar
```

The image is ~4 GB. CUDA 12.8 devel headers and nvcc are included for compiling GPU kernels. No NVIDIA driver or GPU access is needed — the build only generates PTX/SASS code, it doesn't execute on GPU.

## Hermetic CUDA vs system CUDA

The XLA build uses **hermetic CUDA**: Bazel downloads specific CUDA 12.8 redistributable packages (cudart, cublas, cufft, etc.) regardless of what's installed on the host. This ensures reproducible builds. The `LOCAL_CUDA_PATH` env var can override this to use system CUDA — but this requires version compatibility between system headers and XLA's expected API.

Inside the Docker container, we use the hermetic build (no `LOCAL_CUDA_PATH`), which avoids all version compatibility issues.

# xla_rocm

Get [Nx](https://github.com/elixir-nx/nx), [EXLA](https://github.com/elixir-nx/nx/tree/main/exla), and [Bumblebee](https://github.com/elixir-nx/bumblebee) running on AMD GPUs via ROCm and NVIDIA GPUs via CUDA.

Pre-compiled EXLA binaries don't support ROCm 7.2, recent AMD GPU architectures (gfx1150, gfx1151), or Blackwell (sm_120). This project provides setup scripts that patch the XLA source and build from source, so you can run GPU-accelerated ML workloads on AMD and NVIDIA hardware from Elixir.

## Tested hardware

| Machine | CPU | GPU | Arch | RAM |
|---------|-----|-----|------|-----|
| Laptop | Ryzen AI 9 HX 370 | Radeon 890M | gfx1150 | 96 GB |
| Desktop | Ryzen AI MAX+ 395 | Radeon 8060S | gfx1151 | 96 GB |

Both use unified memory (iGPU shares system RAM).

NVIDIA GPUs with CUDA 12.8+ are also supported (e.g. RTX 5070, Blackwell sm_120).

## Prerequisites

- **Arch Linux**
- **Elixir** 1.17+ / OTP 26+
- **clang**, **make**, **gcc**
- For ROCm: ROCm 7.2+ (`hsa-rocr`, `rocminfo` at minimum)
- For CUDA: CUDA Toolkit 12.8+, cuDNN

## Quick start

### AMD GPU (ROCm)

```bash
git clone https://github.com/chgeuer/xla_rocm
cd xla_rocm
./scripts/setup_rocm.sh    # installs packages, patches XLA, builds (~20-40 min)
```

### NVIDIA GPU (CUDA)

```bash
git clone https://github.com/chgeuer/xla_rocm
cd xla_rocm
./scripts/setup_cuda.sh    # installs packages, builds XLA for CUDA (~20-40 min)
```

Then:

```bash
# Tensor ops on the GPU
mix run -e 'IO.inspect(Nx.multiply(Nx.tensor([1,2,3]), Nx.tensor([4,5,6])))'
#=> #Nx.Tensor<s32[3] EXLA.Backend<rocm:0, ...> [4, 10, 18]>

# BERT fill-mask inference on the GPU
just bert "The capital of France is [MASK]."
```

## Using pre-built binaries in other projects

The full XLA build takes 20-40 minutes. Consumer projects can skip that entirely
by using the pre-built archive from [GitHub releases](https://github.com/chgeuer/xla_rocm/releases).
Only the small EXLA NIF wrapper needs to compile locally (~1 minute).

### Option A: One-liner install

From your project directory:

```bash
curl -sSL https://raw.githubusercontent.com/chgeuer/xla_rocm/master/scripts/install_prebuilt.sh | bash
```

This downloads the archive, patches deps, and compiles EXLA against it.

### Option B: Manual setup

1. Add EXLA to your `mix.exs` — **pin versions to match the pre-built archive**
   and set the archive URL so rebuilds are automatic:

   ```elixir
   defmodule MyApp.MixProject do
     use Mix.Project

     # Pre-built ROCm XLA archive — skips the hour-long Bazel build.
     # Set XLA_BUILD=true to build from source instead.
     xla_rocm_archive_url =
       "https://github.com/chgeuer/xla_rocm/releases/download/v0.9.1-rocm/xla_extension-0.9.1-x86_64-linux-gnu-rocm.tar.gz"

     unless System.get_env("XLA_BUILD") do
       System.put_env("XLA_ARCHIVE_URL", System.get_env("XLA_ARCHIVE_URL") || xla_rocm_archive_url)
       System.put_env("XLA_TARGET", System.get_env("XLA_TARGET") || "rocm")
     end

     System.put_env("CC", System.get_env("CC") || "clang")
     System.put_env("CXX", System.get_env("CXX") || "clang++")

     # ... rest of mix.exs ...

     defp deps do
       [
         {:nx, "~> 0.10.0"},
         {:exla, "~> 0.10.0"}
       ]
     end
   end
   ```

   This way `mix deps.get && mix deps.compile` just works — even after
   wiping `deps/` and `_build/`. No env vars to remember, no justfile needed.

   > ⚠️ The pre-built archive is compiled against xla 0.9.1 / exla 0.10.0.
   > Using a newer exla (e.g. 0.11.0) will fail with `undefined symbol` errors
   > because the C++ ABI changed between versions.

2. Configure your `config/config.exs`:

   ```elixir
   config :nx, :default_backend, {EXLA.Backend, client: :rocm}
   config :exla, :clients, rocm: [platform: :rocm, preallocate: false]
   config :nx, :default_defn_options, compiler: EXLA, client: :rocm
   ```

3. Compile:

   ```bash
   mix deps.get && mix deps.compile
   ```

   That's it — `mix.exs` handles the archive URL and compiler settings automatically.

### Building and publishing new archives

If you've built XLA from source and want to publish updated archives:

```bash
just package rocm         # creates dist/ with archive + SHA256
just release v0.9.1-rocm  # publishes to GitHub releases
```

## What the setup scripts do

### ROCm (`setup_rocm.sh`)

1. Installs ROCm packages (`hip-runtime-amd`, `rocblas`, `miopen-hip`, `hipcub`, …)
2. Installs `bazelisk` (downloads the correct Bazel 7.4.1 for XLA)
3. Fixes Arch-specific issues (circular `flang` symlink)
4. Patches the XLA source (see below)
5. Builds XLA from source with ROCm support
6. Compiles the EXLA NIF with `clang++` (required for C++ ABI compatibility)

### CUDA (`setup_cuda.sh`)

1. Installs CUDA toolkit and cuDNN
2. Installs `bazelisk` (downloads the correct Bazel 7.4.1 for XLA)
3. Auto-detects GPU compute capability (e.g. sm_120 for Blackwell)
4. Builds XLA from source with CUDA support
5. Compiles the EXLA NIF with `clang++`

### XLA patches

The upstream XLA source (pinned by the `xla` hex package) needs 5 fixes for ROCm 7.2:

| Patch | File | Issue |
|-------|------|-------|
| SOVERSION | `third_party/gpus/rocm_configure.bzl` | XLA looks for `libamdhip64.so.6`, ROCm 7.x ships `.so.7` |
| GPU arch | `xla/stream_executor/device_description.h` | `gfx1150`/`gfx1151` not in `kSupportedGfxVersions` |
| rocprim API | `xla/stream_executor/rocm/cub_sort_kernel_rocm.cu.cc` | `rocprim::detail::float_bit_mask` removed in ROCm 7.2 |
| Linker | `BUILD_FLAGS` | `--no-gc-sections` preserves the ROCm platform static initializer |
| Build dep | `deps/xla/extension/BUILD` | Direct dep on `rocm:all_runtime` to preserve `alwayslink` |

Full patch details with diffs are in [BUILD_FROM_SCRATCH.md](BUILD_FROM_SCRATCH.md).

## Usage

### justfile recipes

```
just setup-rocm   # one-time ROCm build (~20-40 min)
just setup-cuda   # one-time CUDA build (~20-40 min)
just test-rocm    # smoke test — verify AMD GPU works
just test-cuda    # smoke test — verify NVIDIA GPU works
just info         # show platforms & config
just iex          # interactive Elixir shell
just bert         # BERT fill-mask on default GPU
just bert-cuda    # BERT fill-mask on NVIDIA GPU
just test-cpu     # run on CPU instead
just rocm-info    # AMD GPU system info
just cuda-info    # NVIDIA GPU system info
just start-rocm   # start named ROCm node (for clustering)
just start-cuda   # start named CUDA node (for clustering)
just cluster      # connect both GPU nodes & show info
```

### GPU target selection

Set `GPU_TARGET` to switch backends at runtime:

```bash
GPU_TARGET=rocm mix run ...  # AMD GPU (default)
GPU_TARGET=host mix run ...  # CPU fallback
GPU_TARGET=cuda mix run ...  # NVIDIA GPU (requires separate CUDA build)
```

### Elixir API

```elixir
# Check what's available
XlaRocm.info()
#=> %{platforms: %{host: 24, rocm: 1}, rocm_available: true, ...}

# Quick verification
XlaRocm.smoke_test()

# Tensor operations (run on GPU automatically)
Nx.multiply(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))

# Bumblebee inference
{:ok, model} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"})
{:ok, tok} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})
serving = Bumblebee.Text.fill_mask(model, tok)
Nx.Serving.run(serving, "The capital of [MASK] is Paris.")
```

## Memory configuration

AMD APUs with unified memory need `preallocate: false` to prevent EXLA from grabbing 90% of system RAM on startup. This is already configured in `config/config.exs`.

For large models (70B+), increase the kernel TTM limit to allow the GPU to address more system RAM:

```bash
# Temporary (until reboot)
echo 20971520 | sudo tee /sys/module/ttm/parameters/pages_limit

# Persistent — add to /etc/default/grub GRUB_CMDLINE_LINUX_DEFAULT:
#   amdttm.pages_limit=20971520 ttm.pages_limit=20971520
```

For massive models, use lazy transfers to avoid memory spikes:

```elixir
serving = Bumblebee.Text.generation(
  model_info, tokenizer, generation_config,
  defn_options: [compiler: EXLA, lazy_transfers: :always]
)
```

## Using both GPUs simultaneously

CUDA and ROCm EXLA cannot coexist in the same BEAM process (C++ linker collisions). Use distributed Erlang with short names to run each GPU in its own node:

```bash
# Terminal 1 — start ROCm node
just start-rocm
# or: GPU_TARGET=rocm elixir --sname rocm -S mix run --no-halt

# Terminal 2 — start CUDA node
just start-cuda
# or: GPU_TARGET=cuda elixir --sname cuda -S mix run --no-halt

# Terminal 3 — connect from a client
elixir --sname client -S mix run -e '
  XlaRocm.Cluster.connect(:rocm)
  XlaRocm.Cluster.connect(:cuda)
  IO.inspect(XlaRocm.Cluster.gpu_info(), pretty: true)

  # Run work on a specific GPU
  XlaRocm.Cluster.run_on(:cuda, fn ->
    Nx.multiply(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
  end)
'
```

Or use `just cluster` to connect and show GPU info from both nodes.

## Project structure

```
├── config/
│   ├── config.exs          # EXLA client config (ROCm, preallocate: false)
│   └── runtime.exs         # GPU_TARGET env var switching
├── lib/
│   ├── xla_rocm.ex         # XlaRocm.info/0, XlaRocm.smoke_test/0
│   └── xla_rocm/
│       └── cluster.ex      # Distributed Erlang clustering for dual-GPU
├── scripts/
│   ├── setup_rocm.sh       # Automated ROCm setup: packages, patches, build
│   ├── setup_cuda.sh       # Automated CUDA setup: packages, build
│   ├── package.sh          # Package built archives for distribution
│   └── install_prebuilt.sh # Quick install for consumer projects (~1 min)
├── justfile                 # Task runner recipes
├── BUILD_FROM_SCRATCH.md    # Detailed build instructions & patch diffs
└── mix.exs
```

## References

- [elixir-nx/xla](https://github.com/elixir-nx/xla) — Pre-compiled XLA extension for Elixir
- [Increasing VRAM on AMD AI APUs](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux/) — Jeff Geerling
- [ROCm documentation](https://rocm.docs.amd.com/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) — NVIDIA CUDA
- [Bumblebee](https://github.com/elixir-nx/bumblebee) — Transformer models for Elixir
- [Distributed Erlang](https://www.erlang.org/doc/system/distributed.html) — Erlang distribution protocol

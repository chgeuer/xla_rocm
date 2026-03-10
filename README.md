# xla_rocm

Get [Nx](https://github.com/elixir-nx/nx), [EXLA](https://github.com/elixir-nx/nx/tree/main/exla), and [Bumblebee](https://github.com/elixir-nx/bumblebee) running on AMD GPUs via ROCm.

Pre-compiled EXLA binaries don't support ROCm 7.2 or recent AMD GPU architectures (gfx1150, gfx1151). This project provides a setup script that patches the XLA source and builds from source, so you can run GPU-accelerated ML workloads on AMD hardware from Elixir.

## Tested hardware

| Machine | CPU | GPU | Arch | RAM |
|---------|-----|-----|------|-----|
| Laptop | Ryzen AI 9 HX 370 | Radeon 890M | gfx1150 | 96 GB |
| Desktop | Ryzen AI MAX+ 395 | Radeon 8060S | gfx1151 | 96 GB |

Both use unified memory (iGPU shares system RAM).

## Prerequisites

- **Arch Linux** with ROCm 7.2+ (`hsa-rocr`, `rocminfo` at minimum)
- **Elixir** 1.17+ / OTP 26+
- **clang**, **make**, **gcc**
- An AMD GPU visible to `rocminfo`

## Quick start

```bash
git clone https://github.com/chgeuer/xla_rocm
cd xla_rocm
./scripts/setup_rocm.sh    # installs packages, patches XLA, builds (~20-40 min)
```

Then:

```bash
# Tensor ops on the GPU
mix run -e 'IO.inspect(Nx.multiply(Nx.tensor([1,2,3]), Nx.tensor([4,5,6])))'
#=> #Nx.Tensor<s32[3] EXLA.Backend<rocm:0, ...> [4, 10, 18]>

# BERT fill-mask inference on the GPU
just bert "The capital of France is [MASK]."
```

## What the setup script does

1. Installs ROCm packages (`hip-runtime-amd`, `rocblas`, `miopen-hip`, `hipcub`, …)
2. Installs `bazelisk` (downloads the correct Bazel 7.4.1 for XLA)
3. Fixes Arch-specific issues (circular `flang` symlink)
4. Patches the XLA source (see below)
5. Builds XLA from source with ROCm support
6. Compiles the EXLA NIF with `clang++` (required for C++ ABI compatibility)

### XLA patches

The upstream XLA source (pinned by the `xla` hex package) needs 5 fixes for ROCm 7.2:

| Patch | File | Issue |
|-------|------|-------|
| SOVERSION | `third_party/gpus/rocm_configure.bzl` | XLA looks for `libamdhip64.so.6`, ROCm 7.x ships `.so.7` |
| GPU arch | `xla/stream_executor/device_description.h` | `gfx1150`/`gfx1151` not in `kSupportedGfxVersions` |
| rocprim API | `xla/stream_executor/rocm/cub_sort_kernel_rocm.cu.cc` | `rocprim::detail::float_bit_mask` removed in ROCm 7.2 |
| Linker | `BUILD_FLAGS` | `--no-gc-sections` preserves the ROCm platform static initializer |
| Build dep | `deps/xla/extension/BUILD` | Direct dep on `rocm:all_runtime` to preserve `alwayslink` |

Full patch details with diffs are in [INSTALL.md](INSTALL.md).

## Usage

### justfile recipes

```
just setup      # one-time build (~20-40 min)
just test       # smoke test — verify GPU works
just info       # show platforms & config
just iex        # interactive Elixir shell
just bert       # BERT fill-mask (default prompt)
just bert "I love [MASK] music."  # custom prompt
just test-cpu   # run on CPU instead
just rocm-info  # system GPU info
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

## Project structure

```
├── config/
│   ├── config.exs          # EXLA client config (ROCm, preallocate: false)
│   └── runtime.exs         # GPU_TARGET env var switching
├── lib/
│   └── xla_rocm.ex         # XlaRocm.info/0, XlaRocm.smoke_test/0
├── scripts/
│   └── setup_rocm.sh       # Automated setup: packages, patches, build
├── justfile                 # Task runner recipes
├── INSTALL.md               # Detailed build instructions & patch diffs
└── mix.exs
```

## References

- [elixir-nx/xla](https://github.com/elixir-nx/xla) — Pre-compiled XLA extension for Elixir
- [Increasing VRAM on AMD AI APUs](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux/) — Jeff Geerling
- [ROCm documentation](https://rocm.docs.amd.com/)
- [Bumblebee](https://github.com/elixir-nx/bumblebee) — Transformer models for Elixir

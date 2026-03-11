# Installation Guide: Nx + EXLA + Bumblebee on AMD Radeon 890M (ROCm) & NVIDIA RTX 5070 (CUDA)

## System Information

- **CPU:** AMD Ryzen AI 9 HX 370 (24 threads)
- **RAM:** 96GB
- **GPU 1:** NVIDIA GeForce RTX 5070 Max-Q (8GB VRAM, sm_120 Blackwell)
- **GPU 2:** AMD Radeon 890M (gfx1150, unified memory / iGPU)
- **OS:** Arch Linux, kernel 6.18.13-arch1-1
- **Elixir:** 1.20.0-rc.2 / OTP 28

## Phase 1: System Prerequisites (ROCm / AMD Radeon 890M)

### 1.1 — Install Bazelisk (Bazel version manager)

XLA requires Bazel 7.4.1, but Arch ships Bazel 9.0.0 which is incompatible.
`bazelisk` auto-downloads the correct Bazel version based on `.bazelversion`.

```bash
# If bazel is already installed, remove it first (conflicts with bazelisk)
sudo pacman -Rns bazel

# Install bazelisk
sudo pacman -S --noconfirm bazelisk
```

### 1.2 — Install ROCm HIP Runtime and Math Libraries

The XLA build system (`rocm_configure.bzl`) requires these ROCm packages:

```bash
# HIP runtime (provides hipcc, hip_version.h, libamdhip64.so)
sudo pacman -S --noconfirm rocm-hip-runtime

# ROCm math libraries (all required by XLA)
sudo pacman -S --noconfirm rocblas miopen-hip rocsolver hipfft hipsparse rccl rocrand hipblas

# Additional required libraries
sudo pacman -S --noconfirm hipsolver hiprand hipcub
```

**Full list of ROCm packages installed:**
- `hsa-rocr` (already installed — HSA runtime)
- `rocm-core` (already installed — version files)
- `rocm-device-libs` (already installed)
- `rocminfo` (already installed)
- `rocm-hip-runtime` (meta: hip-runtime-amd, rocm-llvm, rocm-cmake, rocm-language-runtime)
- `rocblas`, `rocsolver`, `rocrand`, `rocfft`, `roctracer`
- `hipblas`, `hipblaslt`, `hipfft`, `hipsparse`, `hipsolver`, `hiprand`
- `miopen-hip`
- `rccl`
- `hipcub`

### 1.3 — Fix Circular Symlink (Arch-specific)

Arch's ROCm packages ship a broken circular symlink:

```bash
# /opt/rocm/lib/llvm/bin/flang -> flang (points to itself)
sudo rm /opt/rocm/lib/llvm/bin/flang
```

### 1.4 — EXLA NIF Must Be Compiled with Clang

The XLA extension is compiled with clang. The EXLA NIF must also use clang++
to avoid C++ ABI mismatches (the exception catch hierarchy breaks with mixed
gcc/clang++ compiled objects):

```bash
export CC=clang
export CXX=clang++
```

### 1.4 — Verify ROCm Library Detection

Run the XLA config detection script to verify all libraries are found:

```bash
cd ~/.cache/xla_build/xla-*/
ROCM_PATH=/opt/rocm python3 third_party/gpus/find_rocm_config.py
```

Expected output should list version numbers for: `hipfft`, `hipruntime`, `hipsolver`,
`hipsparse`, `miopen`, `rocblas`, `rocfft`, `rocm_toolkit_path`, `rocm_version_number`,
`rocrand`, `rocsolver`, `roctracer`.

## Phase 2: Elixir Project Setup

### 2.1 — Add Dependencies to `mix.exs`

```elixir
defp deps do
  [
    {:nx, "~> 0.9"},
    {:exla, "~> 0.9"},
    {:bumblebee, "~> 0.6"}
  ]
end
```

### 2.2 — Create `config/config.exs`

```elixir
import Config

config :nx, :default_backend, EXLA.Backend

# CRITICAL: preallocate: false for the Radeon 890M!
# It shares system RAM — without this, EXLA grabs 90% of 96GB and freezes the OS.
config :exla, :clients,
  rocm: [platform: :rocm, preallocate: false]

config :nx, :default_defn_options, compiler: EXLA, client: :rocm
```

### 2.3 — Create `config/runtime.exs`

```elixir
import Config

gpu_target =
  System.get_env("GPU_TARGET", "rocm")
  |> String.downcase()
  |> String.to_atom()

case gpu_target do
  :cuda ->
    config :exla, :clients,
      cuda: [platform: :cuda, preallocate: true, memory_fraction: 0.8]
    config :nx, :default_defn_options, compiler: EXLA, client: :cuda

  :rocm ->
    config :exla, :clients,
      rocm: [platform: :rocm, preallocate: false]
    config :nx, :default_defn_options, compiler: EXLA, client: :rocm
end
```

## Phase 3: Build EXLA from Source for ROCm

### 3.1 — Restore `.bazelversion` (Required for Bazelisk)

The XLA Makefile deletes `.bazelversion` during the clone step. Bazelisk needs it
to download the correct Bazel version (7.4.1):

```bash
echo "7.4.1" > ~/.cache/xla_build/xla-*/. bazelversion
```

### 3.2 — Build with Environment Variables

```bash
cd /path/to/ex_nx_framework

# Required: Build from source targeting ROCm
export XLA_BUILD=true
export XLA_TARGET=rocm

# Required: Use clang as host compiler (gcc doesn't support -Qunused-arguments)
export TF_ROCM_CLANG=1
export CLANG_COMPILER_PATH=/usr/bin/clang

# Required: Suppress boringssl const-pointer warning and preserve ROCm platform initializer
export BUILD_FLAGS="--copt=-Wno-error=incompatible-pointer-types-discards-qualifiers --linkopt=-Wl,--no-gc-sections"

# Recommended: Prevent CUDA compiler from crashing the Erlang VM
export ELIXIR_ERL_OPTIONS="+sssdio 128"

# Fetch and compile
mix deps.get
mix deps.compile
```

**⚠️ This build takes 30-60+ minutes** and uses significant CPU and RAM.

### Troubleshooting Build Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `No repository visible as '@rules_python'` | Bazel 9.0.0 incompatibility | Use `bazelisk` + `.bazelversion` with 7.4.1 |
| `HIP Runtime version file not found` | Missing `hip-runtime-amd` | `sudo pacman -S rocm-hip-runtime` |
| `Cannot find rocm library hiprand` | Missing ROCm math libs | Install all packages from 1.2 |
| `Too many levels of symbolic links` | Circular symlink in ROCm | `sudo rm /opt/rocm/lib/llvm/bin/flang` |
| `gcc: unrecognized option -Qunused-arguments` | GCC used as host compiler | Set `TF_ROCM_CLANG=1` |
| `C++ ABI mismatch / unknown exception in NIF` | EXLA NIF compiled with gcc, XLA with clang | Set `CC=clang CXX=clang++` when compiling EXLA |
| `missing input file hipcub_version.hpp` | Missing hipcub package | `sudo pacman -S hipcub` |
| `explicit specialization of undeclared template float_bit_mask` | ROCm 7.2 changed rocprim API | See Patch 1 below |
| `unsupported AMDGPU version: gfx1150` | gfx1150 not in XLA's supported list | See Patch 2 below |
| `Could not load libamdhip64.so.6` | SOVERSION wrong for ROCm 7.x | See Patch 3 below |

## Phase 4: TTM Memory Limits (For Large Models)

The Radeon 890M shares system RAM. By default, the kernel limits GPU access to ~47GB.
For large models (70B+), increase the limit:

```bash
# Calculate pages for 80GB: (80 * 1024 * 1024 * 1024) / 4096 = 20971520

# Option A: Runtime (temporary, no reboot)
echo 20971520 | sudo tee /sys/module/ttm/parameters/pages_limit

# Option B: Persistent (requires reboot)
# Add to /etc/default/grub GRUB_CMDLINE_LINUX_DEFAULT:
#   amdttm.pages_limit=20971520 ttm.pages_limit=20971520
sudo grub-mkconfig -o /boot/grub/grub.cfg
# Then reboot

# Verify
cat /sys/module/ttm/parameters/pages_limit
# Should show: 20971520
```

## Phase 5: NVIDIA RTX 5070 Setup (After ROCm Works)

### 5.1 — Automated Setup

The fastest path is the setup script, which mirrors the ROCm setup:

```bash
./scripts/setup_cuda.sh
```

This script:
1. Checks for CUDA toolkit and cuDNN
2. Auto-detects GPU compute capability (e.g. sm_120 for Blackwell)
3. Sets `TF_CUDA_COMPUTE_CAPABILITIES` accordingly
4. Builds XLA from source with `XLA_TARGET=cuda`
5. Verifies CUDA platform detection

### 5.2 — Manual Setup

Install CUDA Toolkit and cuDNN:

```bash
sudo pacman -S --noconfirm cuda cudnn
```

### 5.3 — Rebuild EXLA for CUDA

```bash
export XLA_BUILD=true
export XLA_TARGET=cuda
export TF_CUDA_COMPUTE_CAPABILITIES=12.0  # for Blackwell sm_120
export ELIXIR_ERL_OPTIONS="+sssdio 128"
export CC=clang
export CXX=clang++
mix deps.compile exla --force
```

### 5.4 — Update Config

Add CUDA client alongside ROCm in `config/config.exs`:

```elixir
config :exla, :clients,
  rocm: [platform: :rocm, preallocate: false],
  cuda: [platform: :cuda, preallocate: true, memory_fraction: 0.8]
```

**⚠️ Important:** You cannot run CUDA and ROCm EXLA in the same BEAM process
(C++ linker collisions). For dual-GPU, use distributed Erlang clustering — see below.

## Phase 6: Dual-GPU Clustering (Both GPUs Simultaneously)

CUDA and ROCm EXLA link different C++ runtimes and cannot coexist in one process.
Use distributed Erlang with `--sname` to run each GPU in its own BEAM node:

```bash
# Terminal 1: ROCm node
GPU_TARGET=rocm elixir --sname rocm -S mix run --no-halt

# Terminal 2: CUDA node
GPU_TARGET=cuda elixir --sname cuda -S mix run --no-halt

# Terminal 3: Client that connects to both
elixir --sname client -S mix run -e '
  XlaRocm.Cluster.connect(:rocm)
  XlaRocm.Cluster.connect(:cuda)
  IO.inspect(XlaRocm.Cluster.gpu_info(), pretty: true)
'
```

Or use the justfile recipes: `just start-rocm`, `just start-cuda`, `just cluster`.

### Dispatching work to a specific GPU

```elixir
# Run on NVIDIA GPU
XlaRocm.Cluster.run_on(:cuda, fn ->
  Nx.multiply(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
end)

# Run on AMD GPU
XlaRocm.Cluster.run_on(:rocm, fn ->
  Nx.multiply(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
end)
```

## Current Status

### ✅ Fully Working

- **Nx + EXLA on ROCm GPU:** AMD Radeon 890M (gfx1150) running natively via ROCm 7.2.
  Tensor operations, defn JIT compilation, and Bumblebee BERT fill-mask inference
  all running on the GPU with `EXLA.Backend<rocm:0>`.
- **CPU fallback:** Also available via `GPU_TARGET=host` if needed.

### Key Patches Applied to XLA Source

The XLA source is cloned by the `xla` Elixir dep during `mix deps.compile` into:

```
~/.cache/xla_build/xla-<git-rev>/
```

For xla 0.9.1, the directory is `~/.cache/xla_build/xla-870d90fd098c480fb8a426126bd02047adb2bc20/`.
All patches below are relative to that directory.

#### Patch 1: `xla/stream_executor/rocm/cub_sort_kernel_rocm.cu.cc`

ROCm 7.2 replaced `rocprim::detail::float_bit_mask` with a new `rocprim::traits::define<T>` API.
Replace the `namespace rocprim { namespace detail { ... } }` block (lines 32–60) with:

```cpp
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
// ... keep the old code for ROCm < 7.2 ...
#endif
```

#### Patch 2: `xla/stream_executor/device_description.h`

Add `"gfx1150"` to `kSupportedGfxVersions[]` (around line 160):

```cpp
  static constexpr absl::string_view kSupportedGfxVersions[]{
      "gfx900",   // MI25
      "gfx906",   // MI50 / MI60
      "gfx908",   // MI100
      "gfx90a",   // MI200
      "gfx942",   // MI300
      "gfx950",   // MI355
      "gfx1030",  // RX68xx / RX69xx
      "gfx1100",  // RX7900
      "gfx1101", "gfx1150",  // Radeon 890M (Strix APU)
      "gfx1200", "gfx1201",
  };
```

#### Patch 3: `third_party/gpus/rocm_configure.bzl`

Fix SOVERSION detection for ROCm 7.x (two places, lines ~736 and ~755):

```python
# Before (wrong for ROCm 7.x):
"%{hip_soversion_number}": "6" if int(rocm_config.rocm_version_number) >= 60000 else "5",
"%{rocblas_soversion_number}": "4" if int(rocm_config.rocm_version_number) >= 60000 else "3",

# After:
"%{hip_soversion_number}": "7" if int(rocm_config.rocm_version_number) >= 70000 else ("6" if int(rocm_config.rocm_version_number) >= 60000 else "5"),
"%{rocblas_soversion_number}": "5" if int(rocm_config.rocm_version_number) >= 70000 else ("4" if int(rocm_config.rocm_version_number) >= 60000 else "3"),
```

#### Patch 4: `deps/xla/extension/BUILD` (in the Elixir project)

Change the ROCm platform dependency to use the direct target (preserves `alwayslink`):

```python
# Before:
"//xla/stream_executor:rocm_platform"

# After:
"//xla/stream_executor/rocm:all_runtime"
```

#### Patch 5: `deps/xla/lib/xla.ex` (in the Elixir project)

Add `gfx1150` to the AMDGPU targets list:

```elixir
# Before:
~s/--action_env=TF_ROCM_AMDGPU_TARGETS="gfx900,...,gfx1100,gfx1200,gfx1201"/

# After:
~s/--action_env=TF_ROCM_AMDGPU_TARGETS="gfx900,...,gfx1100,gfx1150,gfx1200,gfx1201"/
```

### Running with GPU_TARGET

Use the `GPU_TARGET` environment variable to select the backend:

```bash
# ROCm GPU (default)
mix run -e 'IO.inspect(Nx.multiply(Nx.tensor([1,2,3]), Nx.tensor([4,5,6])))'

# CPU fallback
GPU_TARGET=host mix run -e '...'

# CUDA (once CUDA toolkit installed and EXLA rebuilt)
GPU_TARGET=cuda mix run -e '...'
```

```elixir
# In iex -S mix:
t = Nx.tensor([1, 2, 3])
Nx.multiply(t, t)
# Should return: #Nx.Tensor<s32[3] [1, 4, 9]>
```

### Test Bumblebee

```elixir
{:ok, model_info} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})
serving = Bumblebee.Text.fill_mask(model_info, tokenizer)
Nx.Serving.run(serving, "The capital of France is [MASK].")
```

For large models on the 890M, use lazy transfers:

```elixir
serving = Bumblebee.Text.generation(
  model_info,
  tokenizer,
  generation_config,
  defn_options: [compiler: EXLA, lazy_transfers: :always]
)
```

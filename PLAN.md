# Practical Guide: Elixir ML on RTX 5070 & Radeon 890M

Pre-compiled EXLA binaries don't support these GPUs yet. You need to build XLA from source.

## 1. NVIDIA RTX 5070 (Blackwell, sm_120)

Standard binaries max out at sm_90. You need CUDA 12.8+ to target sm_120.

**Prerequisites:** CUDA Toolkit 12.8, cuDNN 9.7+, Clang 18, Bazel 7.

**Environment variables:**

```bash
export XLA_BUILD=true
export XLA_TARGET=cuda
export ELIXIR_ERL_OPTIONS="+sssdio 128"  # prevent CUDA compiler from crashing the BEAM
```

**Nx config:**

```elixir
config :nx, :default_backend, {EXLA.Backend, client: :cuda}
config :nx, :default_defn_options, [compiler: EXLA, client: :cuda]
```

## 2. AMD Radeon 890M (Strix, gfx1150) & 96GB unified RAM

The 890M shares system RAM. The kernel restricts GPU access by default, causing OOM on large models.

### Step A: Unlock system RAM for the GPU

Calculate pages for 80 GB: `(80 * 1024 * 1024 * 1024) / 4096 = 20971520`

Add to kernel boot parameters (`/etc/default/grub`):

```
amdttm.pages_limit=20971520 ttm.pages_limit=20971520
```

Update grub and reboot. Verify with `dmesg | grep "amdgpu.*memory"`.

### Step B: Install ROCm and build EXLA

ROCm 7.2 added gfx1150 support. Build EXLA from source:

```bash
export XLA_BUILD=true
export XLA_TARGET=rocm
```

**Alternative:** If ROCm compilation fails, use [Ortex](https://github.com/elixir-nx/ortex) (ONNX Runtime) with MIGraphX on Linux to skip the XLA build entirely.

## 3. Bumblebee & memory configuration

EXLA pre-allocates 90% of GPU memory by default. On a unified-memory APU with 96 GB, this freezes the OS.

**Disable pre-allocation:**

```elixir
config :exla, :clients,
  rocm: [platform: :rocm, preallocate: false],
  cuda: [platform: :cuda, preallocate: true, memory_fraction: 0.8]
```

**Lazy model loading** for large LLMs on the 890M:

```elixir
serving = Bumblebee.Text.generation(
  model_info, tokenizer, generation_config,
  defn_options: [compiler: EXLA, lazy_transfers: :always]
)
```

## 4. Using both GPUs simultaneously

CUDA and ROCm EXLA can't coexist in the same BEAM process (C++ linker collisions). Use distributed Erlang:

1. Run a Docker container with CUDA for the RTX 5070
2. Run a separate Docker container with ROCm for the 890M
3. Connect via `Node.connect/1`
4. Route requests with `Nx.Serving.distributed/1`

## References

1. [CUDA Toolkit for NVIDIA Blackwell](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/)
2. [elixir-nx/xla — Pre-compiled XLA extension](https://github.com/elixir-nx/xla)
3. [Increasing VRAM on AMD AI APUs — Jeff Geerling](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux/)
4. [DirectML Execution Provider — ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
5. [ONNX Runtime for Radeon GPUs — AMD ROCm docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html)
6. [EXLA — Hexdocs](https://hexdocs.pm/exla/)
7. [Bumblebee — Hexdocs](https://hexdocs.pm/bumblebee/)
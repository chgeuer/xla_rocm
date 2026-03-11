# APU Configuration

| | |
|---|---|
| **Upstream project** | [Linux kernel](https://kernel.org) (`amdgpu` driver), [ROCm](https://github.com/ROCm), [EXLA](https://github.com/elixir-nx/nx) |
| **Category** | System and application configuration — not code patches, but required for stable operation on APUs |
| **Affects** | AMD APUs with shared memory (Radeon 890M, 8060S, etc.) |

## EXLA BFC allocator: `preallocate: false`

EXLA's BFC (Best-Fit with Coalescing) allocator grabs GPU memory at startup. With the default `preallocate: true`, it claims **90%** of reported GPU memory upfront. On discrete GPUs with dedicated VRAM, this is fast and harmless.

On APUs, **GPU memory is system RAM**. Pre-allocating 90% of a 96 GB machine would grab ~86 GB and freeze the OS.

```elixir
# For APUs — allocate on demand within the BFC ceiling
config :exla, :clients,
  rocm: [platform: :rocm, preallocate: false]

# For discrete GPUs — upfront allocation is fine
config :exla, :clients,
  cuda: [platform: :cuda, preallocate: true, memory_fraction: 0.8]
```

This is an EXLA configuration choice for any APU with shared memory. It is unrelated to the XLA infeed staging fix ([05-infeed-staging.md](05-infeed-staging.md)).

## Kernel parameters

### `amdgpu.noretry=0`

**Default:** `-1` (auto, which resolves to `1` / no-retry on gfx11+)

**What it does:** When the GPU accesses a host memory page that isn't mapped in its address space, the default behaviour on RDNA 3+ is to report a fatal fault. Setting `noretry=0` allows the GPU to retry, giving the kernel's page fault handler time to map the page.

**When it helps:** General stability for GPU compute workloads on APUs. Not sufficient on its own to fix the infeed staging issue (that requires the XLA patch), but reduces the chance of spurious faults during normal operation.

### `ttm.pages_limit` / `amdttm.pages_limit`

**Default:** ~12 million pages (~47 GB on a 96 GB system)

**Recommended:** `20971520` (80 GB)

**What it does:** Sets the maximum number of 4 KB pages the GPU may map from system RAM via the TTM (Translation Table Maps) subsystem. This is a **ceiling**, not a reservation — no memory is pinned until the GPU actually allocates it.

**When it helps:** Large model loading (70B+ parameters). For small embedding models (BERT-base, ~400 MB), the default limit is sufficient.

## Applying at runtime

```bash
echo 0 | sudo tee /sys/module/amdgpu/parameters/noretry
echo 20971520 | sudo tee /sys/module/ttm/parameters/pages_limit
```

A convenience script is provided at `scripts/patch_params.sh` in the [basileus](https://github.com/chgeuer/basileus) project.

## Applying persistently

The method depends on your bootloader. For Limine with UKI (e.g. Omarchy):

```bash
# /etc/default/limine — append to KERNEL_CMDLINE:
KERNEL_CMDLINE[default]+="amdgpu.noretry=0 amdttm.pages_limit=20971520 ttm.pages_limit=20971520"

# Rebuild UKI:
sudo limine-update
```

For GRUB:

```bash
# /etc/default/grub — append to GRUB_CMDLINE_LINUX_DEFAULT:
amdgpu.noretry=0 amdttm.pages_limit=20971520 ttm.pages_limit=20971520

sudo grub-mkconfig -o /boot/grub/grub.cfg
```

## Verify

```bash
cat /sys/module/amdgpu/parameters/noretry    # should be 0
cat /sys/module/ttm/parameters/pages_limit   # should be 20971520
```

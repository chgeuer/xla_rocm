# xla_rocm — Patch Documentation

This directory documents every issue that prevents the upstream XLA/EXLA stack from working on ROCm 7.2 with gfx1150/gfx1151 APUs, and the patches applied by `setup_rocm.sh` to fix them.

## How the pieces fit together

```
┌──────────────────────────────────────────────────────────┐
│  Your Elixir application (e.g. basileus)                 │
│    uses Bumblebee + Nx.Serving                           │
├──────────────────────────────────────────────────────────┤
│  EXLA  (Elixir NIF)                                      │
│    compiles defn → XLA HLO, manages buffers              │
│    ↕ calls into libxla_extension.so                      │
├──────────────────────────────────────────────────────────┤
│  XLA  (C++, built by Bazel from OpenXLA/xla)             │
│    HLO compiler, PjRt runtime, stream executors          │
│    ↕ links against ROCm libraries                        │
├──────────────────────────────────────────────────────────┤
│  ROCm  (HIP runtime, ROCr, SDMA engine)                  │
│    hipMalloc, hipMemcpy, kernel dispatch                 │
│    ↕ talks to kernel driver                              │
├──────────────────────────────────────────────────────────┤
│  Linux kernel  (amdgpu driver, TTM, IOMMU)               │
│    GPU memory management, page fault handling            │
└──────────────────────────────────────────────────────────┘
```

Issues can arise at any layer. The patches are grouped by which layer they fix:

## Patch index

### Build-time (XLA won't compile or recognise the GPU)

| Doc | Issue | Upstream |
|-----|-------|----------|
| [01 — SOVERSION](01-soversion.md) | ROCm 7.x changed `libamdhip64.so` SOVERSION from 6→7 | XLA |
| [02 — GPU architecture](02-gpu-architecture.md) | gfx1150/gfx1151 not in XLA's supported GPU list | XLA |
| [03 — rocprim API](03-rocprim-api.md) | ROCm 7.2 removed `float_bit_mask`, replaced with traits API | XLA (uses ROCm API) |
| [04 — Linker GC](04-linker-gc.md) | ROCm platform registration stripped by linker + missing `alwayslink` | XLA + EXLA |

### Runtime (GPU faults during inference)

| Doc | Issue | Upstream |
|-----|-------|----------|
| [05 — Infeed staging](05-infeed-staging.md) | SDMA engine faults on unpinned host memory in infeed path | XLA |
| [06 — Kernel parameters](06-kernel-parameters.md) | APU-specific `amdgpu` driver settings for shared memory stability | Linux kernel / ROCm |

## Which patches ship in the pre-built archive?

The pre-built archive (`xla_extension-*.tar.gz` on GitHub releases) is a compiled `libxla_extension.so`. It includes patches 01–05 (everything that affects compiled C++ code). Patch 06 is a system configuration, not part of the binary.

| Release | Patches included |
|---------|-----------------|
| v0.9.1-rocm | 01, 02, 03, 04 (missing 05 — infeed staging) |
| v0.9.2-rocm | 01, 02, 03, 04, 05 |

## Upstream contribution path

| Patch | Where to contribute | Complexity |
|-------|-------------------|------------|
| 01 — SOVERSION | [openxla/xla](https://github.com/openxla/xla) — `third_party/gpus/rocm_configure.bzl` | Low — version check |
| 02 — GPU arch | [openxla/xla](https://github.com/openxla/xla) — `device_description.h` | Low — add two strings |
| 03 — rocprim | [openxla/xla](https://github.com/openxla/xla) — `cub_sort_kernel_rocm.cu.cc` | Medium — API migration with backward compat |
| 04 — Linker | [elixir-nx/nx](https://github.com/elixir-nx/nx) — EXLA `extension/BUILD` + docs | Low — one dep change |
| 05 — Infeed | [openxla/xla](https://github.com/openxla/xla) — `infeed_manager.cc` | Low — 6 lines, matches existing eager-path pattern |
| 06 — Kernel params | Documentation only | N/A |

Patch 05 (infeed staging) is the strongest upstream candidate — it's a correctness bug where the infeed path is inconsistent with the eager path, and the fix is minimal.

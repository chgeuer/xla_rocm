# GPU Architecture Support (gfx1150 / gfx1151)

| | |
|---|---|
| **Upstream project** | [OpenXLA/XLA](https://github.com/openxla/xla) |
| **Files** | `xla/stream_executor/device_description.h`, `deps/xla/lib/xla.ex` |
| **Category** | Build + Runtime — XLA won't compile kernels or recognise the GPU |
| **Affects** | Strix Point (gfx1150) and Strix Halo (gfx1151) RDNA 4 APUs |
| **Patch in** | `setup_rocm.sh` phases 6 and 8b |

## Symptom

```
unsupported AMDGPU version: gfx1150
```

The GPU is detected by `rocminfo` but XLA refuses to use it at runtime.

## Root cause

XLA maintains an allowlist of supported AMD GPU architectures in two places:

1. **`device_description.h`** — `kSupportedGfxVersions` array checked at runtime.
2. **`xla.ex`** — `TF_ROCM_AMDGPU_TARGETS` environment variable that controls which GPU ISA targets Bazel compiles kernel code for.

Strix Point (gfx1150, Radeon 890M) and Strix Halo (gfx1151, Radeon 8060S) are RDNA 4 parts released after the XLA version pinned by the `xla` hex package. Neither appears in the upstream allowlist.

## Fix

1. Add `"gfx1150"` and `"gfx1151"` to the `kSupportedGfxVersions` array in `device_description.h`, between `gfx1101` and `gfx1200`.
2. Add `gfx1150,gfx1151` to the `TF_ROCM_AMDGPU_TARGETS` string in `deps/xla/lib/xla.ex`.

## Verify

```bash
# Check your GPU arch:
rocminfo | grep 'Name:.*gfx'
# Should show e.g. gfx1150

# After building with the patch:
GPU_TARGET=rocm mix run -e 'IO.inspect(EXLA.NIF.get_supported_platforms())'
# Should include :rocm
```

## Upstream status

AMD publishes the gfx ISA versions in the [LLVM AMDGPU documentation](https://llvm.org/docs/AMDGPUUsage.html). XLA will add gfx1150/gfx1151 once upstream contributors test on these parts.

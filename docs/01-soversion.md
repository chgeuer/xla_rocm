# SOVERSION Detection for ROCm 7.x

| | |
|---|---|
| **Upstream project** | [OpenXLA/XLA](https://github.com/openxla/xla) |
| **File** | `third_party/gpus/rocm_configure.bzl` |
| **Category** | Build — XLA won't compile without this fix |
| **Affects** | Any ROCm 7.x build (not specific to APUs) |
| **Patch in** | `setup_rocm.sh` phase 8c |

## Symptom

```
Could not load libamdhip64.so.6
```

The XLA build system generates Bazel rules that link against specific `.so.N` versioned libraries. With ROCm 7.x, the generated rules reference `.so.6` (HIP) and `.so.4` (rocblas), but the installed libraries are `.so.7` and `.so.5`.

## Root cause

`rocm_configure.bzl` uses a version-detection function that maps the ROCm version to shared-library SONAMEs. The upstream logic was written for ROCm 5.x/6.x and hardcodes:

- `libamdhip64.so.6`
- `librocblas.so.4`

ROCm 7.0 bumped both SONAMEs.

## Fix

Add a version check: when `rocm_version_number >= 70000`, emit SOVERSION 7 for HIP and 5 for rocblas.

The patch modifies two locations in `rocm_configure.bzl` where the SOVERSION string is selected.

## Verify

```bash
# These files must exist on the build host:
ls /opt/rocm/lib/libamdhip64.so.7
ls /opt/rocm/lib/librocblas.so.5

# After patching, the XLA build should find them.
```

## Upstream status

This will be fixed upstream once XLA officially supports ROCm 7.x. Track the [XLA ROCm configuration](https://github.com/openxla/xla/tree/main/third_party/gpus) for updates.

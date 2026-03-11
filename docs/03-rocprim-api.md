# rocprim API Change in ROCm 7.2

| | |
|---|---|
| **Upstream project** | [OpenXLA/XLA](https://github.com/openxla/xla) (uses [ROCm/rocPRIM](https://github.com/ROCm/rocPRIM) API) |
| **File** | `xla/stream_executor/rocm/cub_sort_kernel_rocm.cu.cc` |
| **Category** | Build — compilation fails without this fix |
| **Affects** | Any ROCm 7.2+ build (not specific to APUs) |
| **Patch in** | `setup_rocm.sh` phase 8d |

## Symptom

```
explicit specialization of undeclared template entity 'float_bit_mask'
```

## Root cause

XLA's GPU sort kernel uses `rocprim::detail::float_bit_mask<T>` to define custom radix sort codecs for Eigen's half-precision types (`Eigen::half`, `Eigen::bfloat16`) and TensorFlow's float8 types.

ROCm 7.2 refactored rocprim internals:

- **Removed:** `rocprim::detail::float_bit_mask<T>`
- **Replaced with:** `rocprim::traits::define<T>` — a traits-based API that bundles bit width, sign bit mask, and exponent/mantissa decomposition into a single struct.

## Fix

Wrap the specialisation code in `#if (TF_ROCM_VERSION >= 70200)` / `#else` blocks:

- **ROCm ≥ 7.2:** Use `rocprim::traits::define<T>` to declare `bit_count`, `sign_bit`, `exponent_bit_count`, and `mantissa_bit_count` for each custom type. Also provide `rocprim::detail::radix_key_codec_base` specialisations using `radix_key_codec_floating`.
- **ROCm < 7.2:** Keep the original `float_bit_mask` specialisations.

This is the largest patch (~90 lines of C++). See `setup_rocm.sh` phase 8d for the full replacement.

## Verify

The build completes past the `cub_sort_kernel_rocm.cu.cc` compilation step (visible in Bazel output).

## Upstream status

This is a breaking API change in rocprim that XLA needs to adapt to. Track [ROCm/rocPRIM releases](https://github.com/ROCm/rocPRIM/releases) and [XLA ROCm issues](https://github.com/openxla/xla/issues?q=rocm).

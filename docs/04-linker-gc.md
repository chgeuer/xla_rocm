# Linker GC and ROCm Platform Registration

| | |
|---|---|
| **Upstream project** | [OpenXLA/XLA](https://github.com/openxla/xla), [elixir-nx/nx](https://github.com/elixir-nx/nx) (EXLA extension BUILD file) |
| **Files** | Build flags (`BUILD_FLAGS`), `deps/xla/extension/BUILD` |
| **Category** | Build — ROCm platform silently missing at runtime |
| **Affects** | Any ROCm build via EXLA (not specific to APUs) |
| **Patch in** | `setup_rocm.sh` phases 6 and 9 |

## Symptom

XLA compiles with `--config=rocm` and no errors, but at runtime:

```elixir
EXLA.NIF.get_supported_platforms()
# => %{host: ...}  — no :rocm key
```

All computation silently falls back to CPU.

## Root cause

Two independent issues conspire to strip the ROCm platform registration:

### Issue A: Linker garbage collection

XLA's ROCm platform registers itself via a C++ static initialiser (`REGISTER_MODULE_INITIALIZER`). Nothing in the EXLA NIF calls this initialiser directly — it runs at shared-library load time as a side effect.

With the default linker flag `--gc-sections`, the linker sees that no code references the registration object and removes it. The `.init_array` entry that would run the registrar at load time is garbage-collected.

### Issue B: Missing `alwayslink` attribute

The EXLA extension's `BUILD` file depends on `//xla/stream_executor:rocm_platform`. This is a thin Bazel target that does **not** carry `alwayslink = True`. Even without `--gc-sections`, the linker may drop unreferenced objects from static libraries.

The target `//xla/stream_executor/rocm:all_runtime` bundles the same code but with `alwayslink = True`, forcing the linker to keep all objects regardless of reference count.

## Fix

Both fixes are required:

1. **Build flags:** Add `--linkopt=-Wl,--no-gc-sections` to `BUILD_FLAGS` (passed to Bazel).
2. **BUILD file:** Change the dependency from `//xla/stream_executor:rocm_platform` to `//xla/stream_executor/rocm:all_runtime`.

## Verify

```elixir
GPU_TARGET=rocm mix run -e '
  platforms = EXLA.NIF.get_supported_platforms()
  IO.inspect(Map.keys(platforms))
'
# => [:host, :rocm]
```

## Upstream status

This is an EXLA packaging issue. The `BUILD` file in `deps/xla/extension/` is maintained by the [elixir-nx/nx](https://github.com/elixir-nx/nx) project. A PR adding the `all_runtime` dependency and documenting the `--no-gc-sections` requirement for ROCm would be the right upstream fix.

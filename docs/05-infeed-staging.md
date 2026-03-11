# Infeed Staging Through Pinned Memory

| | |
|---|---|
| **Upstream project** | [OpenXLA/XLA](https://github.com/openxla/xla) |
| **File** | `xla/service/gpu/infeed_manager.cc` |
| **Category** | Runtime — GPU memory fault during inference |
| **Affects** | ROCm APUs with shared memory (gfx1150, gfx1151); potentially any GPU where SDMA cannot access unpinned host pages |
| **Patch in** | `setup_rocm.sh` phase 8e |

## Symptom

```
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x7f...
Reason: Page not present or supervisor privilege.
```

Occurs when using `lazy_transfers: :always` with Bumblebee/Nx.Serving on a model with many parameters (e.g. BERT-base with ~200 weight tensors). Simple `defn` functions with 1–2 inputs work fine.

## Background: what `lazy_transfers` does

EXLA's `lazy_transfers: :always` option changes how inputs reach the GPU. Instead of transferring all inputs upfront (eager), it compiles XLA HLO with **infeed operations**. The GPU computation blocks at each infeed point, and a separate BEAM process feeds data through XLA's infeed mechanism on demand.

For BERT-base, this creates ~200 infeed operations and ~805 HLO infeed/outfeed instructions, moving ~400 MB of data.

## Root cause: two different H2D copy paths

XLA has two distinct host-to-device copy paths, and only one stages through pinned memory:

### Eager path (safe) — `PjRtStreamExecutorClient::BufferFromHostBuffer`

Used when inputs are passed directly to a computation. This path allocates pinned host memory via `host_memory_allocator()->AllocateRaw()` (which calls `hipHostMalloc`), copies the source data there with `memcpy`, then DMAs from pinned to device.

### Infeed path (broken) — `InfeedManager::CopyBufferToDevice`

Used when data is fed during execution via `TransferToInfeed`. This path calls `stream->Memcpy(device_dst, host_src, size)` **directly from the source pointer** — no staging buffer.

The source pointer comes from BEAM process memory (Erlang NIF binaries allocated with `malloc`). This memory is **not** pinned or registered with the GPU runtime.

## Why it works on CUDA and discrete GPUs

On CUDA, `cuMemcpyHtoDAsync` transparently handles unpinned host memory — the CUDA driver internally stages through a pinned bounce buffer or uses the CPU to assist the DMA.

Discrete AMD GPUs with dedicated VRAM access host memory through the PCIe bus, which may handle page mapping differently.

## Why it fails on gfx1150/gfx1151 APUs

On Strix APUs, the GPU and CPU share the same physical RAM. Host-to-device "copies" use the **SDMA (System DMA) engine**, which reads directly from host virtual addresses. The SDMA engine requires source pages to be GPU-accessible: either pinned via `hipHostMalloc`, registered via `hipHostRegister`, or part of the GPU's GTT mapping.

BEAM heap memory is none of these. The SDMA engine starts reading and **faults partway through** when it encounters a page not mapped in its address space.

### Evidence from verbose ROCm logs (`AMD_LOG_LEVEL=4`)

```
hipMemcpyHtoDAsync(dst=0x7f0840600000, src=0x7f079bab2048, size=9437184)
HSA Copy copy_engine=0x1, src=0x7f079bab2048, size=9437184, engineType=2
Host wait on completion_signal=0x7f0b1cf9be80
Memory Fault on address 0x7f079baf6000
```

The fault address `0x7f079baf6000` is **271 KB into the 9 MB source buffer** (a `f32[768,3072]` weight tensor). The SDMA engine read ~68 pages successfully before hitting an unmapped one.

## What we tried that didn't work

| Approach | Why it failed |
|----------|---------------|
| `amdgpu.noretry=0` kernel parameter | The SDMA fault is below the kernel's page fault handler |
| `HSA_ENABLE_SDMA=0` (disable SDMA) | The shader-based copy path has the same issue |
| `hipHostRegister` on source memory | Silently fails on BEAM heap memory (allocation flags incompatible) |
| Changing EXLA NIF buffer semantics to `kImmutableOnlyDuringCall` | Changes the eager path, but the infeed is a separate code path |

## The fix

Stage through pinned memory in `CopyBufferToDevice`, matching the eager path:

```cpp
// Before (upstream):
se::StreamExecutor* executor = stream->parent();
se::DeviceMemoryHandle buffer(executor, executor->AllocateArray<uint8_t>(size));
TF_RETURN_IF_ERROR(stream->Memcpy(buffer.memory_ptr(), source, size));
return std::move(buffer);

// After (patched):
se::StreamExecutor* executor = stream->parent();
se::DeviceMemoryHandle buffer(executor, executor->AllocateArray<uint8_t>(size));

auto pinned = executor->HostMemoryAllocate(size);
if (pinned.ok()) {
  std::memcpy((*pinned)->opaque(), source, size);
  TF_RETURN_IF_ERROR(
      stream->Memcpy(buffer.memory_ptr(), (*pinned)->opaque(), size));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
} else {
  TF_RETURN_IF_ERROR(stream->Memcpy(buffer.memory_ptr(), source, size));
}

return std::move(buffer);
```

- `HostMemoryAllocate` → `hipHostMalloc` (pinned, DMA-safe)
- `memcpy` → CPU-side copy from BEAM binary into pinned buffer
- `stream->Memcpy` → SDMA from pinned to device (safe)
- `BlockHostUntilDone` → wait for DMA before freeing pinned buffer
- Fallback preserves original behaviour for platforms where direct DMA works

## Verify

```elixir
# This must not crash:
{:ok, m} = Bumblebee.load_model({:hf, "BAAI/bge-base-en-v1.5"})
{:ok, t} = Bumblebee.load_tokenizer({:hf, "BAAI/bge-base-en-v1.5"})
serving = Bumblebee.Text.text_embedding(m, t,
  compile: [batch_size: 1, sequence_length: 512],
  defn_options: [compiler: EXLA, lazy_transfers: :always]
)
%{embedding: tensor} = Nx.Serving.run(serving, "hello world")
IO.puts("#{Nx.size(tensor)} dims")  # => 768 dims
```

## Upstream status

This is a correctness bug in XLA's GPU infeed path. The eager `BufferFromHostBuffer` path already does pinned staging — the infeed path simply missed it. The fix is safe for all platforms (the fallback path preserves existing behaviour) and would be a valid upstream contribution to [OpenXLA/XLA](https://github.com/openxla/xla).

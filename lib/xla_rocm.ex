defmodule XlaRocm do
  @moduledoc """
  Patches and configuration for running Nx + EXLA + Bumblebee on AMD ROCm GPUs.

  This project gets the Elixir ML stack (Nx, EXLA, Bumblebee) running on AMD GPUs
  via ROCm 7.2+, including bleeding-edge architectures like gfx1150 (Radeon 890M)
  and gfx1151 (Radeon 8060S) that aren't yet supported by upstream XLA.

  ## Quick start

      # Install and build (one-time, ~20-40 minutes)
      ./scripts/setup_rocm.sh

      # Run GPU-accelerated tensor operations
      mix run -e 'IO.inspect(Nx.multiply(Nx.tensor([1,2,3]), Nx.tensor([4,5,6])))'

      # Run BERT inference on the GPU
      mix run -e '
        {:ok, model} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"})
        {:ok, tok} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})
        serving = Bumblebee.Text.fill_mask(model, tok)
        IO.inspect(Nx.Serving.run(serving, "The capital of France is [MASK]."))
      '

  ## GPU target selection

  Set the `GPU_TARGET` environment variable to switch backends at runtime:

  - `rocm` (default) — AMD GPU via ROCm
  - `cuda` — NVIDIA GPU via CUDA (requires separate build)
  - `host` — CPU via XLA JIT (fallback)

  ## What gets patched

  The `scripts/setup_rocm.sh` script applies 5 patches to the XLA source:

  1. **SOVERSION fix** — ROCm 7.x ships `libamdhip64.so.7` but XLA looks for `.so.6`
  2. **GPU arch support** — adds `gfx1150`/`gfx1151` to XLA's supported GPU list
  3. **rocprim API** — adapts to the new `rocprim::traits::define<T>` API in ROCm 7.2
  4. **Linker fix** — `--no-gc-sections` prevents stripping the ROCm platform initializer
  5. **Build dep fix** — direct dep on `rocm:all_runtime` to preserve `alwayslink`
  """

  @doc """
  Returns system info about the current EXLA/ROCm setup.

  ## Examples

      iex> info = XlaRocm.info()
      iex> is_map(info)
      true

  """
  def info do
    platforms = EXLA.NIF.get_supported_platforms()

    %{
      platforms: platforms,
      rocm_available: Map.has_key?(platforms, :rocm),
      gpu_devices: Map.get(platforms, :rocm, 0),
      cpu_devices: Map.get(platforms, :host, 0),
      nx_backend: Application.get_env(:nx, :default_backend),
      exla_clients: Application.get_env(:exla, :clients)
    }
  end

  @doc """
  Runs a quick smoke test to verify GPU compute works.

  ## Examples

      XlaRocm.smoke_test()

  """
  def smoke_test do
    IO.puts("Platforms: #{inspect(EXLA.NIF.get_supported_platforms())}")

    t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = Nx.multiply(t, t)
    IO.puts("t² = #{inspect(Nx.to_list(result))}")

    backend = result.__struct__
    IO.puts("Backend: #{inspect(backend)}")
    IO.puts("✅ Working!")
    :ok
  end
end

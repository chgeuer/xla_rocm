import Config

# Override GPU target at runtime via environment variable:
#   GPU_TARGET=cuda  — use NVIDIA RTX 5070
#   GPU_TARGET=rocm  — use AMD Radeon 890M
#   GPU_TARGET=host  — use CPU via XLA JIT
gpu_target =
  System.get_env("GPU_TARGET", "rocm")
  |> String.downcase()
  |> String.to_atom()

# Validate the requested platform is compiled into EXLA.
# Fallback order: requested → other GPU → host.
# This prevents silent misrouting and still prefers GPU over CPU.
effective_target =
  case gpu_target do
    :host ->
      :host

    platform ->
      try do
        platforms = EXLA.NIF.get_supported_platforms()

        cond do
          Map.has_key?(platforms, platform) ->
            platform

          # Requested :cuda but only :rocm available (or vice versa) — use the available GPU
          Map.has_key?(platforms, :rocm) ->
            IO.puts(
              "[xla_rocm] ⚠️  GPU_TARGET=#{platform} not available, falling back to :rocm"
            )

            :rocm

          Map.has_key?(platforms, :cuda) ->
            IO.puts(
              "[xla_rocm] ⚠️  GPU_TARGET=#{platform} not available, falling back to :cuda"
            )

            :cuda

          true ->
            IO.puts(
              "[xla_rocm] ⚠️  GPU_TARGET=#{platform} not available, no GPU found. Using :host"
            )

            :host
        end
      rescue
        # EXLA NIF not loaded yet during initial compilation
        _ -> platform
      end
  end

case effective_target do
  :cuda ->
    config :exla, :clients,
      host: [platform: :host],
      cuda: [platform: :cuda, preallocate: true, memory_fraction: 0.8]

    config :nx, :default_backend, {EXLA.Backend, client: :cuda}
    config :nx, :default_defn_options, compiler: EXLA, client: :cuda

  :rocm ->
    config :exla, :clients,
      host: [platform: :host],
      rocm: [platform: :rocm, preallocate: false]

    config :nx, :default_backend, {EXLA.Backend, client: :rocm}
    config :nx, :default_defn_options, compiler: EXLA, client: :rocm

  _host ->
    config :exla, :clients,
      host: [platform: :host]

    config :nx, :default_backend, {EXLA.Backend, client: :host}
    config :nx, :default_defn_options, compiler: EXLA, client: :host
end

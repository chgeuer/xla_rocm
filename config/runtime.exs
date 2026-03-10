import Config

# Override GPU target at runtime via environment variable:
#   GPU_TARGET=cuda  — use NVIDIA RTX 5070
#   GPU_TARGET=rocm  — use AMD Radeon 890M
#   GPU_TARGET=host  — use CPU via XLA JIT (default)
gpu_target =
  System.get_env("GPU_TARGET", "rocm")
  |> String.downcase()
  |> String.to_atom()

case gpu_target do
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

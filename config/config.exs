import Config

# Nx defaults — use EXLA as the backend on ROCm GPU
config :nx, :default_backend, {EXLA.Backend, client: :rocm}

# EXLA client configuration
# ROCm: preallocate: false is CRITICAL for the Radeon 890M since it shares
# system RAM. Without this, EXLA will try to grab 90% of 96GB and freeze the OS.
config :exla, :clients,
  host: [platform: :host],
  rocm: [platform: :rocm, preallocate: false]

config :nx, :default_defn_options, compiler: EXLA, client: :rocm

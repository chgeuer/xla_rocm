# xla_rocm — Nx + EXLA + Bumblebee on AMD ROCm GPUs

# Default: show available recipes
default:
    @just --list

# One-time setup: install packages, patch XLA, build from source (~20-40 min)
setup:
    ./scripts/setup_rocm.sh

# Quick smoke test: verify GPU is detected and tensor ops work
test:
    mix run -e 'XlaRocm.smoke_test()'

# Show detected platforms and GPU info
info:
    mix run -e 'IO.inspect(XlaRocm.info(), pretty: true)'

# Interactive Elixir shell with EXLA loaded
iex:
    iex -S mix

# Run BERT fill-mask inference on the GPU
bert prompt="The capital of France is [MASK].":
    mix run -e ' \
      {:ok, model} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"}) ; \
      {:ok, tok} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"}) ; \
      serving = Bumblebee.Text.fill_mask(model, tok) ; \
      result = Nx.Serving.run(serving, "{{ prompt }}") ; \
      for p <- result.predictions, do: IO.puts("  #{p.token} (#{Float.round(p.score * 100, 1)}%)") \
    '

# Run with CPU backend (no GPU)
test-cpu:
    GPU_TARGET=host mix run -e 'XlaRocm.smoke_test()'

# Run project tests
test-unit:
    mix test

# Compile the project
build:
    mix compile

# Fetch dependencies
deps:
    mix deps.get

# Clean build artifacts
clean:
    mix deps.clean exla --build
    mix clean

# Full clean including XLA build cache
clean-all:
    mix deps.clean exla --build
    mix deps.clean xla --build
    mix clean
    rm -rf _build

# Show ROCm GPU info from the system
rocm-info:
    @echo "=== ROCm version ===" && cat /opt/rocm/.info/version
    @echo "=== GPU ===" && rocminfo 2>/dev/null | grep -A3 'Agent [0-9]' | grep -E 'Name|Marketing' | head -4
    @echo "=== HIP ===" && /opt/rocm/bin/hipconfig 2>/dev/null | head -3

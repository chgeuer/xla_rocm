# xla_rocm — Nx + EXLA + Bumblebee on AMD ROCm & NVIDIA CUDA GPUs

# Default: show available recipes
default:
    @just --list

# --- ROCm ---

# One-time ROCm setup: install packages, patch XLA, build from source (~20-40 min)
setup-rocm:
    ./scripts/setup_rocm.sh

# Quick smoke test: verify GPU is detected and tensor ops work
test-rocm:
    mix run -e 'XlaRocm.smoke_test()'

# --- General ---

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

# --- CUDA ---

# One-time setup: install CUDA packages, build XLA from source (~20-40 min)
setup-cuda:
    ./scripts/setup_cuda.sh

# Quick smoke test with CUDA backend
test-cuda:
    GPU_TARGET=cuda mix run -e 'XlaRocm.smoke_test()'

# Run BERT fill-mask on NVIDIA GPU
bert-cuda prompt="The capital of France is [MASK].":
    GPU_TARGET=cuda mix run -e ' \
      {:ok, model} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"}) ; \
      {:ok, tok} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"}) ; \
      serving = Bumblebee.Text.fill_mask(model, tok) ; \
      result = Nx.Serving.run(serving, "{{ prompt }}") ; \
      for p <- result.predictions, do: IO.puts("  #{p.token} (#{Float.round(p.score * 100, 1)}%)") \
    '

# Show NVIDIA GPU info from the system
cuda-info:
    @echo "=== NVIDIA GPU ===" && nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>/dev/null || echo "nvidia-smi not found"
    @echo "=== CUDA ===" && nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found"

# --- Dual-GPU Clustering ---

# Start a named ROCm node (for dual-GPU clustering)
start-rocm:
    GPU_TARGET=rocm elixir --sname rocm -S mix run --no-halt

# Start a named CUDA node (for dual-GPU clustering)
start-cuda:
    GPU_TARGET=cuda elixir --sname cuda -S mix run --no-halt

# Connect to both GPU nodes and show cluster info
cluster:
    elixir --sname client -S mix run -e ' \
      XlaRocm.Cluster.connect(:rocm) ; \
      XlaRocm.Cluster.connect(:cuda) ; \
      IO.inspect(XlaRocm.Cluster.gpu_info(), pretty: true) \
    '

# --- Packaging ---

# Package built XLA archives for distribution (creates dist/)
package target="all":
    ./scripts/package.sh {{ target }}

# Publish packaged archives as a GitHub release
release tag:
    gh release create {{ tag }} --title "XLA with ROCm/CUDA patches ({{ tag }})" dist/*

# --- System Info ---

# Show ROCm GPU info from the system
rocm-info:
    @echo "=== ROCm version ===" && cat /opt/rocm/.info/version
    @echo "=== GPU ===" && rocminfo 2>/dev/null | grep -A3 'Agent [0-9]' | grep -E 'Name|Marketing' | head -4
    @echo "=== HIP ===" && /opt/rocm/bin/hipconfig 2>/dev/null | head -3

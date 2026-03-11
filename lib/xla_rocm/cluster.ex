defmodule XlaRocm.Cluster do
  @moduledoc """
  Distributed Erlang clustering for dual-GPU setups.

  CUDA and ROCm EXLA cannot coexist in the same BEAM process (C++ linker
  collisions). This module connects separate GPU nodes via `--sname` so you
  can dispatch work to either GPU from any node in the cluster.

  ## Quick start

      # Terminal 1 — ROCm node
      GPU_TARGET=rocm elixir --sname rocm -S mix run --no-halt

      # Terminal 2 — CUDA node
      GPU_TARGET=cuda elixir --sname cuda -S mix run --no-halt

      # Terminal 3 — client
      elixir --sname client -S mix run -e '
        XlaRocm.Cluster.connect(:rocm)
        XlaRocm.Cluster.connect(:cuda)
        IO.inspect(XlaRocm.Cluster.gpu_info())
      '

  Or use the justfile recipes: `just start-rocm`, `just start-cuda`, `just cluster`.
  """

  @doc """
  Connects to a named node on the same machine.

  Accepts a short name atom (e.g. `:rocm`) and connects to `rocm@hostname`.

  ## Examples

      iex> XlaRocm.Cluster.connect(:rocm)
      true

  """
  def connect(short_name) when is_atom(short_name) do
    hostname = node_hostname()
    full_name = :"#{short_name}@#{hostname}"
    Node.connect(full_name)
  end

  @doc """
  Lists all connected nodes.

  ## Examples

      iex> XlaRocm.Cluster.nodes()
      [:rocm@myhost, :cuda@myhost]

  """
  def nodes do
    Node.list()
  end

  @doc """
  Gathers GPU info from all connected nodes (and the local node).

  Returns a map of `%{node_name => info}` where `info` is the result
  of calling `XlaRocm.info/0` on each node.

  ## Examples

      iex> XlaRocm.Cluster.gpu_info()
      %{
        rocm@myhost: %{rocm_available: true, ...},
        cuda@myhost: %{cuda_available: true, ...}
      }

  """
  def gpu_info do
    all_nodes = [node() | Node.list()]

    all_nodes
    |> Enum.map(fn n ->
      Task.async(fn -> {n, :rpc.call(n, XlaRocm, :info, [])} end)
    end)
    |> Task.await_many(5_000)
    |> Map.new()
  end

  @doc """
  Runs a function on a specific remote node.

  ## Examples

      XlaRocm.Cluster.run_on(:cuda, fn ->
        Nx.multiply(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
      end)

  """
  def run_on(short_name, fun) when is_atom(short_name) and is_function(fun, 0) do
    hostname = node_hostname()
    full_name = :"#{short_name}@#{hostname}"
    :rpc.call(full_name, :erlang, :apply, [fun, []])
  end

  @doc """
  Returns the hostname portion of the current node name.
  """
  def node_hostname do
    node()
    |> Atom.to_string()
    |> String.split("@")
    |> List.last()
  end
end

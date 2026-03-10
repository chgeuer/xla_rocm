defmodule XlaRocm.MixProject do
  use Mix.Project

  def project do
    [
      app: :xla_rocm,
      version: "0.1.0",
      elixir: "~> 1.20-rc",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.9"},
      {:exla, "~> 0.9"},
      {:bumblebee, "~> 0.6"}
    ]
  end
end

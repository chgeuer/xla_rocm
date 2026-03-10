defmodule XlaRocmTest do
  use ExUnit.Case
  doctest XlaRocm

  test "info returns platform map" do
    info = XlaRocm.info()
    assert is_map(info.platforms)
    assert info.cpu_devices > 0
  end
end

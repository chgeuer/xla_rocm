defmodule XlaRocmTest do
  use ExUnit.Case
  doctest XlaRocm

  test "info returns platform map" do
    info = XlaRocm.info()
    assert is_map(info.platforms)
    assert info.cpu_devices > 0
    assert is_boolean(info.rocm_available)
    assert is_boolean(info.cuda_available)
  end
end

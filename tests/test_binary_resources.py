from candle_dvm.device_bin import load_c220_binary

def test_c220_binary_is_packaged_and_nonempty():
    data = load_c220_binary()
    assert isinstance(data, bytes)
    assert len(data) > 0

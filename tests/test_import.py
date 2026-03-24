def test_imports_package():
    import candle_dvm
    assert hasattr(candle_dvm, "__file__")

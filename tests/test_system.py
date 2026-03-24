import pytest


@pytest.mark.requires_910b
def test_system_initializes_and_loads_c220_binary():
    from candle_dvm.system import System

    sys = System()
    sys.init(0)
    assert sys.arch_name() == "c220"
    assert sys.has_vector_handle()


@pytest.mark.requires_910b
def test_get_system_singleton():
    from candle_dvm.system import get_system

    s1 = get_system()
    s2 = get_system()
    assert s1 is s2
    assert s1.arch_name() == "c220"


@pytest.mark.requires_910b
def test_device_memory_round_trip():
    from candle_dvm.system import System

    sys = System()
    sys.init(0)
    stream = sys.create_stream()
    try:
        size = 1024
        dev_ptr = sys.malloc_device(size)
        try:
            # Write pattern to host buffer, copy to device, read back
            import ctypes
            host_buf = (ctypes.c_ubyte * size)(*([0xAB] * size))
            host_out = (ctypes.c_ubyte * size)()
            sys.memcpy_h2d(dev_ptr, ctypes.addressof(host_buf), size, stream)
            sys.sync_stream(stream)
            sys.memcpy_d2h(ctypes.addressof(host_out), dev_ptr, size, stream)
            sys.sync_stream(stream)
            assert bytes(host_out) == bytes(host_buf)
        finally:
            sys.free_device(dev_ptr)
    finally:
        sys.destroy_stream(stream)

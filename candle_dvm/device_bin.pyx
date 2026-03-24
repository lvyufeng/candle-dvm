from importlib import resources

def load_c220_binary() -> bytes:
    return resources.files("candle_dvm.data").joinpath("g_vkernel_c220.bin").read_bytes()

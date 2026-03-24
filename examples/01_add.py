#!/usr/bin/env python3
"""Minimal end-to-end example: element-wise add on 910B hardware.

Usage::

    python examples/01_add.py
"""

import numpy as np
import candle_dvm as dvm


@dvm.kernel()
def my_add(k, x, y):
    a = k.load(x.shape, dvm.float32)
    b = k.load(y.shape, dvm.float32)
    return k.store(k.add(a, b))


def main():
    x = np.full([32, 32], 0.1, np.float32)
    y = np.full([32, 32], 0.3, np.float32)
    z = my_add(x, y)

    expected = x + y
    print("Result sample (first 4 elements):", z.ravel()[:4])
    print("Expected sample (first 4 elements):", expected.ravel()[:4])
    print("Match:", np.allclose(z, expected, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    main()

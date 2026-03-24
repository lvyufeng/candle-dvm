# cython: language_level=3
"""Cython declaration file for candle_dvm.isa encode helpers."""

# ---------------------------------------------------------------------------
# Encode helpers  (cpdef -- accessible from both Cython and Python)
# ---------------------------------------------------------------------------
cpdef unsigned long long make_acc_head(
    unsigned long long opcode,
    unsigned long long ext,
    unsigned long long size,
)

cpdef unsigned long long make_simd_head(
    unsigned long long opcode,
    unsigned long long ext,
    unsigned long long size,
)

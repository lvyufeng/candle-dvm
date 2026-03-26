# cython: language_level=3
"""Cython declaration file for candle_dvm.isa encode helpers.

Module-level Python constants (access via ``from candle_dvm.isa import ...``):
    BINS_ADD, BINS_MUL, BINS_DIV, BINS_MAX, BINS_MIN
        BinarySOpType indices for scalar binary ops (Batch C).
    BINARY_SCALAR_OPCODE_TABLE : dict[(int, int), int]
        Maps ``(BinarySOpType, DataType)`` to ``vSimdInsnID``.
"""

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

cpdef list encode_unary(
    unsigned long long opcode,
    unsigned long long xd,
    unsigned long long xn,
    unsigned long long count,
)

cpdef list encode_binary_scalar(
    unsigned long long opcode,
    unsigned long long xn,
    unsigned long long xd,
    unsigned long long count,
    unsigned long long scalar_bits,
)

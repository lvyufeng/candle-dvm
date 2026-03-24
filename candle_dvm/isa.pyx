# cython: language_level=3
"""DVM ISA constants and instruction-head encode helpers.

Constants are derived from the upstream ``isa.h`` header.  The two helper
functions ``make_acc_head`` and ``make_simd_head`` correspond to the C
``vMakeAccHead`` / ``vMakeSimdHead`` macros but **without** the
``g_system.*_func_offset_`` lookup tables -- they use the raw opcode ID
directly.  The offset tables will be wired in a later phase.
"""

# ===================================================================
# Access opcode constants  (vAccInsnID)
# ===================================================================
V_LOAD                = 0
V_LOAD_DUMMY          = 1
V_LOAD_VIEW           = 2
V_SLOAD               = 3
V_LOAD_CC             = 4
V_MULTI_LOAD          = 5
V_PINGPONG_LOAD       = 6
V_PINGPONG_PEER_LOAD  = 7
V_PEER_LOAD           = 8
V_PEER_LOAD_MIX       = 9
V_STORE               = 10
V_STORE_ATOMIC        = 11
V_STORE_COND          = 12
V_SSTORE              = 13
V_SLICE_STORE         = 14
V_STORE_AG            = 15
V_STORE_RS            = 16
V_PEER_STORE          = 17
V_PEER_STORE_MIX      = 18
V_ACCESS_NONE         = 19

# ===================================================================
# SIMD opcode constants  (vSimdInsnID)
# ===================================================================
V_COPY                = 0
V_COPY_CUBE_TILE      = 1
V_NOP                 = 2
V_BROADCAST_Y         = 3
V_BROADCAST_S         = 4
V_SQRT                = 5
V_ABS                 = 6
V_LOG                 = 7
V_EXP                 = 8
V_ROUND               = 9
V_FLOOR               = 10
V_CEIL                = 11
V_TRUNC               = 12
V_ADDS                = 13
V_MULS                = 14
V_DIVS                = 15
V_SDIV                = 16
V_CMPS                = 17
V_ADD                 = 18
V_SUB                 = 19
V_MUL                 = 20
V_DIV                 = 21
V_MIN                 = 22
V_MAX                 = 23
V_CMP                 = 24
V_CAST_FP32_TO_FP16   = 25
V_CAST_FP32_TO_INT32  = 26
V_RSUM_X              = 27
V_RSUM_Y              = 28
V_RMAX_X              = 29
V_RMAX_Y              = 30
V_RMIN_X              = 31
V_RMIN_Y              = 32
V_RSUM_JOIN           = 33
V_SEL                 = 34
V_POW                 = 35
V_CLR_PAD             = 36
V_ELEMENT_ANY         = 37
V_ONE_HOT             = 38
V_BROADCAST_X_B16     = 39
V_BROADCAST_S_B16     = 40
V_SQRT_FP16           = 41
V_ABS_FP16            = 42
V_LOG_FP16            = 43
V_EXP_FP16            = 44
V_ADDS_FP16           = 45
V_MULS_FP16           = 46
V_DIVS_FP16           = 47
V_SDIV_FP16           = 48
V_CMPS_FP16           = 49
V_ADD_FP16            = 50
V_SUB_FP16            = 51
V_MUL_FP16            = 52
V_DIV_FP16            = 53
V_MIN_FP16            = 54
V_MAX_FP16            = 55
V_CAST_FP16_TO_BOOL   = 56
V_CAST_FP16_TO_FP32   = 57
V_CAST_FP16_TO_INT32  = 58
V_RMAX_X_FP16         = 59
V_RMAX_Y_FP16         = 60
V_RMIN_X_FP16         = 61
V_RMIN_Y_FP16         = 62
V_CLR_PAD_B16         = 63
V_CMP_FP16            = 64
V_SEL_FP16            = 65
V_ISFINITE_FP16       = 66
V_ONE_HOT_B16         = 67
V_CAST_BOOL_TO_FP16   = 68
V_BROADCAST_X_B32     = 69
V_ADD_INT32           = 70
V_SUB_INT32           = 71
V_MUL_INT32           = 72
V_MIN_INT32           = 73
V_MAX_INT32           = 74
V_SEL_INT32           = 75
V_CAST_INT32_TO_FP16  = 76
V_CAST_INT32_TO_FP32  = 77
V_ISFINITE            = 78
V_MAXS                = 79
V_MINS                = 80
V_MAXS_FP16           = 81
V_MINS_FP16           = 82
V_ADDS_INT32          = 83
V_MULS_INT32          = 84
V_MAXS_INT32          = 85
V_MINS_INT32          = 86
V_REMOVEPAD_U16       = 87
V_REMOVEPAD           = 88
V_ATOMICCUM           = 89
V_ATOMICCUM_FP16      = 90
V_CAST_FP32_TO_BF16   = 91
V_CAST_BF16_TO_FP32   = 92
V_CAST_BF16_TO_INT32  = 93
V_CMP_INT32           = 94
V_ABS_INT32           = 95
V_CMPS_INT32          = 96
V_ADDS_BF16           = 97
V_MULS_BF16           = 98
V_MAXS_BF16           = 99
V_MINS_BF16           = 100
V_CMP_BF16            = 101
V_CMPS_BF16           = 102
V_ADD_BF16            = 103
V_SUB_BF16            = 104
V_MUL_BF16            = 105
V_MIN_BF16            = 106
V_MAX_BF16            = 107
V_ISFINITE_BF16       = 108
V_SEL_BF16            = 109
V_NONE                = 110

# ===================================================================
# Bitfield offset / mask constants
# ===================================================================

# -- common area --
V_HEAD_SIMD_FLAG_OFFSET    = 0
V_HEAD_ID_OFFSET           = 48
V_HEAD_ID_MASK             = 0xFFFF
V_HEAD_EVENT_MASK          = 0x7

# -- simd head --
V_HEAD_BAR_FLAG_OFFSET     = 2
V_HEAD_SET_FLAG_OFFSET     = 3
V_HEAD_WAIT_FLAG_OFFSET    = 4
V_HEAD_BACK_SET_OFFSET     = 5
V_HEAD_BACK_WAIT_OFFSET    = 6
V_HEAD_SIZE_OFFSET         = 7
V_HEAD_SET_EVENT_OFFSET    = 10
V_HEAD_WAIT_EVENT_OFFSET   = 13
V_HEAD_B_SET_EVENT_OFFSET  = 16
V_HEAD_B_WAIT_EVENT_OFFSET = 19
V_HEAD_EXT_OFFSET          = 22
V_HEAD_EXT_MASK            = 0x3FFFFFF
V_HEAD_SIZE_MASK           = 0x7

# -- load/store head --
V_M_HEAD_SET_FLAG_OFFSET   = 2
V_M_HEAD_WAIT_FLAG_OFFSET  = 3
V_M_HEAD_SIZE_OFFSET       = 4
V_M_HEAD_SET_EVENT_OFFSET  = 8
V_M_HEAD_WAIT_EVENT_OFFSET = 11
V_M_HEAD_EXT_OFFSET        = 14
V_M_HEAD_EXT_MASK          = 0x3FFFFFFFF
V_M_HEAD_SIZE_MASK         = 0xF

# -- common masks --
V_X_BITS                   = 18
V_C_X_BITS                 = 13
V_X_MASK                   = 0x3FFFF
V_RS_MASK                  = 0xF

# -- entry constants --
V_ENTRY_TYPE_V             = 0
V_ENTRY_TYPE_VE            = 1
V_ENTRY_TYPE_C             = 2
V_ENTRY_TYPE_P             = 3
V_ENTRY_MASK_TYPE          = 7
V_ENTRY_FLAG_CUBE_MIX      = 8
V_ENTRY_FLAG_EXTERN_CODE   = 16
V_ENTRY_CODE_SIZE_OFFSET   = 8
V_ENTRY_CODE_SIZE_BITS     = 12
V_ENTRY_V_TILE_TAIL_OFFSET = 32
V_ENTRY_V_TILE_TAIL_BITS   = 8
V_ENTRY_V_TILE_BODY_OFFSET = 40
V_ENTRY_V_TILE_BODY_BITS   = 24


# ===================================================================
# Function offset tables for g_vkernel_c220.bin
# Maps raw opcode enum values to code addresses within the binary.
# The ID field in the instruction head contains the actual function
# address, NOT an index -- it is used by the VM to jump directly
# to the handler code for each instruction.
# Source: dvm/vm.cc g_access_func_offset_c220[] / g_simd_func_offset_c220[]
# ===================================================================

# Access function offset table: raw_opcode -> code address in binary
ACCESS_FUNC_OFFSET = [
    0x5954,  # V_LOAD = 0
    0x5b18,  # V_LOAD_DUMMY = 1
    0x5b60,  # V_LOAD_VIEW = 2
    0x5ea8,  # V_SLOAD = 3
    0x0000,  # V_LOAD_CC = 4 (not in c220)
    0x61ec,  # V_MULTI_LOAD = 5
    0x6554,  # V_PINGPONG_LOAD = 6
    0x67a0,  # V_PINGPONG_PEER_LOAD = 7
    0x6c28,  # V_PEER_LOAD = 8
    0x6ecc,  # V_PEER_LOAD_MIX = 9
    0x7150,  # V_STORE = 10
    0x7334,  # V_STORE_ATOMIC = 11
    0x7634,  # V_STORE_COND = 12
    0x77f0,  # V_SSTORE = 13
    0x7bac,  # V_SLICE_STORE = 14
    0x7de0,  # V_STORE_AG = 15
    0x7f18,  # V_STORE_RS = 16
    0x8080,  # V_PEER_STORE = 17
    0x81c4,  # V_PEER_STORE_MIX = 18
    0x0000,  # V_ACCESS_NONE = 19
]

# SIMD function offset table: raw_opcode -> code address in binary
SIMD_FUNC_OFFSET = [
    0x831c,  # V_COPY = 0
    0x0000,  # V_COPY_CUBE_TILE = 1 (not in c220)
    0x8350,  # V_NOP = 2
    0x85fc,  # V_BROADCAST_Y = 3
    0x870c,  # V_BROADCAST_S = 4
    0x874c,  # V_SQRT = 5
    0x878c,  # V_ABS = 6
    0x87cc,  # V_LOG = 7
    0x880c,  # V_EXP = 8
    0x884c,  # V_ROUND = 9
    0x888c,  # V_FLOOR = 10
    0x88cc,  # V_CEIL = 11
    0x890c,  # V_TRUNC = 12
    0x894c,  # V_ADDS = 13
    0x8998,  # V_MULS = 14
    0xd7ec,  # V_DIVS = 15
    0xd700,  # V_SDIV = 16
    0x8f54,  # V_CMPS = 17
    0x89e4,  # V_ADD = 18
    0x8a30,  # V_SUB = 19
    0x8a7c,  # V_MUL = 20
    0x8ac8,  # V_DIV = 21
    0x8b14,  # V_MIN = 22
    0x8b60,  # V_MAX = 23
    0x8bac,  # V_CMP = 24
]


# ===================================================================
# Encode helpers
# ===================================================================

cpdef unsigned long long make_acc_head(
    unsigned long long opcode,
    unsigned long long ext,
    unsigned long long size,
):
    """Build a load/store instruction head word.

    Matches ``vMakeAccHead`` from isa.h.  Applies the
    ``ACCESS_FUNC_OFFSET`` lookup to map the raw opcode enum value
    to the binary dispatch table index used by g_vkernel_c220.bin.

    Parameters
    ----------
    opcode : int
        Access opcode (e.g. ``V_LOAD``).
    ext : int
        Extension field (34 bits).
    size : int
        Instruction length field (4 bits).

    Returns
    -------
    int
        Packed 64-bit head word.
    """
    cdef unsigned long long hw_id = <unsigned long long>ACCESS_FUNC_OFFSET[opcode]
    return (
        (ext << V_M_HEAD_EXT_OFFSET)
        | (size << V_M_HEAD_SIZE_OFFSET)
        | (hw_id << V_HEAD_ID_OFFSET)
    )


cpdef unsigned long long make_simd_head(
    unsigned long long opcode,
    unsigned long long ext,
    unsigned long long size,
):
    """Build a SIMD instruction head word.

    Matches ``vMakeSimdHead`` from isa.h.  Applies the
    ``SIMD_FUNC_OFFSET`` lookup to map the raw opcode enum value
    to the binary dispatch table index used by g_vkernel_c220.bin.

    Parameters
    ----------
    opcode : int
        SIMD opcode (e.g. ``V_ADD``).
    ext : int
        Extension field (26 bits).
    size : int
        Instruction length field (3 bits).

    Returns
    -------
    int
        Packed 64-bit head word.
    """
    cdef unsigned long long hw_id = <unsigned long long>SIMD_FUNC_OFFSET[opcode]
    return (
        (ext << V_HEAD_EXT_OFFSET)
        | (size << V_HEAD_SIZE_OFFSET)
        | (hw_id << V_HEAD_ID_OFFSET)
        | (1 << V_HEAD_SIMD_FLAG_OFFSET)
    )

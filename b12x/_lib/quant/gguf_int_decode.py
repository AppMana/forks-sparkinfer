"""Exact Q8_0/Q6_K register conversion for BF16 tensor-core fragments.

FP16 block bases and integer subscales are multiplied in FP32; round only
once, at the final BF16 value. No FP8 surrogate or rescaled codebook is used.
"""
from cutlass import Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def q8_0_pair_to_bf16(packed, base, *, loc=None, ip=None):
    """Decode two adjacent signed bytes and the original FP16 block base."""
    result = llvm.inline_asm(
        T.i32(),
        [Uint32(packed).ir_value(loc=loc, ip=ip), Uint32(base).ir_value(loc=loc, ip=ip)],
        """{
        .reg .b16 dh;
        .reg .s32 q0, q1;
        .reg .f32 d, f0, f1;
        cvt.u16.u32 dh, $2;
        cvt.f32.f16 d, dh;
        bfe.s32 q0, $1, 0, 8;
        bfe.s32 q1, $1, 8, 8;
        cvt.rn.f32.s32 f0, q0;
        cvt.rn.f32.s32 f1, q1;
        mul.f32 f0, f0, d;
        mul.f32 f1, f1, d;
        cvt.rn.bf16x2.f32 $0, f1, f0;
        }""",
        "=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def q6_k_pair_to_bf16(low, high, base, subscale, low_shift, high_shift, *, loc=None, ip=None):
    """Two adjacent Q6_K values, retaining raw low/high byte planes.

    low/high each contain two adjacent bytes. low_shift selects a nibble
    (0 or 4); high_shift selects a two-bit field (0,2,4,6). Both values share
    an original signed K16 subscale and FP16 K256 base.
    """
    result = llvm.inline_asm(
        T.i32(),
        [Uint32(x).ir_value(loc=loc, ip=ip)
         for x in (low, high, base, subscale, low_shift, high_shift)],
        """{
        .reg .b16 dh;
        .reg .u32 lo0, lo1, hi0, hi1, shift;
        .reg .s32 q0, q1, scale;
        .reg .f32 d, ds, f0, f1;
        cvt.u16.u32 dh, $3;
        cvt.f32.f16 d, dh;
        bfe.s32 scale, $4, 0, 8;
        cvt.rn.f32.s32 ds, scale;
        mul.f32 d, d, ds;
        bfe.u32 lo0, $1, $5, 4;
        add.u32 shift, $5, 8;
        bfe.u32 lo1, $1, shift, 4;
        bfe.u32 hi0, $2, $6, 2;
        add.u32 shift, $6, 8;
        bfe.u32 hi1, $2, shift, 2;
        shl.b32 hi0, hi0, 4;
        shl.b32 hi1, hi1, 4;
        or.b32 q0, lo0, hi0;
        or.b32 q1, lo1, hi1;
        sub.s32 q0, q0, 32;
        sub.s32 q1, q1, 32;
        cvt.rn.f32.s32 f0, q0;
        cvt.rn.f32.s32 f1, q1;
        mul.f32 f0, f0, d;
        mul.f32 f1, f1, d;
        cvt.rn.bf16x2.f32 $0, f1, f0;
        }""",
        "=r,r,r,r,r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def q5_k_pair_to_bf16(quants, bases, scale, minimum, *, loc=None, ip=None):
    """Exact GGUF Q5_K multiply then subtract, without fused rounding."""
    result = llvm.inline_asm(
        T.i32(),
        [Uint32(x).ir_value(loc=loc, ip=ip) for x in (quants, bases, scale, minimum)],
        """{
        .reg .b16 dh, mh;
        .reg .u32 q0, q1, upper;
        .reg .f32 d, m, sc, mn, f0, f1;
        cvt.u16.u32 dh, $2;
        shr.u32 upper, $2, 16;
        cvt.u16.u32 mh, upper;
        cvt.f32.f16 d, dh;
        cvt.f32.f16 m, mh;
        cvt.rn.f32.u32 sc, $3;
        cvt.rn.f32.u32 mn, $4;
        mul.f32 d, d, sc;
        mul.f32 m, m, mn;
        bfe.u32 q0, $1, 0, 8;
        bfe.u32 q1, $1, 8, 8;
        cvt.rn.f32.u32 f0, q0;
        cvt.rn.f32.u32 f1, q1;
        mul.f32 f0, f0, d;
        mul.f32 f1, f1, d;
        sub.f32 f0, f0, m;
        sub.f32 f1, f1, m;
        cvt.rn.bf16x2.f32 $0, f1, f0;
        }""",
        "=r,r,r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Uint32(result)

"""Uncompensated E8M0 scales for native BF16 E2M1 register conversion."""

from cutlass import Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def e8m0x4_to_bfloat2x2(packed: Uint32, *, loc=None, ip=None):
    """Return scale pairs (s0,s2), (s1,s3), including E8M0 code zero.

    E8M0 byte zero is 2**-127, a representable BF16 subnormal. Normal
    bytes map directly to the BF16 exponent. Preparation rejects byte 255.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u32 b0, b1, b2, b3, h0, h1, h2, h3, tmp;
            .reg .pred p;
            and.b32 b0, $2, 255;
            shr.u32 b1, $2, 8;
            and.b32 b1, b1, 255;
            shr.u32 b2, $2, 16;
            and.b32 b2, b2, 255;
            shr.u32 b3, $2, 24;
        """
        + "\n".join(
            f"shl.b32 h{i}, b{i}, 7;\nsetp.eq.u32 p, b{i}, 0;\nselp.b32 h{i}, 64, h{i}, p;"
            for i in range(4)
        )
        + """
            shl.b32 tmp, h2, 16;
            or.b32 $0, h0, tmp;
            shl.b32 tmp, h3, 16;
            or.b32 $1, h1, tmp;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )

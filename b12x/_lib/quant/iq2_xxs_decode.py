"""IQ2_XXS register decoder shared by SIMT and tensor-core fragment loaders."""

from cutlass import Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def decode_iq2_xxs_eight(
    indices,
    base,
    signs_scale,
    group,
    table_addr,
    *,
    shared_lut=False,
    loc=None,
    ip=None,
):
    """Decode eight consecutive values, rounding once after the exact scale.

    indices: one 8-bit grid selector; signs_scale: original GGUF
    uint32 for 32 weights; group: the 8-weight group within those 32 weights.
    Returns four BF16 pairs. The table contains 256x8 BF16 magnitudes.
    """
    load = "ld.shared.v4.u32" if shared_lut else "ld.global.nc.v4.u32"
    addr_type = ".u32" if shared_lut else ".u64"
    address_setup = "cvt.u32.u64 table, $8;" if shared_lut else "mov.u64 table, $8;"
    address_calc = (
        "mad.lo.u32 address, grid, 16, table;"
        if shared_lut
        else "mad.wide.u32 address, grid, 16, table;"
    )
    chunks = []
    chunks.append(f"""
          and.b32 grid, $4, 255;
          {address_calc}
          {load} {{w0, w1, w2, w3}}, [address];
    """)
    for out in range(4):
        chunks.append(f"""
              shl.b32 lo, w{out}, 16;
              and.b32 hi, w{out}, 0xffff0000;
              mul.f32 lo, lo, scale;
              mul.f32 hi, hi, scale;
              cvt.rn.bf16x2.f32 ${out}, hi, lo;
              shr.u32 pair_signs, signs, {out * 2};
              mul.lo.u32 pair_signs, pair_signs, 0x40008000;
              lop3.b32 ${out}, ${out}, pair_signs, 0x80008000, 0x78;
        """)
    asm = (
        """{
       .reg .b16 dh;
       .reg .u32 grid, w0, w1, w2, w3, signs, parity, shift, nibble, pair_signs;
       .reg .f32 scale, subscale, lo, hi;
    """
        + f".reg {addr_type} address, table;\n"
        + address_setup
        + """
       cvt.u16.u32 dh, $5;
       cvt.f32.f16 scale, dh;
       shr.u32 nibble, $6, 28;
       cvt.rn.f32.u32 subscale, nibble;
       add.f32 subscale, subscale, 0f3f000000;
       mul.f32 scale, scale, subscale;
       mul.f32 scale, scale, 0f3e800000;
       mul.lo.u32 shift, $7, 7;
       shr.u32 signs, $6, shift;
       and.b32 signs, signs, 127;
       popc.b32 parity, signs;
       mad.lo.u32 signs, parity, 128, signs;
    """
        + "\n".join(chunks)
        + "\n}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            Uint32(indices).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(signs_scale).ir_value(loc=loc, ip=ip),
            Int32(group).ir_value(loc=loc, ip=ip),
            Int64(table_addr).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def decode_iq2_xxs_fragment(
    indices, base, signs_scale, group, table_addr, pair_offset, *, loc=None, ip=None
):
    """Two BF16 pairs separated by K8 from two adjacent packed grid bytes.

    group is 0 or 2 within a 32-weight group; pair_offset is 0/2/4/6.
    The shared LUT holds 256x8 BF16 magnitudes.
    """
    chunks = []
    for i in range(2):
        chunks.append(f"""
            shr.u32 grid, $2, {i * 8};
            and.b32 grid, grid, 255;
            mad.lo.u32 address, grid, 16, table;
            and.b32 offset, $7, 7;
            shl.b32 offset, offset, 1;
            add.u32 address, address, offset;
            ld.shared.u32 word, [address];
            shl.b32 lo, word, 16;
            and.b32 hi, word, 0xffff0000;
            mul.f32 lo, lo, scale;
            mul.f32 hi, hi, scale;
            cvt.rn.bf16x2.f32 ${i}, hi, lo;
            add.u32 shift, $5, {i};
            mul.lo.u32 shift, shift, 7;
            shr.u32 signs, $4, shift;
            and.b32 signs, signs, 127;
            popc.b32 parity, signs;
            mad.lo.u32 signs, parity, 128, signs;
            shr.u32 signs, signs, $7;
            mul.lo.u32 sign_pair, signs, 0x40008000;
            lop3.b32 ${i}, ${i}, sign_pair, 0x80008000, 0x78;
        """)
    asm = (
        """{
        .reg .b16 dh;
        .reg .u32 table, grid, shift, offset, address, word, signs, parity, sign_pair, nibble;
        .reg .f32 scale, subscale, lo, hi;
        cvt.u32.u64 table, $6;
        cvt.u16.u32 dh, $3;
        cvt.f32.f16 scale, dh;
        shr.u32 nibble, $4, 28;
        cvt.rn.f32.u32 subscale, nibble;
        add.f32 subscale, subscale, 0f3f000000;
        mul.f32 scale, scale, subscale;
        mul.f32 scale, scale, 0f3e800000;
    """
        + "\n".join(chunks)
        + "\n}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(indices).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(signs_scale).ir_value(loc=loc, ip=ip),
            Int32(group).ir_value(loc=loc, ip=ip),
            Int64(table_addr).ir_value(loc=loc, ip=ip),
            Int32(pair_offset).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,r,r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )

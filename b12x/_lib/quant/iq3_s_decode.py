"""IQ3_S register decoder shared by SIMT and tensor-core fragment loaders."""

from cutlass import Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def decode_iq3_s_eight(
    indices,
    base,
    signs,
    subscale,
    table_addr,
    *,
    shared_lut=False,
    loc=None,
    ip=None,
):
    """Exact BF16 decode of eight values with explicit signs and subscale.

    Indices are two 9-bit selectors in 16-bit lanes.
    Signs contains all eight original sign bits; subscale is a four-bit nibble.
    """
    load = "ld.shared.v2.u32" if shared_lut else "ld.global.nc.v2.u32"
    addr_type = ".u32" if shared_lut else ".u64"
    address_setup = "cvt.u32.u64 table, $8;" if shared_lut else "mov.u64 table, $8;"
    address_calc = (
        "mad.lo.u32 address, grid, 8, table;"
        if shared_lut
        else "mad.wide.u32 address, grid, 8, table;"
    )
    chunks = []
    for i in range(2):
        chunks.append(f"""
          bfe.u32 grid, $4, {16 * i}, 9;
          {address_calc}
          {load} {{w0, w1}}, [address];
        """)
        for j in range(2):
            out = 2 * i + j
            chunks.append(f"""
              shl.b32 lo, w{j}, 16;
              and.b32 hi, w{j}, 0xffff0000;
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
       .reg .u32 grid, w0, w1, signs, parity, shift, nibble, pair_signs;
       .reg .f32 scale, subscale, lo, hi;
    """
        + f".reg {addr_type} address, table;\n"
        + address_setup
        + """
       cvt.u16.u32 dh, $5;
       cvt.f32.f16 scale, dh;
       mad.lo.u32 nibble, $7, 2, 1;
       cvt.rn.f32.u32 subscale, nibble;
       mul.f32 scale, scale, subscale;
       mov.u32 signs, $6;
    """
        + "\n".join(chunks)
        + "\n}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            Uint32(indices).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(signs).ir_value(loc=loc, ip=ip),
            Int32(subscale).ir_value(loc=loc, ip=ip),
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
def decode_iq3_s_fragment(
    indices, base, signs_high_scale, group, table_addr, pair_offset, *, loc=None, ip=None
):
    """Two K8-separated BF16 pairs from compact IQ3_S planes.

    indices holds four original grid bytes for K16. Metadata packs two sign
    bytes, the original K32 high-bit byte, and a subscale nibble in bits24:27.
    group selects the K16 half (0/1); pair_offset is 0/2/4/6.
    """
    chunks = []
    for i in range(2):
        chunks.append(f"""
            shr.u32 shift, $7, 2;
            add.u32 shift, shift, {i * 2};
            mad.lo.u32 offset, $5, 4, shift;
            add.u32 offset, offset, 16;
            shr.u32 nibble, $4, offset;
            and.b32 nibble, nibble, 1;
            shl.b32 nibble, nibble, 8;
            shl.b32 shift, shift, 3;
            shr.u32 grid, $2, shift;
            and.b32 grid, grid, 255;
            or.b32 grid, grid, nibble;
            mad.lo.u32 address, grid, 8, table;
            and.b32 offset, $7, 3;
            shl.b32 offset, offset, 1;
            add.u32 address, address, offset;
            ld.shared.u32 word, [address];
            shl.b32 lo, word, 16;
            and.b32 hi, word, 0xffff0000;
            mul.f32 lo, lo, scale;
            mul.f32 hi, hi, scale;
            cvt.rn.bf16x2.f32 ${i}, hi, lo;
            shr.u32 signs, $4, {i * 8};
            and.b32 signs, signs, 255;
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
        shr.u32 nibble, $4, 24;
        and.b32 nibble, nibble, 15;
        mad.lo.u32 nibble, nibble, 2, 1;
        cvt.rn.f32.u32 subscale, nibble;
        mul.f32 scale, scale, subscale;
    """
        + "\n".join(chunks)
        + "\n}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(indices).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(signs_high_scale).ir_value(loc=loc, ip=ip),
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

"""Exact GGUF IQ2_XXS decoding before integration into GEMM fragments."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np
import pytest
import torch

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import cp_async4_shared_global, shared_ptr_to_u32
from b12x._lib.quant.iq2_xxs import iq2_xxs_execution_lut
from b12x._lib.quant.iq2_xxs_decode import decode_iq2_xxs_eight, decode_iq2_xxs_fragment
from b12x._lib.utils import current_cuda_stream
from tests._reference.helpers import require_b12x


class DecodeProbe:
    def __init__(self, shared, fragment=False):
        self.shared = shared
        self.fragment = fragment

    @property
    def __cache_key__(self):
        return (self.shared, self.fragment)

    @cute.jit
    def __call__(
        self,
        blocks: cute.Tensor,
        lut: cute.Tensor,
        out: cute.Tensor,
        count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(blocks, lut, out, count).launch(
            grid=((count * 32 + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        blocks: cute.Tensor,
        lut: cute.Tensor,
        out: cute.Tensor,
        count: cutlass.Int32,
    ):
        t, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        address = lut.iterator.toint()
        if cutlass.const_expr(self.shared):
            storage = cutlass.utils.SmemAllocator().allocate_array(
                cutlass.Uint32, 1024, byte_alignment=16
            )
            smem = shared_ptr_to_u32(storage)
            cp_async4_shared_global(smem + t * 16, address + cutlass.Int64(t) * 16)
            cp_async4_shared_global(smem + (t + 128) * 16, address + cutlass.Int64(t + 128) * 16)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            address = cutlass.Int64(smem)
        index = cutlass.Int64(b) * 128 + t
        block, group = index // 32, index % 32
        if block < count:
            base = cutlass.Uint32(blocks[block, 0]) | (
                cutlass.Uint32(blocks[block, 1]) << 8
            )
            indices = cutlass.Uint32(blocks[block, 2 + (group // 4) * 8 + group % 4])
            gas = cutlass.Uint32(0)
            for j in cutlass.range_constexpr(4):
                gas = gas | (
                    cutlass.Uint32(blocks[block, 6 + (group // 4) * 8 + j]) << (j * 8)
                )
            if cutlass.const_expr(self.fragment):
                # Map each eight-value output group to two MMA-style loads.
                pair_group = group - group % 2
                offset = 2 + (pair_group // 4) * 8 + pair_group % 4
                grid_pair = cutlass.Uint32(blocks[block, offset]) | (
                    cutlass.Uint32(blocks[block, offset + 1]) << 8
                )
                for pair in cutlass.range_constexpr(4):
                    first, second = decode_iq2_xxs_fragment(
                        grid_pair, base, gas, cutlass.Int32(pair_group % 4),
                        address, cutlass.Int32(pair * 2),
                    )
                    value = first
                    if group % 2:
                        value = second
                    out[index, pair] = cutlass.Int32(value)
            else:
                a, c, d, e = decode_iq2_xxs_eight(
                    indices, base, gas, cutlass.Int32(group % 4), address,
                    shared_lut=self.shared,
                )
                out[index, 0], out[index, 1] = cutlass.Int32(a), cutlass.Int32(c)
                out[index, 2], out[index, 3] = cutlass.Int32(d), cutlass.Int32(e)



@pytest.mark.parametrize("shared,fragment", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    "base_bits", [0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x3C01, 0x7BFF, 0xFBFF]
)
def test_iq2_xxs_matches_gguf_all_grid_sign_scale_combinations(shared, fragment, base_bits):
    gguf = pytest.importorskip("gguf")
    device = require_b12x()
    # 256 grid selectors x128 sign codes x16 subscales; adjacent vectors differ.
    groups = 256 * 128 * 16
    raw = np.zeros((groups // 32, 66), dtype=np.uint8)
    raw[:, :2] = np.array([base_bits], dtype=np.uint16).view(np.uint8)
    group = np.arange(groups, dtype=np.uint32)
    payload = raw[:, 2:].reshape(-1, 8)
    payload[:, :4] = (group % 256).astype(np.uint8).reshape(-1, 4)
    # Each 32-weight group shares its scale, while all four sign groups vary.
    signs = ((group // 256) % 128).reshape(-1, 4)
    subscale = (np.arange(groups // 4, dtype=np.uint32) // (256 * 128 // 4)) % 16
    gas = subscale << 28
    for j in range(4):
        gas |= signs[:, j] << (7 * j)
    payload[:, 4:] = gas.astype("<u4").view(np.uint8).reshape(-1, 4)
    expected = torch.from_numpy(
        gguf.dequantize(raw, gguf.GGMLQuantizationType.IQ2_XXS)
    ).bfloat16()
    source = torch.from_numpy(raw).to(device)
    lut = iq2_xxs_execution_lut(device, prepare=True)
    out = torch.empty((groups, 4), device=device, dtype=torch.int32)
    args = tuple(from_dlpack(t, assumed_align=16) for t in (source, lut, out))
    compiled = b12x_compile(
        DecodeProbe(shared, fragment), *args, cutlass.Int32(len(raw)), current_cuda_stream()
    )
    compiled(*args, cutlass.Int32(len(raw)), current_cuda_stream())
    assert torch.equal(
        out.cpu().view(torch.int16).reshape(-1), expected.view(torch.int16).reshape(-1)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, cutlass.Int32(len(raw)), current_cuda_stream())
    source[:, 2:].reshape(-1, 8)[:, :4].bitwise_xor_(255)
    payload[:, :4] ^= 255
    expected = torch.from_numpy(
        gguf.dequantize(raw, gguf.GGMLQuantizationType.IQ2_XXS)
    ).bfloat16()
    out.fill_(-1)
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.equal(
        out.cpu().view(torch.int16).reshape(-1), expected.view(torch.int16).reshape(-1)
    )

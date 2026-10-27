"""Independent GGUF oracles for compact IQ2_S and IQ3_S register decode."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np
import pytest
import torch

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import cp_async4_shared_global, shared_ptr_to_u32
from b12x._lib.quant.iq2_s import iq2_s_execution_lut
from b12x._lib.quant.iq3_s import iq3_s_execution_lut
from b12x._lib.quant.iq2_s_decode import decode_iq2_s_eight, decode_iq2_s_fragment
from b12x._lib.quant.iq3_s_decode import decode_iq3_s_eight, decode_iq3_s_fragment
from b12x._lib.utils import current_cuda_stream
from tests._reference.helpers import require_b12x


class DecodeProbe:
    def __init__(self, recipe, shared, fragment=False):
        self.fragment = fragment
        self.iq2 = recipe == "iq2_s"
        self.shared = shared
        self.lut_bytes = 16384 if self.iq2 else 4096

    @property
    def __cache_key__(self):
        return self.iq2, self.shared, self.fragment

    @cute.jit
    def __call__(self, blocks: cute.Tensor, lut: cute.Tensor, out: cute.Tensor,
                 count: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(blocks, lut, out, count).launch(
            grid=((count * 32 + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, blocks: cute.Tensor, lut: cute.Tensor, out: cute.Tensor,
               count: cutlass.Int32):
        t, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        address = lut.iterator.toint()
        if cutlass.const_expr(self.shared):
            storage = cutlass.utils.SmemAllocator().allocate_array(
                cutlass.Uint32, self.lut_bytes // 4, byte_alignment=16
            )
            smem = shared_ptr_to_u32(storage)
            for chunk in cutlass.range_constexpr(self.lut_bytes // (128 * 16)):
                offset = t + chunk * 128
                cp_async4_shared_global(smem + offset * 16, address + cutlass.Int64(offset) * 16)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            address = cutlass.Int64(smem)
        index = cutlass.Int64(b) * 128 + t
        block, group = index // 32, index % 32
        if block < count:
            base = cutlass.Uint32(blocks[block, 0]) | (cutlass.Uint32(blocks[block, 1]) << 8)
            qh = cutlass.Uint32(blocks[block, 66 + group // 4])
            if cutlass.const_expr(self.iq2):
                grid = cutlass.Uint32(blocks[block, 2 + group]) | (((qh >> (group % 4 * 2)) & 3) << 8)
                signs = cutlass.Uint32(blocks[block, 34 + group])
                scale = (cutlass.Uint32(blocks[block, 74 + group // 4]) >> (group // 2 % 2 * 4)) & 15
                if cutlass.const_expr(self.fragment):
                    start = group - group % 2
                    descriptors = cutlass.Uint32(0)
                    for j in cutlass.range_constexpr(2):
                        descriptor = cutlass.Uint32(blocks[block, 2 + start + j]) | (cutlass.Uint32(blocks[block, 34 + start + j]) << 8)
                        descriptors = descriptors | (descriptor << (j * 16))
                    metadata = qh | (cutlass.Uint32(blocks[block, 74 + group // 4]) << 8)
                    for pair in cutlass.range_constexpr(4):
                        first, second = decode_iq2_s_fragment(descriptors, base, metadata, cutlass.Int32(group // 2 % 2), address, cutlass.Int32(pair * 2))
                        value = first
                        if group % 2:
                            value = second
                        out[index, pair] = cutlass.Int32(value)
                else:
                    a, c, d, e = decode_iq2_s_eight(grid, base, signs, scale, address, shared_lut=self.shared)

            else:
                grid = cutlass.Uint32(0)
                for j in cutlass.range_constexpr(2):
                    value = cutlass.Uint32(blocks[block, 2 + group * 2 + j]) | (((qh >> (group % 4 * 2 + j)) & 1) << 8)
                    grid = grid | (value << (j * 16))
                signs = cutlass.Uint32(blocks[block, 74 + group])
                scale = (cutlass.Uint32(blocks[block, 106 + group // 8]) >> (group // 4 % 2 * 4)) & 15
                if cutlass.const_expr(self.fragment):
                    start = group - group % 2
                    indices = cutlass.Uint32(0)
                    for j in cutlass.range_constexpr(4):
                        indices = indices | (cutlass.Uint32(blocks[block, 2 + start * 2 + j]) << (j * 8))
                    metadata = cutlass.Uint32(blocks[block, 74 + start]) | (cutlass.Uint32(blocks[block, 75 + start]) << 8) | (qh << 16) | (scale << 24)
                    for pair in cutlass.range_constexpr(4):
                        first, second = decode_iq3_s_fragment(indices, base, metadata, cutlass.Int32(group // 2 % 2), address, cutlass.Int32(pair * 2))
                        value = first
                        if group % 2:
                            value = second
                        out[index, pair] = cutlass.Int32(value)
                else:
                    a, c, d, e = decode_iq3_s_eight(grid, base, signs, scale, address, shared_lut=self.shared)
            if cutlass.const_expr(not self.fragment):
                out[index, 0], out[index, 1] = cutlass.Int32(a), cutlass.Int32(c)
                out[index, 2], out[index, 3] = cutlass.Int32(d), cutlass.Int32(e)


def make_blocks(recipe, base_bits):
    iq2 = recipe == "iq2_s"
    grids = 1024 if iq2 else 512
    groups = grids * 256 * 16
    raw = np.zeros((groups // 32, 82 if iq2 else 110), dtype=np.uint8)
    raw[:, :2] = np.array([base_bits], dtype=np.uint16).view(np.uint8)
    group = np.arange(groups, dtype=np.uint32)
    index = group % grids
    signs = ((group // grids) % 256).astype(np.uint8)
    scales = ((group // (grids * 256)) % 16).astype(np.uint8)
    if iq2:
        raw[:, 2:34] = (index & 255).astype(np.uint8).reshape(-1, 32)
        raw[:, 34:66] = signs.reshape(-1, 32)
        high = (index >> 8).reshape(-1, 4)
        raw[:, 66:74] = sum(high[:, j] << (j * 2) for j in range(4)).astype(np.uint8).reshape(-1, 8)
        scale = scales.reshape(-1, 4)
        raw[:, 74:] = (scale[:, 0] | (scale[:, 2] << 4)).reshape(-1, 8)
    else:
        indices = np.stack((index, (index + 137) % grids), axis=-1)
        raw[:, 2:66] = (indices & 255).astype(np.uint8).reshape(-1, 64)
        high = (indices >> 8).reshape(-1, 8)
        raw[:, 66:74] = sum(high[:, j] << j for j in range(8)).astype(np.uint8).reshape(-1, 8)
        raw[:, 74:106] = signs.reshape(-1, 32)
        scale = scales.reshape(-1, 8)
        raw[:, 106:] = (scale[:, 0] | (scale[:, 4] << 4)).reshape(-1, 4)
    return raw


@pytest.mark.parametrize("recipe,shared,fragment", [("iq2_s", False, False), ("iq2_s", True, False), ("iq2_s", True, True), ("iq3_s", False, False), ("iq3_s", True, False), ("iq3_s", True, True)])
@pytest.mark.parametrize("base_bits", [0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x3C01, 0x7BFF, 0xFBFF])
def test_exact_grid_sign_scale_decode_and_replay(recipe, shared, fragment, base_bits):
    gguf = pytest.importorskip("gguf")
    device = require_b12x()
    raw = make_blocks(recipe, base_bits)
    qtype = getattr(gguf.GGMLQuantizationType, recipe.upper())
    expected = torch.from_numpy(gguf.dequantize(raw, qtype)).bfloat16()
    source = torch.from_numpy(raw).to(device)
    lut = (iq2_s_execution_lut if recipe == "iq2_s" else iq3_s_execution_lut)(device, prepare=True)
    out = torch.empty((len(raw) * 32, 4), dtype=torch.int32, device=device)
    args = tuple(from_dlpack(t, assumed_align=16) for t in (source, lut, out))
    compiled = b12x_compile(DecodeProbe(recipe, shared, fragment), *args, cutlass.Int32(len(raw)), current_cuda_stream())
    compiled(*args, cutlass.Int32(len(raw)), current_cuda_stream())
    assert torch.equal(out.cpu().view(torch.int16).reshape(-1), expected.view(torch.int16).reshape(-1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, cutlass.Int32(len(raw)), current_cuda_stream())
    end = 34 if recipe == "iq2_s" else 66
    source[:, 2:end].bitwise_xor_(255)
    raw[:, 2:end] ^= 255
    expected = torch.from_numpy(gguf.dequantize(raw, qtype)).bfloat16()
    out.fill_(-1)
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.equal(out.cpu().view(torch.int16).reshape(-1), expected.view(torch.int16).reshape(-1))

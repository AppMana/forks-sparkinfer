"""Independent GGUF decoder checks for native dense integer register math."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np
import pytest
import torch

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.quant.gguf_int_decode import q8_0_pair_to_bf16, q6_k_pair_to_bf16
from b12x._lib.utils import current_cuda_stream
from tests._reference.helpers import require_b12x


class DecodeProbe:
    def __init__(self, q6):
        self.q6 = q6
        self.pairs = 128 if q6 else 16

    @property
    def __cache_key__(self):
        return self.q6,

    @cute.jit
    def __call__(self, src: cute.Tensor, out: cute.Tensor, count: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(src, out, count).launch(grid=((count*self.pairs+127)//128,1,1),
                                           block=(128,1,1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Tensor, out: cute.Tensor, count: cutlass.Int32):
        t, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        index = cutlass.Int64(b)*128+t
        row, pair = index//self.pairs, index%self.pairs
        if row < count:
            k = pair*2
            if cutlass.const_expr(self.q6):
                lo = k//128*64+k%64
                hi = 128+k//128*32+k%32
                low = cutlass.Uint32(src[row,lo]) | (cutlass.Uint32(src[row,lo+1]) << 8)
                high = cutlass.Uint32(src[row,hi]) | (cutlass.Uint32(src[row,hi+1]) << 8)
                base = cutlass.Uint32(src[row,208]) | (cutlass.Uint32(src[row,209]) << 8)
                result = q6_k_pair_to_bf16(low,high,base,cutlass.Uint32(src[row,192+k//16]),
                                          cutlass.Uint32(k%128//64*4),cutlass.Uint32(k%128//32*2))
            else:
                packed = cutlass.Uint32(src[row,2+k]) | (cutlass.Uint32(src[row,3+k]) << 8)
                base = cutlass.Uint32(src[row,0]) | (cutlass.Uint32(src[row,1]) << 8)
                result = q8_0_pair_to_bf16(packed,base)
            out[index] = cutlass.Int32(result)


@pytest.mark.parametrize('recipe', ['Q8_0', 'Q6_K'])
@pytest.mark.parametrize('base', [0,0x8000,1,0x8001,0x3ff,0x400,0x3c01,0x7bff,0xfbff])
def test_exact_codes_scales_boundaries_and_replay(recipe, base):
    gguf = pytest.importorskip('gguf')
    device = require_b12x()
    q6 = recipe == 'Q6_K'
    if q6:
        raw = np.zeros((256,210),dtype=np.uint8)
        # Every signed subscale sees every code, in all four high-bit planes.
        q = np.broadcast_to(np.arange(256,dtype=np.uint16)%64,(256,256))
        for k in range(256):
            raw[:,k//128*64+k%64] |= ((q[:,k]&15) << (k%128//64*4)).astype(np.uint8)
            raw[:,128+k//128*32+k%32] |= ((q[:,k]>>4) << (k%128//32*2)).astype(np.uint8)
        raw[:,192:208] = np.arange(256,dtype=np.uint8)[:,None]
        raw[:,208:] = np.array([base],dtype=np.uint16).view(np.uint8)
    else:
        raw = np.zeros((8,34),dtype=np.uint8)
        raw[:,2:] = np.arange(256,dtype=np.uint8).reshape(8,32)
        raw[:,:2] = np.array([base],dtype=np.uint16).view(np.uint8)
    source = torch.from_numpy(raw).to(device)
    out = torch.empty(len(raw)*(128 if q6 else 16),dtype=torch.int32,device=device)
    args = tuple(from_dlpack(x,assumed_align=16) for x in (source,out))
    fn = b12x_compile(DecodeProbe(q6),*args,cutlass.Int32(len(raw)),current_cuda_stream())
    fn(*args,cutlass.Int32(len(raw)),current_cuda_stream())

    def check():
        ref = torch.from_numpy(gguf.dequantize(raw,getattr(gguf.GGMLQuantizationType,recipe))).bfloat16()
        assert torch.equal(out.cpu().view(torch.int16).reshape(-1),ref.view(torch.int16).reshape(-1))
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args,cutlass.Int32(len(raw)),current_cuda_stream())
    start,end = (0,208) if q6 else (2,34)
    source[:,start:end].bitwise_xor_(255)
    raw[:,start:end] ^= 255
    out.fill_(-1)
    allocated = torch.cuda.memory_stats(device)['allocation.all.allocated']
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == allocated
    check()

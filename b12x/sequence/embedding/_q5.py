"""Gather exact GGUF Q5_K rows, including batch-sized disk staging tables."""
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x._lib.quant.gguf_int_decode import q5_k_pair_to_bf16
from ._kernel import _EmbeddingProgram, _invalid_index


@dataclass(frozen=True)
class Q5KEmbeddingWeight:
    """Raw, unmodified 176-byte Q5_K blocks in CUDA-addressable storage."""
    values: torch.Tensor

    @property
    def out_features(self):
        return self.values.shape[0]

    @property
    def in_features(self):
        return self.values.shape[1] // 176 * 256


class _Q5Embedding:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, values: cute.Pointer, ids: cute.Pointer, out: cute.Pointer,
                 count: cute.Pointer, capacity: Int32, table_rows: Int64,
                 use_count: Int32, stream: cuda.CUstream):
        self.kernel(values, ids, out, count, capacity, table_rows,
                    use_count).launch(grid=(capacity, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, values: cute.Pointer, ids: cute.Pointer, out: cute.Pointer,
               count: cute.Pointer, capacity: Int32, table_rows: Int64, use_count: Int32):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        live = capacity
        if use_count != 0:
            live = count[0]
        if live < 0 or live > capacity:
            _invalid_index()
        elif row < live:
            index = Int64(ids[row])
            if index < 0 or index >= table_rows:
                _invalid_index()
            else:
                for block in cutlass.range_constexpr(self.width // 256):
                    base = (index * Int64(self.width // 256) + Int64(block)) * 176
                    col = tid * 2
                    group = col // 32
                    lane = col % 32
                    bases = Uint32(0)
                    for byte in cutlass.range_constexpr(4):
                        bases = bases | (Uint32(values[base + byte]) << (byte * 8))
                    scale = Uint32(0)
                    minimum = Uint32(0)
                    if group < 4:
                        scale = Uint32(values[base + 4 + group]) & 63
                        minimum = Uint32(values[base + 8 + group]) & 63
                    else:
                        packed = Uint32(values[base + 12 + group - 4])
                        scale = (packed & 15) | ((Uint32(values[base + group]) >> 2) & 48)
                        minimum = (packed >> 4) | ((Uint32(values[base + 4 + group]) >> 2) & 48)
                    q = Uint32(0)
                    for item in cutlass.range_constexpr(2):
                        low = (Uint32(values[base + 48 + group // 2 * 32 + lane + item]) >> ((group % 2) * 4)) & 15
                        high = (Uint32(values[base + 16 + lane + item]) >> group) & 1
                        q = q | ((low | (high << 4)) << (8 * item))
                    out[Int64(row) * (self.width // 2) + Int64(block * 128 + tid)] = q5_k_pair_to_bf16(q, bases, scale, minimum)


@program_cache
def compile_q5_embedding(width, id_dtype, device):
    key = (width, str(id_dtype), device)
    entry = _Q5Embedding(width)
    raise_if_kernel_resolution_frozen('cute.compile', target=entry, cache_key=key)
    types = (Uint8, Int64 if id_dtype == torch.int64 else Int32, Uint32, Int32)
    pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8) for t in types)
    with torch.cuda.device(device):
        compiled = compile_cute(entry, *pointers, Int32(1), Int64(1), Int32(0),
                                current_cuda_stream(), compile_spec=KernelCompileSpec.from_key(
                                    'sequence.embedding.q5_k', 1, key))
    return attach_programs(_EmbeddingProgram(compiled, types), compiled)


def launch(weight, ids, out, num_rows, *, prepared):
    if ids.numel() == 0:
        return
    compiled, types = prepared
    tensors = (weight.values, ids, out, ids if num_rows is None else num_rows)
    pointers = tuple(make_ptr(t, tensor.data_ptr(), cute.AddressSpace.gmem,
                              assumed_align=t.width // 8) for t, tensor in zip(types, tensors, strict=True))
    with torch.cuda.device(weight.values.device):
        run_compiled(compiled, (*pointers, Int32(ids.numel()), Int64(weight.out_features),
                                Int32(num_rows is not None), current_cuda_stream()))


def check(weight, ids, out, num_rows):
    values = weight.values
    if (not values.is_cuda or values.dtype != torch.uint8 or values.ndim != 2
            or not values.is_contiguous() or values.shape[1] == 0 or values.shape[1] % 176):
        raise ValueError('Q5_K embedding requires contiguous CUDA raw [N,K/256*176] bytes')
    if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous() or ids.numel() >= 2**31:
        raise ValueError('Q5_K embedding requires contiguous integer IDs and Int32 capacity')
    if (out.shape != (*ids.shape, weight.in_features) or out.dtype != torch.bfloat16
            or not out.is_contiguous() or out.device != values.device or ids.device != values.device):
        raise ValueError('Q5_K embedding requires matching BF16 output on the weight device')
    reads = (values, ids)
    if num_rows is not None:
        if (num_rows.device != values.device or num_rows.dtype != torch.int32
                or num_rows.numel() != 1 or not num_rows.is_contiguous() or ids.numel() == 0):
            raise ValueError('Q5_K embedding live count must be CUDA Int32 with positive capacity')
        reads += (num_rows,)
    if any(torch._C._overlaps(out, source) for source in reads):
        raise ValueError('Q5_K embedding output must not alias inputs')

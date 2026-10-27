"""Native Q8_0 gather from the same compact planes used by dense GEMM."""
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
from b12x._lib.quant.gguf_int_decode import q8_0_pair_to_bf16
from ._kernel import _EmbeddingProgram, _invalid_index


class _Q8Embedding:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, values: cute.Pointer, metadata: cute.Pointer,
                 ids: cute.Pointer, out: cute.Pointer, count: cute.Pointer,
                 capacity: Int32, table_rows: Int64, use_count: Int32,
                 stream: cuda.CUstream):
        self.kernel(values, metadata, ids, out, count, capacity, table_rows,
                    use_count).launch(grid=(capacity, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, values: cute.Pointer, metadata: cute.Pointer,
               ids: cute.Pointer, out: cute.Pointer, count: cute.Pointer,
               capacity: Int32, table_rows: Int64, use_count: Int32):
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
                for item in cutlass.range_constexpr((self.width // 2 + 127) // 128):
                    pair = tid + item * 128
                    if pair < self.width // 2:
                        col = pair * 2
                        src = index * Int64(self.width) + Int64(col)
                        packed = Uint32(values[src]) | (Uint32(values[src + 1]) << 8)
                        scale = ((index // 128 * (self.width // 256) + Int64(col // 256)) * 2048
                                 + Int64(col % 256 // 32) * 256 + index % 128 * 2)
                        base = Uint32(metadata[scale]) | (Uint32(metadata[scale + 1]) << 8)
                        out[Int64(row) * (self.width // 2) + Int64(pair)] = q8_0_pair_to_bf16(packed, base)


@program_cache
def compile_q8_embedding(width, id_dtype, device):
    key = (width, str(id_dtype), device)
    entry = _Q8Embedding(width)
    raise_if_kernel_resolution_frozen('cute.compile', target=entry, cache_key=key)
    types = (Uint8, Uint8, Int64 if id_dtype == torch.int64 else Int32, Uint32, Int32)
    pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8) for t in types)
    with torch.cuda.device(device):
        compiled = compile_cute(entry, *pointers, Int32(1), Int64(1), Int32(0),
                                current_cuda_stream(), compile_spec=KernelCompileSpec.from_key(
                                    'sequence.embedding.q8_0', 1, key))
    return attach_programs(_EmbeddingProgram(compiled, types), compiled)


def launch(weight, ids, out, num_rows, *, prepared):
    if ids.numel() == 0:
        return
    compiled, types = prepared
    tensors = (weight.values, weight.metadata, ids, out, ids if num_rows is None else num_rows)
    pointers = tuple(make_ptr(t, tensor.data_ptr(), cute.AddressSpace.gmem,
                              assumed_align=t.width // 8) for t, tensor in zip(types, tensors, strict=True))
    with torch.cuda.device(weight.values.device):
        run_compiled(compiled, (*pointers, Int32(ids.numel()), Int64(weight.out_features),
                                Int32(num_rows is not None), current_cuda_stream()))


def check(weight, ids, out, num_rows):
    values, metadata = weight.values, weight.metadata
    n, k = weight.out_features, weight.in_features
    if (not values.is_cuda or values.dtype != torch.uint8 or not values.is_contiguous()
            or values.shape != (n, k) or k % 256 or k <= 0 or n <= 0):
        raise ValueError('Q8_0 embedding requires compact CUDA [N,K] values and K256 alignment')
    if (metadata.device != values.device or metadata.dtype != torch.uint8
            or not metadata.is_contiguous() or metadata.shape != ((n + 127) // 128, k // 256, 2048)):
        raise ValueError('Q8_0 embedding metadata geometry differs')
    if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous() or ids.numel() >= 2**31:
        raise ValueError('Q8_0 embedding requires contiguous integer IDs and Int32 capacity')
    if (out.shape != (*ids.shape, k) or out.dtype != torch.bfloat16 or not out.is_contiguous()
            or out.device != values.device or ids.device != values.device):
        raise ValueError('Q8_0 embedding requires matching BF16 output on the weight device')
    reads = (values, metadata, ids)
    if num_rows is not None:
        if (num_rows.device != values.device or num_rows.dtype != torch.int32
                or num_rows.numel() != 1 or not num_rows.is_contiguous() or ids.numel() == 0):
            raise ValueError('Q8_0 embedding live count must be CUDA Int32 with positive capacity')
        reads += (num_rows,)
    if any(torch._C._overlaps(out, source) for source in reads):
        raise ValueError('Q8_0 embedding output must not alias inputs')

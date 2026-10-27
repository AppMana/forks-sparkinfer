"""Exact compact row lookup, dynamic replay, and offsets beyond 2 GiB."""
import numpy as np
import pytest
import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled._q8_0 import Q8_0LinearWeight
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import embedding
from b12x._lib.runtime_control import kernel_resolution_guard


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('large,id_dtype', [(False, torch.int32), (False, torch.int64), (True, torch.int64)])
def test_exact_q8_rows_and_mutated_replay(large, id_dtype):
    gguf = pytest.importorskip('gguf')
    k = 5120
    rng = np.random.default_rng(881)
    raw = rng.integers(0, 256, (128, k // 32, 34), dtype=np.uint8)
    raw[..., :2] = rng.normal(0, .003, (128, k // 32, 1)).astype(np.float16).view(np.uint8)
    expected = torch.from_numpy(gguf.dequantize(raw.reshape(128, -1), gguf.GGMLQuantizationType.Q8_0)).bfloat16().cuda()
    weight = blockscaled.pack_weight(torch.from_numpy(raw).cuda(), recipe='q8_0')
    first = 0
    if large:
        first = ((2**31 // k + 127) // 128) * 128
        n = first + 128
        if torch.cuda.mem_get_info()[0] < n * k * 2:
            pytest.skip('needs about 4.5 GiB free')
        values = torch.empty(n, k, dtype=torch.uint8, device='cuda')
        metadata = torch.empty(n // 128, k // 256, 2048, dtype=torch.uint8, device='cuda')
        values[first:].copy_(weight.values)
        metadata[first // 128:].copy_(weight.metadata)
        weight = Q8_0LinearWeight(values, metadata, k, n)
        assert first * k >= 2**31
    ids = torch.full((11,), first, device='cuda', dtype=id_dtype)
    out = torch.empty(11, k, dtype=torch.bfloat16, device='cuda')
    count = torch.tensor(11, dtype=torch.int32, device='cuda')
    plan = embedding.plan(embedding.query_from_call(weight, ids, out=out, num_rows=count), device=ids.device)
    with PreparationSession(device=ids.device, autotune=False, compile_workers=1) as session:
        session.prepare((plan.request(name='Q8 embedding', prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(weight, ids, out=out, num_rows=count))),))
        session.freeze()
        with kernel_resolution_guard('Q8 lookup'):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                embedding.run(weight, ids, out=out, num_rows=count, plan=plan)
            for m in (11, 1, 0, 7):
                local_ids = torch.arange(m, device='cuda') * 11 % 128
                ids.fill_(-1)
                ids[:m].copy_((first + local_ids).to(id_dtype))
                count.fill_(m)
                out.fill_(float('nan'))
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay()
                torch.cuda.synchronize()
                assert allocated == torch.cuda.memory_stats()['allocation.all.allocated']
                assert torch.equal(out[:m].view(torch.int16), expected[local_ids].view(torch.int16))
                assert torch.isnan(out[m:]).all()
            graph.reset()

"""Exact compact row lookup, dynamic replay, and offsets beyond 2 GiB."""
import numpy as np
import pytest
import torch

from b12x.sequence.embedding import Q5KEmbeddingWeight
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import embedding
from b12x._lib.runtime_control import kernel_resolution_guard


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('large,id_dtype', [(False, torch.int32), (False, torch.int64), (True, torch.int64)])
def test_exact_q5_rows_and_mutated_replay(large, id_dtype):
    gguf = pytest.importorskip('gguf')
    k = 256
    rng = np.random.default_rng(881)
    raw = rng.integers(0, 256, (128, k // 256, 176), dtype=np.uint8)
    raw[..., :4] = rng.normal(0, .003, (128, k // 256, 2)).astype(np.float16).view(np.uint8)
    bases = np.array([0., -0., 2**-24, -2**-24, 1., -1., 65504., -65504.], dtype=np.float16)
    raw[:8, 0, :2] = bases.view(np.uint8).reshape(8, 2)
    expected = torch.from_numpy(gguf.dequantize(raw.reshape(128, -1), gguf.GGMLQuantizationType.Q5_K)).bfloat16().cuda()
    weight = Q5KEmbeddingWeight(torch.from_numpy(raw.reshape(128, -1)).cuda())
    row_bytes = k // 256 * 176
    first = 0
    if large:
        first = (2**31 + row_bytes - 1) // row_bytes
        n = first + 128
        if torch.cuda.mem_get_info()[0] < n * row_bytes * 2:
            pytest.skip('needs about 4.5 GiB free')
        values = torch.empty(n, row_bytes, dtype=torch.uint8, device='cuda')
        values[first:].copy_(weight.values)
        weight = Q5KEmbeddingWeight(values)
        assert first * row_bytes >= 2**31
    ids = torch.full((11,), first, device='cuda', dtype=id_dtype)
    out = torch.empty(11, k, dtype=torch.bfloat16, device='cuda')
    count = torch.tensor(11, dtype=torch.int32, device='cuda')
    plan = embedding.plan(embedding.query_from_call(weight, ids, out=out, num_rows=count), device=ids.device)
    with PreparationSession(device=ids.device, autotune=False, compile_workers=1) as session:
        session.prepare((plan.request(name='Q5_K embedding', prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(weight, ids, out=out, num_rows=count))),))
        session.freeze()
        with kernel_resolution_guard('Q5_K lookup'):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_disk_q5_bounded_offset_and_changed_rows(tmp_path):
    from b12x.sequence.embedding._disk import Q5KDiskTable
    gguf = pytest.importorskip('gguf')
    rng = np.random.default_rng(91)
    raw = rng.integers(0, 256, (40, 176), dtype=np.uint8)
    raw[:, :4] = rng.normal(0, .02, (40, 2)).astype(np.float16).view(np.uint8)
    expected = torch.from_numpy(gguf.dequantize(raw, gguf.GGMLQuantizationType.Q5_K)).bfloat16().cuda()
    path = tmp_path / 'table.bin'
    offset = 2**32 + 192
    with path.open('wb') as f:
        f.seek(offset)
        f.write(raw.tobytes())
    table = Q5KDiskTable(path=path, offset=offset, table_rows=40, max_lookups=13,
                        device='cuda')
    ids = torch.empty(13, dtype=torch.int64, device='cuda')
    out = torch.empty(13, 256, dtype=torch.bfloat16, device='cuda')
    try:
        with kernel_resolution_guard('Q5 disk lookup'):
            for selected in ([39, 0, 1, 39, 12], [5], [], [31, 13, 0, 38]):
                n = len(selected)
                ids[:n].copy_(torch.tensor(selected, dtype=torch.int64, device='cuda'))
                out.fill_(float('nan'))
                table.lookup(ids[:n], out[:n])
                torch.cuda.synchronize()
                assert torch.equal(out[:n], expected[selected])
                assert torch.isnan(out[n:]).all()
        assert table.stats()['weight_cache_bytes'] == 13 * 176
    finally:
        table.close()

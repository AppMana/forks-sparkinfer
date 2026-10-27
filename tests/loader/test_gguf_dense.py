"""Compact streaming, large file offsets and block-aligned TP padding."""
import numpy as np
import pytest
import torch

from b12x.loader.gguf_dense import DenseTensor, prepare_dense_partition


def fixture(tmp_path, kind, n=136, k=2304):
    block, size = (256, 210) if kind == "Q6_K" else (32, 34)
    rng = np.random.default_rng(331)
    raw = rng.integers(0, 256, (n, k // block, size), dtype=np.uint8)
    bases = rng.normal(0, .001, (n, k // block, 1)).astype(np.float16).view(np.uint8)
    if kind == "Q6_K":
        raw[..., 208:210] = bases
    else:
        raw[..., :2] = bases
    path = tmp_path / kind
    offset = (1 << 32) + 64
    with path.open('wb') as f:
        f.seek(offset)
        f.write(raw.tobytes())
    return DenseTensor(path, 'test.weight', kind, offset, n, k), raw


@pytest.mark.parametrize('kind', ['Q6_K', 'Q8_0'])
def test_dense_bounded_reader(tmp_path, kind):
    source, raw = fixture(tmp_path, kind)
    block = 256 if kind == 'Q6_K' else 32
    actual = source.read_rows(128, 16, column_start=1280, columns=1280).numpy()
    expected = np.zeros_like(actual)
    expected[:8, :1024 // block] = raw[128:, 1280 // block:]
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='block'):
        source.read_rows(0, 8, column_start=1, columns=256)
    with source.path.open('r+b') as f:
        f.truncate(source.offset + 1)
    with pytest.raises(ValueError, match='Truncated'):
        source.read_rows(0, 8)


@pytest.mark.parametrize('kind', ['Q6_K', 'Q8_0'])
def test_streamed_dense_tp_sum(tmp_path, kind):
    gguf = pytest.importorskip('gguf')
    from b12x.gemm import blockscaled
    from tests.conftest import require_b12x
    from tests.gemm._blockscaled import prepared
    from b12x._lib.runtime_control import kernel_resolution_guard

    device = require_b12x()
    source, raw = fixture(tmp_path, kind)
    ref = torch.from_numpy(gguf.dequantize(raw.reshape(136, -1), getattr(gguf.GGMLQuantizationType, kind))).bfloat16().to(device)
    x = torch.randn(17, 2560, dtype=torch.bfloat16, device=device)
    x[:, 2304:] = 0
    outputs = []
    # I2304 -> padded I2560 -> two K1280 partitions. The second owns
    # four original K256 blocks and one zero block, preserving exact scales.
    for rank in range(2):
        weight = prepare_dense_partition(source, device=device, column_start=rank * 1280, columns=1280)
        expected_packing = blockscaled.pack_weight(source.read_rows(0, 136, column_start=rank * 1280, columns=1280).to(device), recipe=kind.lower())
        assert torch.equal(weight.values, expected_packing.values)
        assert torch.equal(weight.metadata, expected_packing.metadata)
        activation = x[:, rank * 1280:(rank + 1) * 1280].contiguous()
        out = torch.empty(17, 136, dtype=torch.bfloat16, device=device)
        cfg = blockscaled.BlockscaledConfig(mode='a16', tile_m=32, tile_n=128, tile_k=128, split_k=1)
        with prepared(activation, weight, out=out, override=cfg) as plan, kernel_resolution_guard('dense TP replay'):
            for m in (1, 8, 17):
                a, o = activation[:m], out[:m]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    blockscaled.mm(a, weight, out=o, plan=plan)
                a.neg_()
                o.fill_(float('nan'))
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay()
                torch.cuda.synchronize()
                assert allocated == torch.cuda.memory_stats()['allocation.all.allocated']
                a.neg_()
                # Captured output was computed after negation.
                assert torch.isfinite(o).all()
                graph.reset()
            outputs.append(-out.float())
    expected = x[:, :2304].float() @ ref.float().T
    assert (sum(outputs) - expected).norm() / expected.norm() < .004


@pytest.mark.parametrize('kind', ['Q6_K', 'Q8_0'])
def test_dense_gguf_index_roundtrip(tmp_path, kind):
    gguf = pytest.importorskip('gguf')
    from b12x.loader.gguf_dense import index_dense

    _, raw = fixture(tmp_path, kind)
    path = tmp_path / (kind + '.gguf')
    writer = gguf.GGUFWriter(path, 'deepseek41')
    writer.add_tensor('blk.27.ffn_down_shexp.weight', raw.reshape(136, -1),
                      raw_dtype=getattr(gguf.GGMLQuantizationType, kind))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    source = index_dense([path])['blk.27.ffn_down_shexp.weight']
    assert (source.rows, source.columns, source.quant_type) == (136, 2304, kind)
    np.testing.assert_array_equal(source.read_rows(8, 16).numpy(), raw[8:24])
    with pytest.raises(ValueError, match='Duplicate'):
        index_dense([path, path])

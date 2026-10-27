"""Packed Q6_K GEMM: independent GGUF oracle, tails and frozen replay."""

import numpy as np
import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from ._blockscaled import prepared
from ..conftest import require_b12x


@pytest.mark.parametrize("kwargs", [
    {"scale": torch.tensor(1.0)},
    {"global_scale": torch.tensor(1.0)},
    {"global_scale_kind": "reciprocal"},
])
def test_q6_k_rejects_external_scales(kwargs):
    with pytest.raises(ValueError, match="scales are embedded"):
        blockscaled.pack_weight(torch.empty(0), recipe="q6_k", **kwargs)


@pytest.mark.parametrize(
    "config", [(16, 64, 64, 4), (32, 128, 128, 1), (64, 128, 256, 1)]
)
@pytest.mark.parametrize("n", [128, 136])
def test_q6_k_native_gemm_and_changed_input_replay(config, n):
    gguf = pytest.importorskip("gguf")
    device = require_b12x()
    k, capacity = 768, 65
    rng = np.random.default_rng(417)
    raw = rng.integers(0, 256, (n, k // 256, 210), dtype=np.uint8)
    base = rng.normal(0, 0.001, (n, k // 256, 1)).astype(np.float16)
    raw[..., 208:] = base.view(np.uint8)
    raw[::7, :, 208:] = 0
    expected_weight = (
        torch.from_numpy(
            gguf.dequantize(raw.reshape(n, -1), gguf.GGMLQuantizationType.Q6_K)
        )
        .bfloat16()
        .to(device)
    )
    weight = blockscaled.pack_weight(torch.from_numpy(raw).to(device), recipe="q6_k")
    if n == 128:
        assert weight.values.numel() + weight.metadata.numel() == raw.size
    x = torch.randn((capacity, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((capacity, n), device=device, dtype=torch.bfloat16)
    bm, bn, bk, split = config
    override = blockscaled.BlockscaledConfig(
        mode="a16", tile_m=bm, tile_n=bn, tile_k=bk, split_k=split
    )
    with (
        prepared(x, weight, out=out, override=override) as plan,
        kernel_resolution_guard("Q6_K frozen graph"),
    ):
        for m in (1, 3, 8, 17, capacity):
            source, output = x[:m], out[:m]
            expected = source.float() @ expected_weight.float().T
            blockscaled.mm(source, weight, out=output, plan=plan)
            assert torch.isfinite(output).all() and output.norm() > 0
            assert (output.float() - expected).norm() / expected.norm() < 0.004
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                blockscaled.mm(source, weight, out=output, plan=plan)
            source.neg_()
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert (output.float() + expected).norm() / expected.norm() < 0.004

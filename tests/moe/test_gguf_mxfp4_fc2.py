"""Nonuniform contiguous BF16 inputs catch fused-layout leaks in standalone FC2."""

import numpy as np
import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.loader.gguf_mxfp4 import prepare_gguf_mxfp4_fc2
from b12x.moe.fused_moe._impl import prewarm_w4a16_fc2_e8m0, run_w4a16_fc2_e8m0
from benchmarks.benchmark_iq2_xs_moe import capture_lifetime, check
from tests._reference.helpers import require_b12x


@pytest.mark.parametrize("intermediate", [256, 512, 2304])
def test_contiguous_fc2_live_routes_and_graph(intermediate):
    gguf = pytest.importorskip("gguf")
    device = require_b12x()
    e, h = 2, 128
    rng = np.random.default_rng(83)
    raw = rng.integers(0, 256, (e, h, intermediate // 32, 17), dtype=np.uint8)
    raw[..., 0] = rng.integers(112, 121, raw.shape[:-1], dtype=np.uint8)
    reference = (
        torch.from_numpy(
            gguf.dequantize(raw.reshape(e * h, -1), gguf.GGMLQuantizationType.MXFP4)
        )
        .reshape(e, h, intermediate)
        .to(device=device, dtype=torch.bfloat16)
    )
    prepared = prepare_gguf_mxfp4_fc2(torch.from_numpy(raw).to(device))
    launcher = prewarm_w4a16_fc2_e8m0(prepared, route_ids_dtype=torch.int32)
    with kernel_resolution_guard("contiguous MXFP4 FC2 live routes"):
        for routes in (1, 3, 17):
            x = torch.randn(routes, intermediate, device=device, dtype=torch.bfloat16)
            ids = torch.arange(routes, dtype=torch.int32, device=device) % e
            weights = torch.linspace(0.125, 0.875, routes, device=device)
            output = torch.empty(routes, h, device=device, dtype=torch.bfloat16)

            def oracle():
                result = torch.zeros(routes, h, device=device)
                for expert in range(e):
                    mask = ids == expert
                    result[mask] = (
                        x[mask].float() @ reference[expert].float().T
                    ) * weights[mask, None]
                return result

            def run():
                run_w4a16_fc2_e8m0(
                    x, prepared, ids, weights, output=output, launcher=launcher
                )

            run()
            check(output, oracle())
            graph = torch.cuda.CUDAGraph()
            with capture_lifetime(), torch.cuda.graph(graph):
                run()
            x.mul_(0.5)
            ids.add_(1).remainder_(e)
            weights.mul_(0.75)
            expected = oracle()
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"]
                == allocations
            )
            check(output, expected)
            ids.fill_(-1)
            output.fill_(float("nan"))
            graph.replay()
            check(output, torch.zeros_like(output))
            graph.reset()

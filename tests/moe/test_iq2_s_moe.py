"""Native routed IQ2_S_XXS against independently decoded GGUF experts."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe._shared.kernels.w4a16.iq2_xs import prepare_iq2_s_moe_weights
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from tests._reference.helpers import require_b12x


def make_weight(e, n, k, seed, device, recipe="iq2_s"):
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(seed)
    block_bytes = {"iq2_s": 82, "iq2_xs": 74, "iq3_xxs": 98}[recipe]
    raw = rng.integers(0, 256, (e, n, k // 256, block_bytes), dtype=np.uint8)
    raw[..., :2] = (
        rng.normal(0, 0.001, (e, n, k // 256, 1)).astype(np.float16).view(np.uint8)
    )
    reference = (
        torch.from_numpy(
            gguf.dequantize(
                raw.reshape(e * n, -1),
                getattr(gguf.GGMLQuantizationType, recipe.upper()),
            )
        )
        .reshape(e, n, k)
        .bfloat16()
        .to(device)
    )
    return torch.from_numpy(raw).to(device), reference


@pytest.mark.parametrize("m,mode", [(1, "direct"), (8, "direct"), (32, "packed")])
@pytest.mark.parametrize("h,i", [(256, 256), (5120, 2304)])
@pytest.mark.parametrize("down", ["iq2_s", "mxfp4"])
def test_iq2_s_routed_swiglu_and_poisoned_graph_replay(m, mode, h, i, down):
    device = require_b12x()
    e, topk = 4, 2
    w13, ref13 = make_weight(e, i * 2, h, 32, device)
    if down == "mxfp4":
        gguf = pytest.importorskip("gguf")
        rng = np.random.default_rng(17)
        raw = rng.integers(0, 256, (e, h, i // 32, 17), dtype=np.uint8)
        raw[..., 0] = rng.integers(112, 121, raw.shape[:-1], dtype=np.uint8)
        w2 = torch.from_numpy(raw).to(device)
        ref2 = (
            torch.from_numpy(
                gguf.dequantize(raw.reshape(e * h, -1), gguf.GGMLQuantizationType.MXFP4)
            )
            .reshape(e, h, i)
            .to(device=device, dtype=torch.bfloat16)
        )
    else:
        w2, ref2 = make_weight(e, h, i, 17, device, down)
    prepared = prepare_iq2_s_moe_weights(
        w13,
        w2,
        hidden_size=h,
        intermediate_size=i,
        num_experts=e,
        activation="silu",
        w13_layout="w31",
        fc2_format=down if down != "iq2_s" else None,
    )
    assert (
        sum(
            t.numel() * t.element_size()
            for t in (prepared.w13, prepared.w13_scale, prepared.w2, prepared.w2_scale)
        )
        == w13.numel() + w2.numel()
    )
    del w13, w2
    x = torch.randn(m, h, device=device, dtype=torch.bfloat16) * 0.1
    ids = (
        torch.arange(m * topk, device=device, dtype=torch.int32).reshape(m, topk) % e
    ).contiguous()
    probs = torch.softmax(torch.randn(m, topk, device=device), dim=-1)
    out = torch.empty_like(x)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=device,
        block_size_m=8 if mode == "direct" else 16,
    )

    def reference():
        result = torch.zeros(m, h, device=device)
        for t in range(m):
            for r in range(topk):
                expert = int(ids[t, r])
                projected = (x[t].float() @ ref13[expert].float().T).bfloat16()
                gate, up = projected.chunk(2)
                activated = (F.silu(gate.float()) * up.float()).bfloat16()
                result[t] += (activated.float() @ ref2[expert].float().T) * probs[t, r]
        return result

    def run():
        return run_w4a16_moe(
            x,
            prepared,
            probs,
            ids,
            activation="silu",
            output=out,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_mode=mode,
            route_block_size_m=8 if mode == "direct" else 16,
        )

    def check(expected):
        assert torch.isfinite(out).all() and out.norm() > 0
        assert (out.float() - expected).norm() / expected.norm() < 0.02

    expected = reference()
    run()
    check(expected)
    # Warm all reachable launches, then prohibit cache misses during capture/replay.
    with kernel_resolution_guard("IQ2_S routed replay"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.mul_(0.5)
        expected = reference()
        out.fill_(float("nan"))
        buffers.fc1_c_tmp.fill_(float("nan"))
        buffers.fc2_c_tmp.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        check(expected)


@pytest.mark.parametrize(
    "route,capacity,counts",
    [
        ("direct", 8, (1, 3, 8)),
        ("packed", 32, (1, 3, 8, 17, 32)),
    ],
)
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("h,i", [(256, 256), (5120, 2304)])
@pytest.mark.parametrize(
    "recipe,down", [("iq2_s", None), ("iq2_s", "mxfp4")]
)
def test_public_iq2_s_prepared_routes(
    route, capacity, counts, mapped, h, i, recipe, down
):
    from types import SimpleNamespace
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession
    from benchmarks.benchmark_iq2_xs_moe import qualify_capacity

    device = require_b12x()
    global_e, topk = 8, 6
    expert_ids = (7, 0, 5) if mapped else tuple(range(global_e))
    e = len(expert_ids)
    raw13, ref13 = make_weight(e, i * 2, h, 66, device, recipe)
    if down == "mxfp4":
        gguf = pytest.importorskip("gguf")
        rng = np.random.default_rng(85)
        raw = rng.integers(0, 256, (e, h, i // 32, 17), dtype=np.uint8)
        raw[..., 0] = rng.integers(112, 121, raw.shape[:-1], dtype=np.uint8)
        raw2 = torch.from_numpy(raw).to(device)
        ref2 = (
            torch.from_numpy(
                gguf.dequantize(raw.reshape(e * h, -1), gguf.GGMLQuantizationType.MXFP4)
            )
            .reshape(e, h, i)
            .to(device=device, dtype=torch.bfloat16)
        )
    else:
        raw2, ref2 = make_weight(e, h, i, 85, device, down or recipe)
    plan = moe.plan_weights(
        source=moe.PackedSource(
            format=recipe, w13_layout="w31", fc2_format="fp4_e8m0_k32" if down == "mxfp4" else down
        ),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
    )
    bundle = moe.IQMixedWeights(raw13, raw2) if down == "iq3_xxs" else moe.IQMXFP4Weights(raw13, raw2) if down else moe.IQ2SWeights(raw13, raw2)
    experts = moe.prepare_weights(plan=plan, weights=bundle)
    assert experts._impl.representation.value.fc2_format == down
    if down == "iq3_xxs":
        assert plan.prepared_format.fc2_weights is moe.WeightEncoding.IQ3_XXS
        assert plan.prepared_format.fc2_scales is moe.ScaleEncoding.IQ3_XXS
        assert plan.prepared_format.fc2_packing is moe.WeightPacking.IQ3_XXS_COMPACT
    elif down:
        assert plan.prepared_format.fc2_weights is moe.WeightEncoding.FP4_E2M1
        assert plan.prepared_format.fc2_scales is moe.ScaleEncoding.E8M0_K32
        assert plan.prepared_format.fc2_packing is moe.WeightPacking.MMA_PACKED

    def expert_map(device):
        result = torch.full((global_e,), -1, dtype=torch.int32, device=device)
        result[list(expert_ids)] = torch.arange(e, dtype=torch.int32, device=device)
        return result

    layer = SimpleNamespace(
        hidden_size=h,
        intermediate_size=i,
        top_k=topk,
        route_num_experts=global_e,
        expert_ids=expert_ids,
        expert_map=expert_map,
    )

    def oracle(layer, inputs, *, activation):
        result = torch.zeros_like(inputs.output, dtype=torch.float32)
        mapping = {value: index for index, value in enumerate(expert_ids)}
        for t in range(inputs.x.shape[0]):
            for r in range(topk):
                index = mapping.get(int(inputs.ids[t, r]))
                if index is None:
                    continue
                projected = (inputs.x[t].float() @ ref13[index].float().T).bfloat16()
                gate, up = projected.chunk(2)
                activated = (F.silu(gate.float()) * up.float()).bfloat16()
                result[t] += (
                    activated.float() @ ref2[index].float().T
                ) * inputs.probabilities[t, r]
        return result

    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        results = qualify_capacity(
            layer,
            experts,
            session,
            capacity=capacity,
            counts=counts,
            route_mode=route,
            patterns=("balanced", "hot", "imbalanced"),
            repeats=1,
            launches=1,
            reference_fn=oracle,
        )
    assert len(results) == len(counts) * 3


def test_iq2_s_public_numeric_contract():
    from b12x.moe import fused_moe as moe

    def plan(*, mode="a16", h=5120, i=2304, packing=None):
        return moe.plan_weights(
            source=moe.PackedSource(format="iq2_s", w13_layout="w31"),
            activation=moe.ActivationSpec(
                mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=moe.MoEGeometry(
                num_experts=192, hidden_size=h, intermediate_size=i
            ),
            constraints=moe.WeightPlanConstraints(required_packing=packing),
        )

    value = plan()
    assert value.prepared_format.weights is moe.WeightEncoding.IQ2_S
    assert value.prepared_format.scales is moe.ScaleEncoding.IQ2_S
    assert value.prepared_format.packing is moe.WeightPacking.IQ2_S_COMPACT
    assert value._impl.w4a16_weight_layout == "iq2_s"
    assert value._impl.w4a16_scale_format == "iq2_s"
    for kwargs in (
        {"mode": "a8"},
        {"mode": "a4"},
        {"i": 1152},
        {"packing": "iq2_xs_compact"},
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            plan(**kwargs)
    wrong = torch.empty(1, 256, 1, 74, dtype=torch.uint8)
    with pytest.raises(ValueError, match="82"):
        moe.IQ2SWeights(wrong, wrong)


@pytest.mark.parametrize("recipe", ["iq2_xs", "iq2_s"])
def test_mixed_public_contract_rejects_wrong_bundles_and_recipes(recipe):
    from b12x.moe import fused_moe as moe

    source = moe.PackedSource(format=recipe, fc2_format="fp4_e8m0_k32")
    geometry = moe.MoEGeometry(num_experts=1, hidden_size=256, intermediate_size=256)
    plan = moe.plan_weights(
        source=source,
        geometry=geometry,
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
    )
    assert plan._impl.fc2_format == "mxfp4"
    for mode in ("a4", "a8"):
        with pytest.raises((ValueError, NotImplementedError)):
            moe.plan_weights(
                source=source,
                geometry=geometry,
                activation=moe.ActivationSpec(
                    mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16
                ),
            )
    block_bytes, bundle_type = (
        (74, moe.IQ2XSWeights) if recipe == "iq2_xs" else (82, moe.IQ2SWeights)
    )
    wrong = bundle_type(
        torch.zeros(1, 512, 1, block_bytes, dtype=torch.uint8),
        torch.zeros(1, 256, 1, block_bytes, dtype=torch.uint8),
    )
    with pytest.raises(TypeError, match="IQMXFP4Weights"):
        moe.prepare_weights(plan=plan, weights=wrong)
    same_plan = moe.plan_weights(
        source=moe.PackedSource(format=recipe),
        geometry=geometry,
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
    )
    mixed = moe.IQMXFP4Weights(wrong.w13, torch.zeros(1, 256, 8, 17, dtype=torch.uint8))
    with pytest.raises(TypeError):
        moe.prepare_weights(plan=same_plan, weights=mixed)
    with pytest.raises(ValueError, match="mixed source"):
        moe.PackedSource(format="fp4_e8m0_k32", fc2_format="fp4_e8m0_k32")
    with pytest.raises(ValueError, match="mixed source"):
        moe.PackedSource(format=recipe, fc2_format="modelopt_nvfp4")

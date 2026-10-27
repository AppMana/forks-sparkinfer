"""Streaming EP packing must match whole-partition preparation byte for byte."""

import pytest
import torch

from b12x.loader.gguf_experts import ExpertTensor, prepare_iq_expert_partition


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("gate_type,down_type", [
    ("IQ2_XS", "MXFP4"), ("IQ2_XXS", "IQ2_XXS"),
    ("IQ2_XXS", "IQ3_XXS"), ("IQ2_S", "MXFP4"),
    ("IQ2_S", "IQ3_S"), ("IQ3_XXS", "IQ3_S"),
    ("IQ3_XXS", "IQ3_XXS"), ("IQ3_S", "IQ3_S"),
])
def test_streamed_partition_preserves_prepared_bytes(tmp_path, gate_type, down_type):
    from b12x.moe._shared.kernels.w4a16.iq2_xs import _prepare_iq_moe_weights

    torch.manual_seed(51)
    sizes = {"IQ2_XS": 74, "IQ2_XXS": 66, "IQ2_S": 82,
             "IQ3_XXS": 98, "IQ3_S": 110, "MXFP4": 17}
    raw = []
    sources = []
    # Two K256 blocks expose per-expert base vs metadata concatenation errors.
    for name, kind in (("gate", gate_type), ("up", gate_type), ("down", down_type)):
        block = 32 if kind == "MXFP4" else 256
        data = torch.randint(0, 256, (3, 512, 512 // block, sizes[kind]), dtype=torch.uint8)
        if kind == "MXFP4":
            data[..., 0] = 127
        else:
            data[..., :2] = torch.tensor([0.01], dtype=torch.float16).view(torch.uint8)
        path = tmp_path / name
        # Sparse-file offset also covers 64-bit addressing in the streaming API.
        offset = (1 << 32) + 32
        with path.open("wb") as f:
            f.seek(offset)
            f.write(data.numpy().tobytes())
        sources.append(ExpertTensor(path, name, kind, offset, 3, 512, 512, block, sizes[kind]))
        raw.append(data)
    ids = [2, 0]
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession
    from benchmarks.benchmark_iq2_xs_moe import qualify_capacity
    from types import SimpleNamespace
    import gguf
    import torch.nn.functional as F

    plan = moe.plan_weights(
        source=moe.PackedSource(format=gate_type.lower(), w13_layout="w31",
                              fc2_format=None if gate_type == down_type else
                              "fp4_e8m0_k32" if down_type == "MXFP4" else down_type.lower()),
        activation=moe.ActivationSpec(mode="a16", nonlinearity="silu",
                                      io_dtype=torch.bfloat16, swiglu_limit=10.0),
        geometry=moe.MoEGeometry(num_experts=2, hidden_size=512, intermediate_size=512),
    )
    bundle = moe.GGUFExpertWeights(*sources, tuple(ids), torch.device("cuda"))
    experts = moe.prepare_weights(plan=plan, weights=bundle)
    actual = experts._impl.representation.value
    from dataclasses import replace
    with pytest.raises(ValueError, match="geometry"):
        moe.prepare_weights(plan=plan, weights=replace(bundle, expert_ids=(0,)))
    with pytest.raises(ValueError, match="format"):
        moe.prepare_weights(plan=plan, weights=replace(bundle, gate=replace(sources[0], quant_type="Q3_K")))
    with pytest.raises(ValueError, match="unique"):
        moe.prepare_weights(plan=plan, weights=replace(bundle, expert_ids=(0, 0)))
    assert experts._impl.w1_fp4.data_ptr() == actual.w13.data_ptr()
    assert experts._impl.w2_fp4.data_ptr() == actual.w2.data_ptr()
    expert_map = torch.tensor([1, -1, 0], dtype=torch.int32, device="cuda")
    expected = _prepare_iq_moe_weights(
        torch.cat((raw[0][ids], raw[1][ids]), dim=1).cuda(), raw[2][ids].cuda(),
        hidden_size=512, intermediate_size=512, num_experts=2,
        activation="silu", w13_layout="w31", recipe=gate_type.lower(),
        fc2_format=None if gate_type == down_type else down_type.lower(),
    )
    for name in ("w13", "w13_scale", "w2", "w2_scale", "w13_global_scale", "w2_global_scale"):
        a, b = getattr(actual, name), getattr(expected, name)
        assert a.shape == b.shape, name
        assert torch.equal(a, b), name
    assert expert_map.tolist() == [1, -1, 0]
    assert actual.num_experts == 2
    assert actual.source_format == expected.source_format
    assert actual.fc2_format == expected.fc2_format
    with pytest.raises(ValueError, match="unique"):
        prepare_iq_expert_partition(*sources, [0, 0], device="cuda")

    refs = []
    for data, source in zip(raw, sources, strict=True):
        decoded = gguf.dequantize(data.numpy().reshape(3 * 512, -1),
                                 getattr(gguf.GGMLQuantizationType, source.quant_type))
        refs.append(torch.from_numpy(decoded).reshape(3, 512, 512).bfloat16().cuda())
    layer = SimpleNamespace(hidden_size=512, intermediate_size=512, top_k=2,
                            route_num_experts=3, expert_ids=tuple(ids),
                            expert_map=lambda device: torch.tensor([1, -1, 0], dtype=torch.int32, device=device))

    def reference(layer, inputs, *, activation):
        out = torch.zeros_like(inputs.output, dtype=torch.float32)
        for t in range(inputs.x.shape[0]):
            for slot in range(layer.top_k):
                e = int(inputs.ids[t, slot])
                if e not in ids:
                    continue
                x = inputs.x[t].float()
                gate = (x @ refs[0][e].float().T).bfloat16().float().clamp(max=10)
                up = (x @ refs[1][e].float().T).bfloat16().float().clamp(-10, 10)
                mid = (F.silu(gate) * up).bfloat16().float()
                out[t] += (mid @ refs[2][e].float().T) * inputs.probabilities[t, slot]
        return out

    for route, capacity, counts in (("direct", 8, (1, 3, 8)), ("packed", 32, (1, 17, 32))):
        with PreparationSession(device=actual.w13.device, autotune=False, compile_workers=1) as session:
            results = qualify_capacity(layer, experts, session, capacity=capacity,
                                       counts=counts, route_mode=route,
                                       patterns=("balanced", "hot"), repeats=1,
                                       launches=1, reference_fn=reference)
        assert len(results) == len(counts) * 2

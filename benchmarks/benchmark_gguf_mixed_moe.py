"""Qualify sampled native IQ expert contributions, optionally with MXFP4 down.

One sampled expert remains resident. Global top-k 6 routes and an explicit
384-entry ownership map exercise nonlocal contributions. This is not a
full-layer latency benchmark or a full-model quality evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import gguf
import numpy as np
import torch
import torch.nn.functional as F

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe._shared.kernels.w4a16.iq2_xs import _prepare_iq_moe_weights
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from benchmarks.benchmark_iq2_xs_moe import capture_lifetime, check


def load_sample(parent, sample, device):
    data = (parent / Path(sample["path"]).name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sample["sha256"]:
        raise ValueError("GGUF sample SHA mismatch")
    k, n, _ = sample["dims"]
    group, size = {"IQ2_XS": (256, 74), "IQ3_XXS": (256, 98), "IQ2_XXS": (256, 66), "IQ2_S": (256, 82), "IQ3_S": (256, 110), "MXFP4": (32, 17)}[
        sample["type"]
    ]
    raw = np.frombuffer(data, dtype=np.uint8).copy().reshape(1, n, k // group, size)
    reference = gguf.dequantize(
        raw.reshape(n, -1), getattr(gguf.GGMLQuantizationType, sample["type"])
    )
    return torch.from_numpy(raw).to(device), torch.from_numpy(reference).to(
        device=device, dtype=torch.bfloat16
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-contract", type=Path)
    parser.add_argument("--stress-clamp", action="store_true",
                        help="Scale inputs until all three clamp boundaries are exercised")
    args = parser.parse_args()
    contract = (json.loads(args.activation_contract.read_text())
                if args.activation_contract else None)
    if args.stress_clamp and contract is None:
        parser.error("--stress-clamp requires --activation-contract")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(device)
    root = Path(__file__).resolve().parents[1]
    sources = (
        sorted((root / "b12x/moe").rglob("*.py"))
        + sorted((root / "b12x/_lib/quant").rglob("*.py"))
        + [root / "b12x/_lib/intrinsics.py", Path(__file__)]
    )
    identity = {
        "command": sys.argv,
        "scope": "one actual sampled expert contribution",
        "gpu": props.name,
        "uuid": str(props.uuid),
        "torch": torch.__version__,
        "source_sha256": hashlib.sha256(
            b"".join(
                str(p.relative_to(root)).encode() + p.read_bytes() for p in sources
            )
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for path in args.manifest:
            manifest = json.loads(path.read_text())
            if contract is not None and any(
                contract[key] != manifest[key] for key in ("repo", "revision", "variant")
            ):
                raise ValueError("Activation contract does not match the weight manifest")
            samples = {sample["name"]: sample for sample in manifest["samples"]}
            for down in samples.values():
                if "ffn_down_exps" not in down["name"]:
                    continue
                gate = samples[down["name"].replace("ffn_down_exps", "ffn_gate_exps")]
                up = samples[down["name"].replace("ffn_down_exps", "ffn_up_exps")]
                if gate["type"] != up["type"] or gate["type"] not in {
                    "IQ2_XS",
                    "IQ3_XXS",
                    "IQ2_XXS",
                    "IQ2_S",
                    "IQ3_S",
                }:
                    continue
                if down["type"] not in {gate["type"], "MXFP4"} and not (gate["type"] == "IQ2_XXS" and down["type"] == "IQ3_XXS") and not (gate["type"] in {"IQ2_S", "IQ3_XXS"} and down["type"] == "IQ3_S"):
                    continue
                layer = int(down["name"].split(".")[1])
                limit = (float(contract["values"]["deepseek41.swiglu_clamp_exp"][layer])
                         if contract is not None else None)
                if limit is not None and (not np.isfinite(limit) or limit <= 0):
                    raise ValueError("SwiGLU clamp must be finite and positive")
                g, gr = load_sample(path.parent, gate, device)
                u, ur = load_sample(path.parent, up, device)
                d, dr = load_sample(path.parent, down, device)
                i, h = gr.shape
                prepared = _prepare_iq_moe_weights(
                    torch.cat((g, u), dim=1),
                    d,
                    hidden_size=h,
                    intermediate_size=i,
                    num_experts=1,
                    activation="silu",
                    w13_layout="w31",
                    recipe=gate["type"].lower(),
                    fc2_format=down["type"].lower() if down["type"] != gate["type"] else None,
                )
                del g, u, d
                payload = sum(
                    t.numel() * t.element_size()
                    for t in (
                        prepared.w13,
                        prepared.w13_scale,
                        prepared.w2,
                        prepared.w2_scale,
                    )
                )
                assert payload == sum(
                    sample["downloaded_bytes"] for sample in (gate, up, down)
                )
                for m, mode in (
                    (1, "direct"),
                    (8, "direct"),
                    (32, "packed"),
                    (128, "packed"),
                ):
                    torch.manual_seed(615 + m)
                    x = torch.randn(m, h, dtype=torch.bfloat16, device=device) * 0.1
                    ids = (
                        torch.arange(6, device=device, dtype=torch.int32)[None, :]
                        .expand(m, 6)
                        .contiguous()
                    )
                    weights = torch.softmax(torch.randn(m, 6, device=device), dim=-1)
                    expert_map = torch.full(
                        (384,), -1, dtype=torch.int32, device=device
                    )
                    expert_map[0] = 0
                    result = torch.empty_like(x)
                    block = 8 if mode == "direct" else 32
                    buffers = make_w4a16_packed_buffers(
                        prepared,
                        m=m,
                        topk=6,
                        dtype=torch.bfloat16,
                        device=device,
                        route_num_experts=384,
                        block_size_m=block,
                    )

                    clamp_counts = {}
                    if args.stress_clamp:
                        # Keep the same real weights; amplify only the probe input.
                        # Require positive gate and both up boundaries, including
                        # after replay halves x. Negative gate stays unbounded.
                        for _ in range(16):
                            gy = x.float() @ gr.float().T
                            uy = x.float() @ ur.float().T
                            clamp_counts = {
                                "gate_above": int((gy > 2 * limit).sum()),
                                "gate_below": int((gy < -2 * limit).sum()),
                                "up_above": int((uy > 2 * limit).sum()),
                                "up_below": int((uy < -2 * limit).sum()),
                            }
                            if all(clamp_counts.values()):
                                break
                            x.mul_(2)
                        else:
                            raise AssertionError("Probe failed to exercise clamp boundaries")

                    def oracle():
                        gate_y = (x.float() @ gr.float().T).bfloat16()
                        up_y = (x.float() @ ur.float().T).bfloat16()
                        if limit is not None:
                            gate_y = gate_y.clamp(max=limit)
                            up_y = up_y.clamp(min=-limit, max=limit)
                        mid = (F.silu(gate_y.float()) * up_y.float()).bfloat16()
                        route_weight = torch.where(ids == 0, weights, 0).sum(dim=-1)
                        return (mid.float() @ dr.float().T) * route_weight[:, None]

                    def run():
                        return run_w4a16_moe(
                            x,
                            prepared,
                            weights,
                            ids,
                            activation="silu",
                            swiglu_limit=limit,
                            expert_map=expert_map,
                            output=result,
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
                            route_block_size_m=block,
                        )

                    run()
                    metrics = check(result, oracle())
                    with kernel_resolution_guard("actual GGUF mixed expert replay"):
                        graph = torch.cuda.CUDAGraph()
                        with capture_lifetime(), torch.cuda.graph(graph):
                            run()
                        x.mul_(0.5)
                        weights.mul_(0.75)
                        ids[::2, 0] = 7
                        expected = oracle()
                        result.fill_(float("nan"))
                        buffers.fc1_c_tmp.fill_(float("nan"))
                        buffers.fc2_c_tmp.fill_(float("nan"))
                        allocations = torch.cuda.memory_stats(device)[
                            "allocation.all.allocated"
                        ]
                        graph.replay()
                        torch.cuda.synchronize(device)
                        assert (
                            torch.cuda.memory_stats(device)["allocation.all.allocated"]
                            == allocations
                        )
                        metrics.update(
                            {
                                "replay_" + k: v
                                for k, v in check(result, expected).items()
                            }
                        )
                        expert_map.fill_(-1)
                        result.fill_(float("nan"))
                        graph.replay()
                        check(result, torch.zeros_like(result))
                        graph.reset()
                    record = {
                        **identity,
                        "repo": manifest["repo"],
                        "revision": manifest["revision"],
                        "variant": manifest["variant"],
                        "tensor": down["name"],
                        "fc1": gate["type"],
                        "fc2": down["type"],
                        "swiglu_limit": limit,
                        "activation_contract_sha256": (
                            hashlib.sha256(args.activation_contract.read_bytes()).hexdigest()
                            if contract is not None else None
                        ),
                        "stress_clamp": args.stress_clamp,
                        "clamp_boundary_counts_before_replay": clamp_counts,
                        "m": m,
                        "topk": 6,
                        "route_mode": mode,
                        "resident_experts": 1,
                        "global_experts": 384,
                        "resident_payload_bytes": payload,
                        "sample_sha256": [s["sha256"] for s in (gate, up, down)],
                        **metrics,
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(record, flush=True)


if __name__ == "__main__":
    main()

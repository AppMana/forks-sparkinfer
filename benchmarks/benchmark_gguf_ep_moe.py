"""Measure full resident GGUF EP partitions through the public prepared API.

Synthetic routes isolate kernel policy; these are not end-to-end serving times.
Every timed shape first checks sampled rows against independent GGUF decoding.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import gguf
import torch
import torch.nn.functional as F

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.loader.gguf_experts import index_experts
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from benchmarks.benchmark_iq2_xs_moe import capture_lifetime, check
from benchmarks.common import nvidia_smi_gpu_mode_snapshot


def reference(sources, x, ids, probabilities, first, count):
    """Decode only the selected experts, one projection at a time."""
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in ids.unique().tolist():
        if not first <= expert < first + count:
            continue
        rows, slots = torch.where(ids == expert)
        values = []
        for source in sources:
            raw = source.read_expert(expert).numpy().reshape(source.rows, -1)
            decoded = gguf.dequantize(
                raw, getattr(gguf.GGMLQuantizationType, source.quant_type)
            )
            weight = torch.from_numpy(decoded).to(device=x.device, dtype=torch.bfloat16)
            if len(values) < 2:
                values.append((x[rows].float() @ weight.float().T).bfloat16())
            else:
                gate, up = values
                mid = (
                    F.silu(gate.float().clamp(max=10)) * up.float().clamp(-10, 10)
                ).bfloat16()
                contribution = mid.float() @ weight.float().T
                result.index_add_(
                    0, rows, contribution * probabilities[rows, slots, None]
                )
            del weight
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-commit", help="Source base commit for synced trees without .git"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[0, 2, 5, 15, 20, 25, 30]
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--rank", type=int, choices=[0, 1], default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if min(*args.tokens, args.iterations, args.trials) <= 0:
        parser.error("Token counts, iterations and trials must be positive")
    manifest = args.checkpoint / "checkpoint-verified.json"
    records = json.loads(manifest.read_text())
    assert len(records) == 7
    paths = []
    for record in records:
        assert record["revision"] == "60ad3651641b5c97679c807e064ab1d7a3ca1455"
        path = args.checkpoint / record["file"]
        assert path.stat().st_size == record["bytes"]
        assert (
            json.loads((args.checkpoint / (path.name + ".verified.json")).read_text())
            == record
        )
        paths.append(path)
    indexed = index_experts(paths)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    source_files = sorted((root / "b12x").rglob("*.py")) + [Path(__file__)]
    identity = dict(
        command=sys.argv,
        commit=args.source_commit
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        worktree=str(root),
        torch=torch.__version__,
        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
        source_sha256=hashlib.sha256(
            b"".join(
                str(p.relative_to(root)).encode() + p.read_bytes() for p in source_files
            )
        ).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        scope="Full 192-expert EP partition; synthetic routes; routing and reduction included; no TP communication",
        rank=args.rank,
    )
    first, count, global_experts, h, intermediate, topk = (
        args.rank * 192,
        192,
        384,
        5120,
        2304,
        6,
    )
    mapping = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    mapping[first : first + count] = torch.arange(
        count, dtype=torch.int32, device=device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for layer in args.layers:
            sources = tuple(
                indexed[f"blk.{layer}.ffn_{part}_exps.weight"]
                for part in ("gate", "up", "down")
            )
            gate, up, down = sources
            assert (gate.experts, gate.columns, gate.rows) == (
                global_experts,
                h,
                intermediate,
            )
            assert gate.quant_type == up.quant_type
            fc2 = (
                None
                if down.quant_type == gate.quant_type
                else (
                    "fp4_e8m0_k32"
                    if down.quant_type == "MXFP4"
                    else down.quant_type.lower()
                )
            )
            plan = moe.plan_weights(
                source=moe.PackedSource(
                    format=gate.quant_type.lower(), fc2_format=fc2, w13_layout="w31"
                ),
                activation=moe.ActivationSpec(
                    mode="a16",
                    nonlinearity="silu",
                    io_dtype=torch.bfloat16,
                    swiglu_limit=10,
                ),
                geometry=moe.MoEGeometry(
                    num_experts=count, hidden_size=h, intermediate_size=intermediate
                ),
            )
            experts = moe.prepare_weights(
                plan=plan,
                weights=moe.GGUFExpertWeights(
                    *sources, tuple(range(first, first + count)), device
                ),
            )
            for m in args.tokens:
                torch.manual_seed(182 + m)
                x = torch.randn(m, h, device=device, dtype=torch.bfloat16)
                ids = torch.empty(m, topk, device=device, dtype=torch.int32)
                probabilities = torch.softmax(
                    torch.randn(m, topk, device=device), dim=-1
                )
                out = torch.empty_like(x)
                modes = ["direct", "packed"] if m <= 8 else ["packed"]
                with PreparationSession(
                    device=device, autotune=False, compile_workers=1
                ) as session:
                    states = {}
                    for mode in modes:
                        declaration = moe.plan_execution(
                            experts=experts,
                            capacity=moe.ExecutionCapacity(
                                max_tokens=m,
                                top_k=topk,
                                route_num_experts=global_experts,
                            ),
                            override=moe.MoeDecodeConfig(
                                backend="w4a16",
                                route_planner="internal",
                                max_active_clusters=None,
                                w4a16_route_mode=mode,
                            ),
                        )

                        def factory(state):
                            scratch = tuple(
                                torch.empty(s.shape, dtype=s.dtype, device=device)
                                for s in state.scratch.scratch_specs()
                            )
                            bound = state.bind(
                                scratch=scratch,
                                a=x,
                                topk_ids=ids,
                                topk_weights=probabilities,
                                output=out,
                                route_expert_map=mapping,
                            )
                            return PreparedCall(
                                run=bound.run, output=out, owners=(scratch, bound)
                            )

                        ids.fill_(first)
                        session.prepare(
                            (
                                declaration.request(
                                    name=f"ep-{mode}",
                                    prepare_call=factory,
                                    benchmark_call=factory,
                                ),
                            )
                        )
                        states[mode] = factory(
                            require_prepared(declaration, "moe.decode")
                        )
                    for pattern in ("balanced", "hot"):
                        row = torch.arange(m, device=device)[:, None]
                        slot = torch.arange(topk, device=device)[None, :]
                        # Mix both EP owners, including the rank under test.
                        ids.copy_(
                            (
                                (row * topk if pattern == "balanced" else 0)
                                + slot * 64
                                + first
                            )
                            % global_experts
                        )
                        samples = torch.tensor(
                            sorted({0, m // 2, m - 1}), device=device
                        )
                        expected = reference(
                            sources,
                            x[samples],
                            ids[samples],
                            probabilities[samples],
                            first,
                            count,
                        )
                        graphs, metrics = {}, {}
                        for mode in modes:
                            call = states[mode]
                            for _ in range(3):
                                call.run()
                            metrics[mode] = check(out[samples], expected)
                            with (
                                kernel_resolution_guard("GGUF EP timing"),
                                capture_lifetime(),
                                session.capture(),
                            ):
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    call.run()
                                out.fill_(float("nan"))
                                allocated = torch.cuda.memory_stats()[
                                    "allocation.all.allocated"
                                ]
                                graph.replay()
                                torch.cuda.synchronize()
                                assert (
                                    torch.cuda.memory_stats()[
                                        "allocation.all.allocated"
                                    ]
                                    == allocated
                                )
                                metrics[mode].update(
                                    {
                                        "replay_" + k: v
                                        for k, v in check(
                                            out[samples], expected
                                        ).items()
                                    }
                                )
                            graphs[mode] = graph
                        before = nvidia_smi_gpu_mode_snapshot()
                        timings = {mode: [] for mode in modes}
                        for trial in range(args.trials):
                            for mode in modes[:: 1 if trial % 2 == 0 else -1]:
                                start, end = (
                                    torch.cuda.Event(enable_timing=True),
                                    torch.cuda.Event(enable_timing=True),
                                )
                                start.record()
                                for _ in range(args.iterations):
                                    graphs[mode].replay()
                                end.record()
                                end.synchronize()
                                timings[mode].append(
                                    start.elapsed_time(end) / args.iterations
                                )
                        record = dict(
                            identity,
                            layer=layer,
                            formats=[gate.quant_type, down.quant_type],
                            tokens=m,
                            pattern=pattern,
                            correctness=metrics,
                            milliseconds=timings,
                            gpu_before=before,
                            gpu_after=nvidia_smi_gpu_mode_snapshot(),
                        )
                        output.write(json.dumps(record) + "\n")
                        output.flush()
                        print(json.dumps(record), flush=True)
                        for graph in graphs.values():
                            graph.reset()
            del experts
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

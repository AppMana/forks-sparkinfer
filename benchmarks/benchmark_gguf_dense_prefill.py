"""Compare the GB10 GGUF prefill policy with the previous 16-row schedule."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess

import gguf
import torch
import torch.nn.functional as F

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.gemm.blockscaled._tuning import BlockscaledConfig
from b12x.loader.gguf_dense import index_dense, prepare_dense_partition
from b12x.preparation import PreparationSession, PreparedCall, require_prepared


def main(args):
    torch.cuda.set_device(0)
    torch.manual_seed(20261026)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = json.loads((args.checkpoint / "checkpoint-verified.json").read_text())
    assert all(
        r["revision"] == "60ad3651641b5c97679c807e064ab1d7a3ca1455" for r in records
    )
    sources = index_dense([args.checkpoint / r["file"] for r in records])
    cases = [
        ("blk.27.ffn_gate_shexp.weight", 1280, 5120),
        ("blk.27.ffn_down_shexp.weight", 5120, 1280),
        ("blk.26.attn_q_a.weight", 1280, 5120),
        ("blk.26.attn_output_a.weight", 1024, 4096),
        ("blk.1.engram_kv.weight", 12800, 6144),
        ("blk.26.attn_q_b.weight", 16384, 1280),
        ("blk.26.attn_output_b.weight", 5120, 4096),
    ]
    flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    identity = {
        "gpu": str(torch.cuda.get_device_properties(0)),
        "b12x_path": str(Path(blockscaled.__file__).resolve()),
        "gpu_before": subprocess.check_output(["nvidia-smi", "-q"], text=True),
        "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_manifest_sha256": hashlib.sha256(
            (args.checkpoint / "checkpoint-verified.json").read_bytes()
        ).hexdigest(),
        "scope": "Planned1024; dynamic decode/prefill counts; same checkpoint weights; cold128MiB eviction and warm graph replay. No full-model throughput claim.",
        "source_commit": args.source_commit,
    }
    with args.output.open("w") as report:
        for name, n, k in cases:
            source = sources[name]
            raw = source.read_rows(0, n, columns=k)
            reference = torch.from_numpy(
                gguf.dequantize(
                    raw.numpy().reshape(n, -1),
                    getattr(gguf.GGMLQuantizationType, source.quant_type),
                )
            ).to(device="cuda", dtype=torch.bfloat16)
            weight = prepare_dense_partition(source, device="cuda", rows=n, columns=k)
            x = torch.randn(1024, k, device="cuda", dtype=torch.bfloat16) * 0.1
            output = torch.empty(1024, n, device="cuda", dtype=torch.bfloat16)
            states = {}
            configs = {"baseline": (16, 64, 64), "policy": (None, None, None)}
            with PreparationSession(
                device=x.device, autotune=False, compile_workers=1
            ) as session:
                for capacity, (tm, tn, tk) in configs.items():
                    plan = blockscaled.plan(
                        blockscaled.query_from_call(x, weight, out=output),
                        override=None
                        if tm is None
                        else BlockscaledConfig(
                            mode="a16", tile_m=tm, tile_n=tn, tile_k=tk, split_k=1
                        ),
                    )

                    def factory(state, capacity=capacity):
                        return PreparedCall(
                            run=lambda: state.run(
                                x, weight.values, weight.metadata, None, out=output
                            )
                        )

                    session.prepare(
                        (
                            plan.request(
                                name="dense prefill row tile comparison",
                                prepare_call=factory,
                            ),
                        )
                    )
                    states[capacity] = require_prepared(
                        plan, "gemm.blockscaled_precision"
                    )
                session.freeze()
            with kernel_resolution_guard("dense prefill row tile benchmark"):
                for m in (1, 8, 127, 128, 512, 1024):
                    expected = x[:m].float() @ reference.float().T
                    graphs, errors = {}, {}
                    baseline_actual = None
                    for capacity, state in states.items():
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            state.run(
                                x[:m],
                                weight.values,
                                weight.metadata,
                                None,
                                out=output[:m],
                            )
                        output.fill_(float("nan"))
                        allocated = torch.cuda.memory_stats()[
                            "allocation.all.allocated"
                        ]
                        graph.replay()
                        torch.cuda.synchronize()
                        assert (
                            allocated
                            == torch.cuda.memory_stats()["allocation.all.allocated"]
                        )
                        actual = output[:m].float()
                        cosine = F.cosine_similarity(
                            actual.flatten(), expected.flatten(), dim=0
                        ).item()
                        relative_l2 = (
                            (actual - expected).norm() / expected.norm()
                        ).item()
                        assert (
                            torch.isfinite(actual).all()
                            and cosine > 0.9999
                            and relative_l2 < 0.01
                        )
                        if baseline_actual is None:
                            baseline_actual = actual.clone()
                        errors[capacity] = {
                            "cosine": cosine,
                            "relative_l2": relative_l2,
                            "equal_to_default": torch.equal(actual, baseline_actual),
                            "max_abs_diff_from_default": (actual - baseline_actual)
                            .abs()
                            .max()
                            .item(),
                        }
                        graphs[capacity] = graph
                    # Capture event records with the work so host enqueue gaps
                    # cannot inflate single-kernel cold-cache measurements.
                    timed = {}
                    for capacity, state in states.items():
                        for mode, iterations in (("cold_ms", 1), ("warm_ms", 10)):
                            begin = torch.cuda.Event(enable_timing=True, external=True)
                            end = torch.cuda.Event(enable_timing=True, external=True)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                if mode == "cold_ms":
                                    flush.zero_()
                                begin.record()
                                for _ in range(iterations):
                                    state.run(
                                        x[:m],
                                        weight.values,
                                        weight.metadata,
                                        None,
                                        out=output[:m],
                                    )
                                end.record()
                            graph.replay()
                            end.synchronize()
                            timed[capacity, mode] = (graph, begin, end, iterations)
                    measurements = {c: {"cold_ms": [], "warm_ms": []} for c in states}
                    for trial in range(10):
                        for capacity in (
                            tuple(states) if trial % 2 == 0 else tuple(reversed(states))
                        ):
                            for mode in ("cold_ms", "warm_ms"):
                                graph, begin, end, iterations = timed[capacity, mode]
                                graph.replay()
                                end.synchronize()
                                measurements[capacity][mode].append(
                                    begin.elapsed_time(end) / iterations
                                )
                    for capacity in states:
                        measurements[capacity].update(
                            {
                                key + "_median": statistics.median(
                                    measurements[capacity][key]
                                )
                                for key in ("cold_ms", "warm_ms")
                            }
                        )
                    row = {
                        **identity,
                        "gpu_after": subprocess.check_output(
                            ["nvidia-smi", "-q"], text=True
                        ),
                        "tensor": name,
                        "format": source.quant_type,
                        "m": m,
                        "n": n,
                        "k": k,
                        "configs": configs,
                        "correctness": errors,
                        "measurements": measurements,
                    }
                    report.write(json.dumps(row) + "\n")
                    report.flush()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in ("tensor", "m", "correctness", "measurements")
                            }
                        ),
                        flush=True,
                    )
                    for graph in graphs.values():
                        graph.reset()
                    for graph, _, _, _ in timed.values():
                        graph.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

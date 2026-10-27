"""Qualify actual GGUF MXFP4 down projections through the native FC2 kernel.

Timings are warm graph measurements of FC2 alone, not full MoE/prefill results.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gguf
import numpy as np
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.loader.gguf_mxfp4 import prepare_gguf_mxfp4_fc2
from b12x.moe.fused_moe._impl import prewarm_w4a16_fc2_e8m0, run_w4a16_fc2_e8m0
from benchmarks.benchmark_iq2_xs_moe import capture_lifetime, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--routes", type=int, nargs="+", default=[1, 6, 48, 192])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.routes) <= 0:
        parser.error("route counts must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    source_files = [
        root / "b12x/loader/gguf_mxfp4.py",
        root / "b12x/_lib/intrinsics.py",
        *sorted((root / "b12x/moe").rglob("*.py")),
    ]
    props = torch.cuda.get_device_properties(device)
    provenance = {
        "command": sys.argv,
        "worktree": str(root),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if (root / ".git").exists()
        else "rsynced-worktree",
        "source_sha256": hashlib.sha256(
            b"".join(
                str(p.relative_to(root)).encode() + p.read_bytes() for p in source_files
            )
        ).hexdigest(),
        "gpu": props.name,
        "uuid": str(props.uuid),
        "cc": [props.major, props.minor],
        "torch": torch.__version__,
        "cache_mode": "warm",
        "scope": "native FC2 projection only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for manifest_path in args.manifest:
            manifest = json.loads(manifest_path.read_text())
            for sample in manifest["samples"]:
                if sample["type"] != "MXFP4" or "ffn_down_exps" not in sample["name"]:
                    continue
                path = manifest_path.parent / Path(sample["path"]).name
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != sample["sha256"]:
                    raise ValueError(f"sample SHA mismatch: {path}")
                k, n, _ = sample["dims"]
                e = sample["experts"]
                raw = (
                    np.frombuffer(data, dtype=np.uint8)
                    .copy()
                    .reshape(e, n, k // 32, 17)
                )
                reference = (
                    torch.from_numpy(
                        gguf.dequantize(
                            raw.reshape(e * n, -1), gguf.GGMLQuantizationType.MXFP4
                        )
                    )
                    .reshape(e, n, k)
                    .to(device=device, dtype=torch.bfloat16)
                )
                prepared = prepare_gguf_mxfp4_fc2(torch.from_numpy(raw).to(device))
                resident_bytes = (
                    prepared.w2.numel() * prepared.w2.element_size()
                    + prepared.w2_scale.numel() * prepared.w2_scale.element_size()
                )
                assert resident_bytes == len(data)
                launcher = prewarm_w4a16_fc2_e8m0(prepared, route_ids_dtype=torch.int32)
                with kernel_resolution_guard("GGUF MXFP4 FC2 qualification"):
                    for routes in args.routes:
                        torch.manual_seed(77 + routes)
                        x = torch.randn(routes, k, dtype=torch.bfloat16, device=device)
                        ids = torch.arange(routes, dtype=torch.int32, device=device) % e
                        probabilities = torch.linspace(
                            0.125, 0.875, routes, device=device
                        )
                        actual = torch.empty(
                            routes, n, dtype=torch.bfloat16, device=device
                        )

                        def oracle():
                            result = torch.zeros(routes, n, device=device)
                            for expert in range(e):
                                selected = ids == expert
                                result[selected] = (
                                    x[selected].float() @ reference[expert].float().T
                                ) * probabilities[selected, None]
                            return result

                        def run():
                            return run_w4a16_fc2_e8m0(
                                x,
                                prepared,
                                ids,
                                probabilities,
                                output=actual,
                                launcher=launcher,
                            )

                        run()
                        metrics = check(actual, oracle())
                        graph = torch.cuda.CUDAGraph()
                        with capture_lifetime(), torch.cuda.graph(graph):
                            run()
                        x.mul_(0.5)
                        probabilities.mul_(0.75)
                        expected = oracle()
                        actual.fill_(float("nan"))
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
                                "replay_" + key: value
                                for key, value in check(actual, expected).items()
                            }
                        )
                        samples = []
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        for _ in range(10):
                            start.record()
                            for _ in range(20):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            samples.append(start.elapsed_time(end) * 1000 / 20)
                        ids.fill_(-1)
                        actual.fill_(float("nan"))
                        graph.replay()
                        check(actual, torch.zeros_like(actual))
                        graph.reset()
                        record = {
                            **provenance,
                            "repo": manifest["repo"],
                            "revision_weights": manifest["revision"],
                            "variant": manifest["variant"],
                            "tensor": sample["name"],
                            "payload_sha256": sample["sha256"],
                            "routes": routes,
                            "h": n,
                            "i": k,
                            "resident_bytes": resident_bytes,
                            "scale_min": int(raw[..., 0].min()),
                            "scale_max": int(raw[..., 0].max()),
                            "graph_us": samples,
                            **metrics,
                        }
                        output.write(json.dumps(record) + "\n")
                        output.flush()
                        print(sample["name"], routes, metrics, flush=True)


if __name__ == "__main__":
    main()

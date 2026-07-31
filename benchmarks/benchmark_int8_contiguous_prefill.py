#!/usr/bin/env python3
"""Compare native SparkInfer and vLLM Triton INT8 contiguous logits."""

from __future__ import annotations

import argparse
import statistics

import torch

from sparkinfer.attention.nsa_indexer.contiguous_kernel import (
    run_contiguous_logits_kernel,
)


HEADS = 64
HEAD_DIM = 128


def _time_ms(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _inputs(rows: int, context: int) -> tuple[torch.Tensor, ...]:
    q = torch.full((rows, HEADS, HEAD_DIM), -1, dtype=torch.int8, device="cuda")
    k = torch.full((context, HEAD_DIM), -1, dtype=torch.int8, device="cuda")
    weights = torch.full((rows, HEADS), 1.0 / HEADS, dtype=torch.float32, device="cuda")
    scales = torch.linspace(0.005, 0.02, context, dtype=torch.float32, device="cuda")
    k_start = torch.zeros(rows, dtype=torch.int32, device="cuda")
    k_end = torch.full((rows,), context, dtype=torch.int32, device="cuda")
    return q, k, weights, scales, k_start, k_end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["8192x2048", "1024x16384", "1024x65536"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from vllm.models.deepseek_v4.nvidia_imma.triton_kernels import (
        mqa_logits_workspace_triton,
    )

    for shape in args.shapes:
        rows, context = (int(value) for value in shape.lower().split("x", 1))
        q, k, weights, scales, k_start, k_end = _inputs(rows, context)

        def native():
            return run_contiguous_logits_kernel(
                q_fp8=q,
                weights=weights,
                k_quant=k,
                k_scale=scales,
                k_start=k_start,
                k_end=k_end,
                preinitialize_invalid_logits=False,
            )

        def triton():
            return mqa_logits_workspace_triton(
                q,
                (k, scales),
                weights,
                k_start,
                k_end,
                qk_int8=True,
            )

        native_out = native()
        triton_out = triton()
        expected = (HEAD_DIM * scales).expand(rows, context)
        torch.cuda.synchronize()
        torch.testing.assert_close(native_out, expected, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(triton_out, expected, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(native_out, triton_out, atol=2e-4, rtol=2e-4)

        native_ms = _time_ms(native, args.warmup, args.iters)
        triton_ms = _time_ms(triton, args.warmup, args.iters)
        native_median = statistics.median(native_ms)
        triton_median = statistics.median(triton_ms)
        print(
            f"rows={rows} context={context} correctness=pass "
            f"native_median_ms={native_median:.4f} "
            f"triton_median_ms={triton_median:.4f} "
            f"native_over_triton={native_median / triton_median:.4f}x "
            f"native_raw_ms={native_ms} triton_raw_ms={triton_ms}"
        )


if __name__ == "__main__":
    main()

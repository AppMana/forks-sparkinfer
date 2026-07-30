#!/usr/bin/env python3
"""Compare native SparkInfer and vLLM Triton INT8 paged indexer logits."""

from __future__ import annotations

import argparse
import statistics

import torch

from sparkinfer.attention.nsa_indexer.kernel import run_paged_logits_kernel


PAGE_SIZE = 64
HEADS = 64
HEAD_DIM = 128
TOKEN_BYTES = HEAD_DIM + 4


def _time_us(fn, warmup: int, iters: int) -> list[float]:
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
        samples.append(start.elapsed_time(end) * 1_000)
    return samples


def _inputs(context: int, rows: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    pages = (context + PAGE_SIZE - 1) // PAGE_SIZE
    q = torch.full((rows, HEADS, HEAD_DIM), -1, dtype=torch.int8, device=device)
    weights = (
        torch.arange(1, HEADS + 1, dtype=torch.float32, device=device) / HEADS
    ).expand(rows, -1)
    page_table = torch.arange(pages, dtype=torch.int32, device=device).expand(
        rows, -1
    )
    seqlens = torch.full((rows,), context, dtype=torch.int32, device=device)

    interleaved = torch.empty(
        (pages, PAGE_SIZE, 1, TOKEN_BYTES), dtype=torch.uint8, device=device
    )
    interleaved[..., :HEAD_DIM].view(torch.int8).fill_(-1)
    interleaved[..., HEAD_DIM:].view(torch.float32).fill_(0.01)

    planar = torch.empty_like(interleaved)
    planar_rows = planar.view(pages, -1)
    planar_rows[:, : PAGE_SIZE * HEAD_DIM].view(torch.int8).fill_(-1)
    planar_rows[:, PAGE_SIZE * HEAD_DIM :].view(torch.float32).fill_(0.01)
    return q, weights, page_table, seqlens, interleaved, planar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 8192, 10769])
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from vllm.models.deepseek_v4.nvidia_imma import triton_kernels

    triton_kernels.indexer_cache_is_int8 = lambda: True
    for context in args.contexts:
        q, weights, page_table, seqlens, interleaved, planar = _inputs(
            context, args.rows
        )
        expected = HEAD_DIM * 0.01 * float(weights[0].sum())

        def native():
            return run_paged_logits_kernel(
                q_fp8=q,
                weights=weights,
                index_k_cache=interleaved,
                real_page_table=page_table,
                seqlens_per_query=seqlens,
                page_size=PAGE_SIZE,
            )

        def triton():
            return triton_kernels.fp8_paged_mqa_logits_triton(
                q[:, None],
                planar,
                weights,
                seqlens[:, None],
                page_table,
                context,
            )

        native_out = native()
        triton_out = triton()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            native_out[:, :context],
            torch.full_like(native_out[:, :context], expected),
            rtol=2e-4,
            atol=2e-4,
        )
        torch.testing.assert_close(
            triton_out,
            torch.full_like(triton_out, expected),
            rtol=2e-4,
            atol=2e-4,
        )
        native_us = _time_us(native, args.warmup, args.iters)
        triton_us = _time_us(triton, args.warmup, args.iters)
        native_median = statistics.median(native_us)
        triton_median = statistics.median(triton_us)
        print(
            f"context={context} rows={args.rows} correctness=pass "
            f"native_median_us={native_median:.2f} "
            f"triton_median_us={triton_median:.2f} "
            f"native_over_triton={native_median / triton_median:.4f}x "
            f"native_raw_us={native_us} triton_raw_us={triton_us}"
        )


if __name__ == "__main__":
    main()

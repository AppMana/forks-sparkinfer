"""Drive the launch paths with fabricated arguments, on a machine with no GPU.

Each entry below calls the *real* launch function -- the same one production
calls -- with :class:`FakeCudaTensor` metadata in place of live tensors.  The
launch function computes its ``KernelCompileSpec`` and its compile arguments
exactly as it always does; ``sparkinfer._lib.compiler.launch`` then compiles and
stops short of executing, because ``compile_only()`` is in effect.

Nothing here reimplements a compile path.  If a kernel's spec derivation
changes, this follows it automatically, and if a configuration below is wrong
it fails here at build time rather than producing an object that keys onto
something production never asks for.

Every configuration is attempted independently and failures are reported rather
than raised, so one bad entry costs one kernel's worth of AOT and not the whole
build.  The kernels it misses still compile on first use.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .aot_args import FakeCudaTensor, compile_only


@dataclass(frozen=True)
class Deployment:
    """The model geometry and serving shape an AOT build is targeting.

    Defaults are the deployed DSV4 checkpoints on the Sparks (both
    ``appmana/deepseek-v4-nvfp4-fp8`` and ``appmana/deepseek-v4-int4-int8``
    carry identical attention and MoE geometry) at the TP2 serving config in
    ``tools/ampere/dgx_spark_serve_dsv4_tp2.sh``.  See
    ``sparkinfer/_lib/aot_matrix.py`` for the full derivation.
    """

    num_heads: int = 32  # 64 attention heads at TP2
    q_head_dim: int = 576  # head_dim 512 + qk_rope_head_dim 64
    d_v: int = 512
    index_topk: int = 512
    sliding_window: int = 128
    swa_page_size: int = 256  # vLLM's SWA pool page size
    # compress_ratios distinct values: 0 (SWA-only), 4 (C4A), 128 (C128A)
    layer_classes: tuple[int, ...] = (0, 4, 128)
    decode_rows: tuple[int, ...] = (6, 12, 24, 48)
    num_splits: tuple[int, ...] = (1, 2, 4)
    sm_count: int = 48  # GB10
    num_experts_per_tok: int = 6
    experts_per_rank: int = 128  # 256 routed experts at TP2
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    # micro is selected iff num_tokens <= 8 and num_tokens*num_topk < 64
    # (_impl.py:1995-1999); at topk 6 that is every m in 1..8.
    micro_tokens: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    dynamic_tokens: tuple[int, ...] = (16, 64)
    # nvfp4 only. The int4-int8 deployment's MoE does NOT go through
    # _get_micro_kernel / _get_dynamic_kernel at all: _get_activation_kernel_spec
    # rejects w4a16 outright (moe/fused_moe/_impl.py:1359-1362) and dispatch
    # goes to sparkinfer/moe/_shared/kernels/w4a16/kernel.py, a separate family
    # with its own compile entry points (compile_w4a16_fused_moe:5960,
    # compile_w4a16_fused_moe_hybrid:6635, compile_w4a16_gemm:5803,
    # compile_w4a16_topk_sum). Driving those is a further piece of work; until
    # then the int4-int8 MoE kernels compile on first use.
    quant_modes: tuple[str, ...] = ("nvfp4",)


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Plan:
    """One named configuration and the thunk that compiles it."""

    name: str
    run: Callable[[], Any]
    tags: tuple[str, ...] = field(default_factory=tuple)


def _decode_plans(dep: Deployment) -> list[Plan]:
    import torch

    from ..attention._shared.mla import kernel as mla_kernel
    from ..attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat

    plans: list[Plan] = []
    for compress in dep.layer_classes:
        has_extra = compress != 0
        topk = dep.sliding_window if compress == 0 else dep.index_topk
        extra_topk = dep.index_topk if has_extra else 0
        for rows in dep.decode_rows:
            for splits in dep.num_splits:
                name = (
                    f"decode/c{compress}/rows{rows}/splits{splits}"
                )

                def run(
                    rows=rows, splits=splits, topk=topk,
                    extra_topk=extra_topk, has_extra=has_extra,
                ):
                    q_all = FakeCudaTensor(
                        (rows, dep.num_heads, dep.q_head_dim), torch.bfloat16
                    )
                    kv_flat = FakeCudaTensor((1 << 24,), torch.uint8)
                    swa_indices = FakeCudaTensor((rows, topk), torch.int32)
                    mid_out = FakeCudaTensor(
                        (rows, dep.num_heads, splits, dep.d_v), torch.bfloat16
                    )
                    mid_lse = FakeCudaTensor(
                        (rows, dep.num_heads, splits), torch.float32
                    )
                    swa_len = FakeCudaTensor((rows,), torch.int32)
                    extra_kv = FakeCudaTensor((1 << 24,), torch.uint8)
                    extra_idx = FakeCudaTensor(
                        (rows, max(extra_topk, 1)), torch.int32
                    )
                    extra_len = FakeCudaTensor((rows,), torch.int32)
                    num_chunks = (topk + 63) // 64 + (extra_topk + 63) // 64
                    return mla_kernel._sparse_mla_decode_grid_flat_launch(
                        q_all, kv_flat, swa_indices, mid_out, mid_lse, swa_len,
                        extra_kv, extra_idx, extra_len,
                        1.0, 1.0,
                        ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE,
                        False,
                        dep.swa_page_size, topk, extra_topk,
                        (topk + 63) // 64,
                        splits, max(1, -(-num_chunks // splits)),
                        656, dep.swa_page_size, 656,
                        1, 16, 0,
                        has_extra, True,
                    )

                plans.append(Plan(name, run, ("attention", "decode")))
    return plans


def _merge_plans(dep: Deployment) -> list[Plan]:
    import torch

    from ..attention._shared.mla import merge as mla_merge

    plans: list[Plan] = []
    for rows in dep.decode_rows:
        for chunks in dep.num_splits:
            name = f"sink_merge/rows{rows}/chunks{chunks}"

            def run(rows=rows, chunks=chunks):
                tmp_output = FakeCudaTensor(
                    (rows, dep.num_heads, chunks, dep.d_v), torch.bfloat16
                )
                tmp_lse = FakeCudaTensor(
                    (rows, dep.num_heads, chunks), torch.float32
                )
                num_chunks_ptr = FakeCudaTensor((1,), torch.int32)
                output = FakeCudaTensor(
                    (rows, dep.num_heads, dep.d_v), torch.bfloat16
                )
                attn_sink = FakeCudaTensor((dep.num_heads,), torch.float32)
                return mla_merge._sparse_mla_split_decode_merge_flat_launch(
                    tmp_output, tmp_lse, num_chunks_ptr, output, attn_sink,
                    tmp_output, tmp_lse, output,
                    chunks, True,
                )

            plans.append(Plan(name, run, ("attention", "merge")))
    return plans


def _moe_plans(dep: Deployment) -> list[Plan]:
    """The MoE families were already device-free; they just needed driving.

    ``mac_override`` is passed explicitly because the default queries the
    device for its max active cluster count, which a GPU-less build cannot do
    and which is part of the compile key.
    """
    import torch

    from ..moe.fused_moe import _impl as moe_impl

    plans: list[Plan] = []
    for quant_mode in dep.quant_modes:
        for tokens in dep.micro_tokens:
            plans.append(
                Plan(
                    f"moe_micro/{quant_mode}/m{tokens}",
                    (
                        lambda m=tokens, q=quant_mode: moe_impl._get_micro_kernel(
                            dep.experts_per_rank,
                            m,
                            dep.hidden_size,
                            dep.moe_intermediate_size,
                            dep.num_experts_per_tok,
                            topk_ids_dtype=torch.int32,
                            fast_math=True,
                            mac_override=dep.sm_count,
                            quant_mode=q,
                        )
                    ),
                    ("moe", "micro"),
                )
            )
        for tokens in dep.dynamic_tokens:
            plans.append(
                Plan(
                    f"moe_dynamic/{quant_mode}/m{tokens}",
                    (
                        lambda m=tokens, q=quant_mode: moe_impl._get_dynamic_kernel(
                            dep.experts_per_rank,
                            m,
                            dep.hidden_size,
                            dep.moe_intermediate_size,
                            dep.num_experts_per_tok,
                            m * dep.num_experts_per_tok,
                            topk_ids_dtype=torch.int32,
                            fast_math=True,
                            mac_override=dep.sm_count,
                            quant_mode=q,
                        )
                    ),
                    ("moe", "dynamic"),
                )
            )
    return plans


def build_plans(dep: Deployment | None = None) -> list[Plan]:
    dep = dep or Deployment()
    plans: list[Plan] = []
    for builder in (_decode_plans, _merge_plans, _moe_plans):
        try:
            plans.extend(builder(dep))
        except Exception as exc:  # a family that will not even enumerate
            plans.append(
                Plan(
                    f"{builder.__name__}:unavailable",
                    (lambda exc=exc: (_ for _ in ()).throw(exc)),
                    ("unavailable",),
                )
            )
    return plans


def precompile(
    dep: Deployment | None = None,
    *,
    verbose: bool = True,
) -> list[Result]:
    """Compile every planned configuration. Returns one Result per plan."""
    results: list[Result] = []
    with compile_only():
        for plan in build_plans(dep):
            try:
                plan.run()
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                results.append(Result(plan.name, False, detail))
                if verbose:
                    print(f"[aot]   FAIL {plan.name}: {detail}", flush=True)
                    print(
                        "".join(traceback.format_exception_only(type(exc), exc)),
                        flush=True,
                    )
            else:
                results.append(Result(plan.name, True))
                if verbose:
                    print(f"[aot]   ok   {plan.name}", flush=True)
    return results

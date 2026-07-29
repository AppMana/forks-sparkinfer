"""The CuTe-DSL configurations the shipped AOT cache is expected to carry.

This is a *floor*, checked by ``scripts/build_aot_cache.py verify`` against a
capture before that capture is packaged.  It is not a gate on the wheel and not
a gate on serving: the AOT cache is a cache, so a configuration nobody captured
still compiles on first use.  What it prevents is the narrower failure where a
warmup quietly stops exercising a kernel and the resulting near-empty cache
ships looking exactly like a good one.

Parameters below are read from the deployed checkpoints on the Spark rather
than assumed.  Both ``appmana/deepseek-v4-nvfp4-fp8`` and
``appmana/deepseek-v4-int4-int8`` carry identical attention and MoE geometry
and differ only in quantization::

    num_attention_heads      64      -> 32 per rank at TP2
    num_key_value_heads       1
    head_dim                512
    qk_rope_head_dim         64
    index_topk              512
    sliding_window          128
    num_hidden_layers        43
    n_routed_experts        256      -> 128 per rank at TP2
    num_experts_per_tok       6      <- NOT 8; this one matters, see below
    moe_intermediate_size  2048
    hidden_size            4096
    compress_ratios         44 entries over {0, 4, 128}
    torch_dtype        bfloat16

Serving flags come from ``tools/ampere/dgx_spark_serve_dsv4_tp2.sh`` in the
vLLM fork: ``--tensor-parallel-size 2``, ``--max-num-seqs 4``, and
``{"method": "dspark", "num_speculative_tokens": 5}``.  GB10 has 48 SMs.

Consequences that actually change the matrix:

* ``num_experts_per_tok`` is **6**.  The MoE micro backend is selected iff
  ``num_tokens <= 8`` and ``num_tokens * num_topk < 64``
  (``moe/fused_moe/_impl.py:1995-1999``, with ``_MICRO_MAX_TOKENS = 8`` at
  ``:1704`` and the cutover 64 at ``:1701``).  At topk 6 that is
  ``6 * 8 = 48 < 64``, so micro covers ``m = 1..8`` -- eight cubins, because
  ``_fc1_chunks_for_m`` makes ``rows_per_chunk = 16 * m`` and the resulting
  ``_cfg`` is part of the kernel cache key
  (``moe/_shared/kernels/micro.py:80-90, 388-412``).  Had topk been 8,
  ``8 * 8 = 64`` is not ``< 64`` and micro would stop at ``m = 7``.  That is
  the assumption previously flagged as unverified; it was wrong.

* The three ``compress_ratios`` values give three layer classes -- SWA-only
  (0), C4A (4), C128A (128) -- and each gets its own decode and prefill cubin.

* ``sm_count`` matters less than feared.  ``_dsv4_h16_auto``
  (``attention/_shared/mla/kernel.py:181-208``) has a 48-SM Spark-specific
  branch, but its production call site (``kernel.py:2906-2908``) does not pass
  ``sm_count``, so that branch is dead there.  ``sm_count`` reaches the compile
  key only through ``_wave_balanced_num_splits`` (``kernel.py:227-282``) via
  ``plan_unified_decode_splits`` (``kernel.py:285-380``), and for MoE only
  through ``m1_fc2_onepass`` when ``m == 1`` (``micro.py:694``).

Deliberately not hardcoded: the exact ``(num_splits, chunks_per_split)``
tuples.  They fall out of ``_wave_balanced_num_splits`` from ``sm_count``,
``num_chunks`` and the live row count, and the row counts come from vLLM's
CUDA-graph capture list, which moves with ``--max-num-seqs`` and the
speculative-decode length.  Pinning them here would produce a file that
silently stops matching the first time a serving flag changes.  The capture
enumerates them by construction; this file asserts only that enough showed up.

Follow-up worth taking, not done here and no longer urgent now that AOT is a
cache: the two MoE families can already be compiled with no GPU at all.
``_get_micro_kernel`` (``_impl.py:6586``) and ``_get_dynamic_kernel``
(``_impl.py:7189``) take only scalars, and every one of their compile arguments
is already a fake ``make_ptr`` or ``make_fake_compact_tensor``
(``_impl.py:6676-6695`` and ``:7499-7551``).  Their only device dependencies are
``current_cuda_stream()`` and ``get_num_sm`` / ``get_max_active_clusters``, and
both already accept scalar overrides (``mac_override``, ``max_active_ctas``).
The three attention families are what would need real work: they pass
``compile_args = runtime_args``, i.e. live ``from_dlpack`` tensors, at
``kernel.py:2449-2493``, ``merge.py:556-561`` and ``prefill_mg.py:3767-3786``.
"""

from __future__ import annotations

from dataclasses import dataclass

# The only architectures sparkinfer runs on. gating.py:29-32 accepts compute
# capability 12.0 and 12.1 and nothing else, so no other arch can reach these
# kernels: Ampere is served by flash_mla and the vLLM fork's Triton kernels.
# The "a" suffix is what a live device resolves to
# (cutlass/base_dsl/runtime/cuda.py:139-145), and the resolved name is what the
# compile cache key contains, so it is also what a GPU-less builder must set.
TARGET_ARCHS: tuple[str, ...] = ("sm_120a", "sm_121a")


@dataclass(frozen=True)
class KernelCoverage:
    """One kernel id the capture is expected to have exercised."""

    kernel_id: str
    min_configs: int
    why: str


# Kernel ids are the first field of the KernelCompileSpec built at each launch
# site; they appear verbatim in every cache manifest as ``kernel_id``.
REQUIRED_COVERAGE: tuple[KernelCoverage, ...] = (
    KernelCoverage(
        kernel_id="attention.mla.sm120.decode",
        min_configs=3,
        why=(
            "spec at attention/_shared/mla/kernel.py:2596; every decode launch. "
            "Three layer classes (compress_ratios {0, 4, 128}) is the floor; a "
            "capture that also sweeps the graph batch sizes produces more, "
            "since num_splits enters the key."
        ),
    ),
    KernelCoverage(
        kernel_id="attention.mla.sink_merge",
        min_configs=2,
        why=(
            "spec at attention/_shared/mla/merge.py:588; launched after every "
            "decode (kernel.py:3183). Keyed on static_num_chunks = num_splits, "
            "at least {1, 2} across the layer classes. vLLM always supplies "
            "attn_sink, so only the sink variant is reachable."
        ),
    ),
    KernelCoverage(
        kernel_id="attention.mla.sm120.prefill_mg",
        min_configs=1,
        why=(
            "spec at attention/_shared/mla/prefill_mg.py:3851; every "
            "mode='extend' launch. Token count is dynamic, so this is one cubin "
            "per layer class and one is the floor."
        ),
    ),
    KernelCoverage(
        kernel_id="integration.tp_moe.dynamic",
        min_configs=1,
        why=(
            "spec at moe/fused_moe/_impl.py:7553. mma_tiler_mn is the only "
            "token-dependent field (_impl.py:7228 -> :1405) and depends on m "
            "only through routed_rows = m * 6, so the whole decode band and "
            "normal prefill chunks share one cubin."
        ),
    ),
    KernelCoverage(
        kernel_id="integration.tp_moe.micro_direct",
        min_configs=1,
        why=(
            "spec at moe/fused_moe/_impl.py:6696. Reached for num_tokens 1..8 "
            "at topk 6; each m is its own cubin because rows_per_chunk = 16 * m "
            "(micro.py:80-90) feeds _cfg, which is in the kernel cache key. A "
            "decode-only capture may legitimately see a single m, so the floor "
            "is one."
        ),
    ),
)

REQUIRED_KERNEL_IDS: frozenset[str] = frozenset(c.kernel_id for c in REQUIRED_COVERAGE)


def check_coverage(observed: dict[str, int]) -> list[str]:
    """Return the coverage shortfalls for ``{kernel_id: distinct config count}``.

    Empty list means the capture met the floor.
    """
    problems: list[str] = []
    for coverage in REQUIRED_COVERAGE:
        count = observed.get(coverage.kernel_id, 0)
        if count < coverage.min_configs:
            problems.append(
                f"{coverage.kernel_id}: {count} compiled configuration(s), "
                f"expected at least {coverage.min_configs} -- {coverage.why}"
            )
    return problems

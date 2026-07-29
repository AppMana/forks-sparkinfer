"""The CuTe-DSL configurations production actually launches.

This is the coverage contract for the shipped AOT cache: what
``scripts/build_aot_cache.py verify`` requires to be present before a wheel is
allowed to claim it is AOT.  It is derived from the real call sites, not from
what the kernels *could* be asked to do -- the DSL kernels accept a far wider
space than DSV4-on-GB10 ever reaches, and compiling that space is neither
possible nor useful.

Derivation, with citations into the consumer (paths are in the vLLM fork
``forks-vllm-gb12x``) and into this repo:

Only two sparkinfer entry-point families are reachable from vLLM:

* ``sparkinfer.attention.compressed_mla`` -- ``plan``/``bind``/``run`` from
  ``vllm/models/deepseek_v4/nvidia_sm12x/kernels.py:118,207``, reached via
  ``.../attention.py:218-257``.
* ``sparkinfer.moe.fused_moe`` -- from
  ``vllm/model_executor/layers/fused_moe/experts/sparkinfer_moe.py:219,269,326``,
  selected by the NVFP4 oracle at ``.../oracle/nvfp4.py:111-116``.

Fixed for the deployment, so not axes at all: ``dtype=bfloat16`` and
``kv_dtype=uint8`` are literals (``kernels.py:122-123``); ``num_q_heads=32``
(64 heads / TP2, padded by ``get_padded_num_q_heads``); SWA pool page size 256
(``nvidia_sm12x/attention.py:134-138``); quant mode ``nvfp4``.

Two consequences worth stating because they *remove* work:

* vLLM's SWA page size is 256, and sparkinfer's sm121 single-pass decode route
  requires ``swa_page_size == 64``
  (``sparkinfer/attention/_shared/mla/compressed_api.py:47``).  That route is
  therefore unreachable from vLLM; every ``mode="decode"`` launch lands in
  ``run_unified_decode``.  It is not in the matrix.
* vLLM always supplies ``attn_sink`` (``nvidia_sm12x/attention.py:202-216``),
  so only the *sink* variant of the merge kernel is ever compiled.

The batch axis is bounded by CUDA-graph capture.  ``vllm/config/vllm.py:1811``
builds the size list and ``vllm/config/compilation.py:1466`` rounds each size
up to ``1 + num_speculative_tokens``; for the only in-repo DSV4/GB10 serving
config (``tools/ampere/dgx_spark_serve_dsv4_tp2.sh``: ``--max-num-seqs 4``,
``dspark`` with 5 speculative tokens, TP2) that yields decode row counts
``{6, 12, 18, 24, 36, 42, 48}``.  Row count is ``DimKey.dynamic()`` in the
decode compile key, so it enters only through ``num_splits`` /
``chunks_per_split`` / the h8-vs-h16 auto policy
(``_shared/mla/kernel.py:181-208, 286-356``) -- which is why seven row counts
collapse to three or four cubins per layer class.

Deliberately NOT hardcoded here: the exact ``(num_splits, chunks_per_split)``
tuples.  They are a function of ``sm_count``, the checkpoint's
``index_topk``/``compress_ratios``, and the serving flags, none of which a
build machine can know.  Encoding them would produce a matrix that silently
stops matching the moment ``--max-num-seqs`` changes.  What is asserted instead
is the set of *kernel ids* that must appear and the minimum number of distinct
compiled configurations for each -- a floor, not an equality, so a warmup that
covers more than the floor still passes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelCoverage:
    """One kernel id that the shipped cache must carry."""

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
            "spec built at _shared/mla/kernel.py:2596; every decode launch. "
            "Three is the floor for SWA-only layers alone (h8/1-split, "
            "h8/2-split, h16) over the captured row counts; a deployment with "
            "C4A or C128A layers adds a fourth per class."
        ),
    ),
    KernelCoverage(
        kernel_id="attention.mla.sink_merge",
        min_configs=2,
        why=(
            "spec at _shared/mla/merge.py:588; launched after every decode "
            "(kernel.py:3163). Keyed on static_num_chunks = num_splits, which "
            "is at least {1, 2} for any layer class."
        ),
    ),
    KernelCoverage(
        kernel_id="attention.mla.sm120.prefill_mg",
        min_configs=1,
        why=(
            "spec at _shared/mla/prefill_mg.py:3851; every mode='extend' "
            "launch. Token count is dynamic, so this is one cubin per layer "
            "class and one is the floor."
        ),
    ),
    KernelCoverage(
        kernel_id="integration.tp_moe.dynamic",
        min_configs=1,
        why=(
            "moe/fused_moe/_impl.py:7262. mma_tiler_mn is the only "
            "token-dependent field and stays (16, 128) for every num_tokens "
            "below 15*num_experts, so the whole decode band and normal "
            "prefill chunks share one cubin."
        ),
    ),
    KernelCoverage(
        kernel_id="integration.tp_moe.micro_direct",
        min_configs=1,
        why=(
            "moe/fused_moe/_impl.py:6694. Reached only for num_tokens*num_topk "
            "below the cutover at _impl.py:1701, i.e. small decode batches; "
            "each such m is its own cubin because rows_per_chunk scales with "
            "m (_shared/kernels/micro.py:81)."
        ),
    ),
)

REQUIRED_KERNEL_IDS: frozenset[str] = frozenset(c.kernel_id for c in REQUIRED_COVERAGE)


def check_coverage(observed: dict[str, int]) -> list[str]:
    """Return the coverage failures for ``{kernel_id: distinct config count}``.

    Empty list means the cache satisfies the contract.
    """
    problems: list[str] = []
    for coverage in REQUIRED_COVERAGE:
        count = observed.get(coverage.kernel_id, 0)
        if count < coverage.min_configs:
            problems.append(
                f"{coverage.kernel_id}: {count} compiled configuration(s), "
                f"need at least {coverage.min_configs} -- {coverage.why}"
            )
    return problems

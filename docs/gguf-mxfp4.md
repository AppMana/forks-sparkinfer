# GGUF MXFP4 expert projections

`b12x.loader.gguf_mxfp4.split_gguf_mxfp4` converts UINT8
`[E,N,K/32,17]` GGML blocks into native adjacent E2M1 nibbles and E8M0
scale bytes. GGML pairs positions j and j+16 in one byte; the native kernel
pairs 2*j and 2*j+1. The conversion preserves all code and scale bits,
including signed-zero codes, without floating-point weight expansion.
Output payload remains 17 bytes per 32 weights. Temporary tensors cover
at most 256 rows of one expert.

`prepare_gguf_mxfp4_fc2` prepares these blocks for the existing native
standalone down-projection kernel. It rejects scale bytes above 247 rather
than invoking the generic loader's saturation. This restriction belongs to
the current native consumer; the byte-layout conversion preserves all 256
scale codes. Source tensors must already be on CUDA. Full-model loading
still needs streaming source-to-prepared ownership to bound peak memory.

## Standalone FC2 layout fix

The multi-chunk standalone FC2 path previously read activation pairs using
the fused FC1 scratch layout. For nonuniform contiguous BF16 inputs at
I=2304, actual GGUF weights produced cosine approximately -0.007 against
an independent reference. Constant-input FC2 tests did not detect this.
The standalone branch now loads adjacent BF16 pairs; the fused branch
continues to consume its four-band layout. Input row offsets use Int64.

On GB10, Torch 2.13.0+cu130 and CUTLASS DSL 4.6.2:

- Fifteen tests cover source-code preservation, rejection of scale saturation,
  native FC2 regression cases, and nonuniform inputs at I=256/512/2304 with
  live route counts 1/3/17. Frozen graph replay changes activations, expert
  IDs and router weights, poisons output, checks stable allocation and checks
  that entirely invalid routes overwrite output with zero.
- Twelve actual Unsloth down-projection cases pass independent GGUF-decoder
  checks at route counts 1/6/48/192, H=5120 and I=2304. These include both
  selected candidate manifests. Cosine exceeds 0.999998, including replay.
  Weight samples are SHA256-checked before loading; compact resident payload
  is checked against the original byte count.

Reproduce with `python -m benchmarks.benchmark_gguf_mxfp4_fc2 --manifest
<sample-manifest> ... --device cuda:0 --output <evidence.jsonl>`.
The JSONL records weight identity, source hash, GPU UUID, correctness and
raw warm-cache graph samples. Those samples measure FC2 alone. They neither
select a prefill winner nor qualify a full mixed-format MoE layer.

## Mixed IQ/MXFP4 fused kernel

The low-level IQ preparation accepts `fc2_format="mxfp4"` with GGUF
MXFP4 down blocks. It packs the down weights for the existing tensor-core
pipeline, retains their exact E8M0 scale bytes, and uses a separate FC2
layout/scale contract. The fused compiler includes that format in its cache
identity and chooses each projection's tiles using its own encoding. The
public typed MoE API exposes this contract as:

```python
from b12x.moe import fused_moe as moe

source = moe.PackedSource(format="iq3_xxs", fc2_format="fp4_e8m0_k32")
weights = moe.IQMXFP4Weights(w13=w13_blocks, w2=down_blocks)
```

Use `format="iq2_xs"` for IQ2_XS gate/up. Gate/up tensors are UINT8
`[E,2*I,H/256,98]` (IQ3) or `[E,2*I,H/256,74]` (IQ2), and down tensors
are UINT8 `[E,H,I/32,17]`. Set `w13_layout` to match the checkpoint's
projection order. Weight planning, preparation, tuning queries (schema 11)
and compiled launches retain the independent down format. Mixed sources
require A16 execution; mismatched bundles and unsupported source combinations
fail during planning or preparation.

Twenty-four public-path cases passed on GB10: same-format IQ3, mixed IQ3/MXFP4
and mixed IQ2/MXFP4, at small/full expert dimensions, with mapped/unmapped
ownership and direct/packed routing. These exercise changing live counts,
inputs, routes and router weights under frozen graph replay. A further 110
contract and IQ regression cases passed.

BF16 tensor-core MXFP4 decoding now uses native E2M1 conversion and ordinary
E8M0 scale decoding. The old path scaled operands down by 2**119 and restored
that factor in the epilogue, losing precision for small scales. A register
oracle covers every FP4 byte combination and E8M0 scale byte 0 through 247
with exact BF16 agreement, including E8M0 zero's 2**-127 scale. Existing
NVFP4 conversion remains separate.

GB10 qualification adds six synthetic IQ3/MXFP4 fused cases at small and
full expert geometry, with direct/packed routing and poisoned graph replay.
The E8M0/native conversion regression selection passed 72 tests, with 16
skipped by their existing conditions. Twelve sampled expert-contribution
cases use actual pinned Unsloth IQ2_XS or IQ3_XXS gate/up with MXFP4 down,
H=5120, I=2304, top-k 6, and M=1/8/32/128. An explicit 384-entry ownership
map retains expert zero and marks the other experts nonlocal. Replay changes
inputs, routes and router weights; an entirely nonlocal map must overwrite
output with zero. Payload size remains equal to the original sampled bytes.

Run these samples with `python -m benchmarks.benchmark_gguf_mixed_moe
--manifest <sample-manifest> ... --device cuda:0 --output <evidence.jsonl>`.
This probe uses unclamped SiLU and a single sampled resident expert. It
qualifies that numerical kernel contract; full-model integration must still
apply the model's per-layer activation settings and validate all resident
experts, distributed reduction and model quality. It makes no full-layer
latency claim.

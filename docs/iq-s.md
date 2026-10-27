# IQ2_S and IQ3_S native decoding

Both formats retain explicit sign bytes and their original GGML codebooks.
They cannot be substituted with the parity-coded IQ2_XS/IQ2_XXS/IQ3_XXS
representations without changing weights.

| Format | Bytes / 256 weights | Grid | Scale reconstruction |
|---|---:|---|---|
| IQ2_S | 82 | 1024 × 8 magnitudes | `base * (subscale + 0.5) * 0.25` per K16 |
| IQ3_S | 110 | 512 × 4 magnitudes | `base * (1 + 2 * subscale)` per K32 |

The register decoders compute the scale and multiply magnitudes in FP32,
then round once to BF16 and apply the explicit signs. Eight-value and MMA
fragment decoders consume raw low/high index bits, signs and scale nibbles.
BF16 lookup tables occupy 16 KiB for IQ2_S and 4 KiB for IQ3_S. Device tables
must be prepared before replay.

`tests/moe/test_iq_s_decode.py` exercises every grid selector, all 256 sign
bytes and 16 subscales, nine FP16 base edge values, global/shared lookup and
MMA fragments. Graph replay changes source bytes, poisons output and checks
stable allocation. The oracle is independent GGUF 0.18.0 dequantization.
The fragment test caught a sign-byte extraction bug: upper bits from the
adjacent descriptor must be masked before the packed BF16 sign operation.

## Routed IQ2_S

`PackedSource(format="iq2_s")` and `IQ2SWeights` prepare same-format routed
experts from UINT8 `[E,N,K/256,82]` blocks. For MXFP4 down, use
`fc2_format="fp4_e8m0_k32"` and `IQMXFP4Weights` with UINT8
`[E,H,I/32,17]` down blocks. H and I must be divisible by 256. Only A16
execution is supported. MoE query schema 15 and candidate contract 18 include
this format.

Packing interleaves low grid bytes and full sign bytes into compact K8
16-bit descriptors. High grid bits and paired subscale nibbles remain in
K32 metadata words. Each block retains 64 descriptor bytes, 16 metadata
bytes and its two-byte base. Preparation temporaries cover at most 256
output rows of one expert; no dequantized weight matrix is retained.

IQ2_S's larger lookup table exposed a fused occupancy accounting error with
MXFP4 down: per-projection estimates did not include the retained FC1 table
beside the larger FC2 scratch region. Fused IQ planning now caps cooperative
occupancy using the final shared allocation and the existing 1536-byte
planning allowance. The GB10 M32 packed reproduction uses 50,688 dynamic
shared bytes, 196 registers and zero local bytes, and the driver reports one
resident CTA per SM. The revised plan requests one and passes public replay.

Twelve low-level routed cases passed at H/I=256/256 and 5120/2304 with
same-format and MXFP4 down, direct/packed routes, changed-input frozen replay,
poisoned scratch and compact payload accounting. The subsequent public/decoder suite passed 73 cases: 54 exact decoder
cases (both fragment layouts included), 16 public execution cases and three
contract checks. Public execution covers direct/packed routes, mapped and
unmapped ownership, changing live counts and entirely nonlocal outputs.
Four pinned Unsloth Q2 layer-15 IQ2_S/MXFP4 expert-contribution cases passed
at M=1/8/32/128, top-k 6, with exact source hashes and payload accounting.
These retain one expert out of 384 and currently use unclamped SiLU. IQ3_S routed staging
and the mixed IQ2_S/IQ3_S and IQ3_XXS/IQ3_S pairs are now implemented.
These checks do not establish full-model quality or select a performance winner.

The existing IQ2_XS/IQ2_XXS/IQ3_XXS routed and contract regression selection
passed 144 cases after the final occupancy change. The pinned Q2 shard header
also contains 40 routed-expert and 40 shared-expert SwiGLU clamp values, all
10.0. The header hash and arrays are retained in `activation-contract.json`;
model integration must preserve and apply these values, and qualify clipping
semantics on inputs that actually reach the clamp.

## Routed IQ3_S and mixed down projections

`PackedSource(format="iq3_s")` with `IQ3SWeights` accepts raw UINT8
`[E,N,K/256,110]` blocks. `IQMixedWeights` supports IQ2_S or IQ3_XXS
gate/up with IQ3_S down. Compact packing retains 64 grid bytes and 46
base/metadata bytes per block. Each K64 metadata group contains two high
index bytes, eight sign bytes and one packed scale byte. Preparation is
bounded to 256 output rows of one expert.

Mixed projections reserve the larger lookup-table capacity and restage the
IQ3_S table after the FC1 grid barrier. Final fused shared-memory capacity
controls cooperative occupancy. Public preparation records each projection's
format independently; replay performs no preparation or policy lookup.

On GB10, 18 low-level and 24 public routed cases passed across same-format
IQ3_S and both mixed pairs, including full H5120/I2304 geometry. The final
regression selection passed 149 cases, including exact packing round trips,
invalid-plan rejection and existing IQ paths. Twelve pinned Unsloth expert
contribution cases passed at M1/8/32/128: layer 15 IQ2_S/MXFP4, layer 20
IQ2_S/IQ3_S and layer 25 IQ3_XXS/IQ3_S. These remain unclamped numerical
probes, not full-model qualification or a prefill performance comparison.

## Checkpoint clamp and streaming ownership

The actual-weight probe accepts `--activation-contract` and verifies its repo,
revision and variant against each sample manifest. It uses the per-layer routed
clamp in both the native launch and independent reference. `--stress-clamp`
requires positive gate and both up boundaries to be crossed even after replay
halves the input; negative gate values remain unbounded. All 32 sampled Q2
cases passed on GB10 across the seven routed format pairs at M1/8/32/128.
This qualifies the routed clamp path, not shared-expert arithmetic or model quality.

`b12x.loader.gguf_experts.prepare_iq_expert_partition` streams selected global
expert IDs into compact resident storage and returns the global-to-local map.
It allocates no full raw partition; temporaries are bounded by one expert triple
and its prepared representation. IQ metadata puts all expert bases before all
remaining metadata, so the loader copies these regions separately. The caller
retains responsibility for activation configuration and collective reduction.
Eight GPU cases match whole-partition preparation byte for byte, including all
Q2 mixed pairs, nonsequential ownership and source offsets beyond 4 GiB. The
combined loader suite passed 11 tests. Public serving binding is described below; vLLM integration remains to be wired.

## Public streamed preparation

Pass `GGUFExpertWeights(gate, up, down, expert_ids, device)` to
`fused_moe.prepare_weights` with an A16 BF16 SiLU `WeightPlan`, w31 order,
matching per-projection formats and local expert geometry. The three sources
are `ExpertTensor` descriptors from the GGUF index; `expert_ids` gives local
slot order. Preparation validates the plan before reading source data and
attaches compact planes directly to the canonical `PreparedExperts` owner.
There is no second packed-weight copy. Bind the matching global-to-local
expert map through `route_expert_map`. Collectives remain caller-owned.

The public streaming suite passed eight codec-pair cases with byte-exact
comparison to whole-partition preparation and independent GGUF numerical
references. Every case exercises direct capacity8 and packed capacity32,
multiple live counts, clamped SiLU, changed-input graph replay, nonlocal and
zero contributions, poisoned scratch, stable addresses, and frozen compiler
resolution. These are single-GPU execution checks, not two-rank qualification.

## Two-Spark EP execution qualification

`benchmarks/qualify_iq_ep.py` runs public planned IQ2_S/IQ3_S execution on
two GB10 ranks at H5120/I2304, global384/top-k6, with one distinct synthetic
resident expert per rank (IDs0 and192). Replicated activations/routes feed
local native contributions followed by exactly one NCCL all-reduce. Direct
capacity8 and packed capacity32 each exercise three live token counts.
Both ranks passed all six cases, including captured collective replay with
changed activations/routes and all-zero probabilities; replay allocations
remain unchanged. Maximum relative L2 was 0.004046 against independently
decoded GGUF references reduced in FP32. Raw per-rank JSONL records include
GPU UUIDs and matching MoE source hashes.

This test establishes the sampled EP2 numerical and graph contract. It uses
synthetic weights and the raw public bundle path, not a complete streamed
checkpoint or vLLM server. It does not select a performance winner.

# IQ3_XXS packed projections

`blockscaled.pack_weight(blocks, recipe="iq3_xxs")` accepts CUDA UINT8
`[N,K/256,98]` GGML blocks. The current implementation supports BF16
activations and outputs on SM120/SM121 through the prepared blockscaled GEMM
API. GB10 is the hardware qualified so far. Prepared routed MoE also supports
IQ3_XXS for both projections through `PackedSource(format="iq3_xxs")` and
`IQ3XXSWeights(w13, w2)`, with UINT8 `[E,N,K/256,98]` source tensors. Mixed
IQ gate/up with MXFP4 down is supported through `PackedSource.fc2_format`
and `IQMXFP4Weights` (see `gguf-mxfp4.md`). IQ2_XXS gate/up with IQ3_XXS
down uses `IQMixedWeights` and `fc2_format="iq3_xxs"` (see `iq2-xxs.md`). The separate IQ3 small-row SIMT
candidate remains pending. Kernel selection requires workload measurements;
these correctness results do not select a prefill or decode winner.

Each 256-weight block contains an FP16 base, 64 one-byte grid indices and
32 bytes of combined signs/subscales. An index selects four magnitudes from
the original 256-entry codebook. Each group of 32 weights has four seven-bit
sign codes with parity and a four-bit subscale. Reconstruction is
`base * (subscale + 0.5) * 0.5 * signed_magnitude`, rounded once to BF16.
Activation scaling and weight requantization are not part of this path.

Dense preparation splits the original bytes into two allocations:

- Grid indices: `[N,K/4]`, tiled by 128 output rows (eight for N tails).
- Metadata: `[ceil(N/128),K/256,4352]`; each tile contains 256 base bytes
  followed by 32 planes of 128 sign/subscale bytes.

For N divisible by 128, payload size remains exactly 98 bytes per 256
weights, or 3.0625 bits per weight. Only the final metadata N tile is padded.
The shared lookup table occupies 2 KiB per CTA. The existing native A16
pipeline stages compact grids and metadata, reconstructs register fragments,
and feeds BF16 tensor-core MMA with FP32 accumulation. There is no resident
expanded weight matrix. Plans prepare all kernels and scratch before replay;
live row counts do not become new compile keys.

MoE preparation uses the existing N64/K128 tiled grid layout and compact
base/sign/subscale planes. Both H and I must be divisible by 256, so a TP2
split of I=2304 into 1152 is rejected. Expert ownership can instead retain
full-width experts and use the prepared route map for nonlocal contributions.
The caller still owns the distributed reduction.

The IQ3 cooperative launch explicitly requests the full shared-memory
carveout. On GB10, the 32-row tile uses 32,256 dynamic shared bytes per CTA,
plus 1,024 reserved bytes. Two CTAs fit the full SM budget but exceed the
DSL's default 64% carveout; leaving that default caused a cooperative-launch
admission failure. This is a correctness fix, not a benchmark result.

## Validation

On GB10 with Torch 2.13.0+cu130 and CUTLASS DSL 4.6.2:

- `tests/moe/test_iq3_xxs_decode.py`: 18 cases, bit-exact against GGUF,
  covering all grid/sign/subscale choices, nine FP16 base edge values,
  global/shared lookups and changed-input graph replay.
- `tests/gemm/test_iq3_xxs.py`: six configurations covering output tails,
  multiple K stages, split-K and live M=1/3/8/17/65. Frozen replay checks
  changed inputs, poisoned outputs and unchanged live allocation.
- Six existing IQ2_XS replay cases passed after extending the shared pipeline.
- `tests/moe/test_iq3_xxs_moe.py`: six low-level routed SiLU cases with
  H/I=256/256 and 5120/2304, direct M=1/8 and packed M=32. Independent
  GGUF references, exact compact payload size, frozen graph replay, changed
  inputs, poisoned output/scratch and stable live allocation all passed.
- Eight public prepared MoE cases cover the same small/full expert dimensions,
  top-k 6, mapped and unmapped expert ownership, direct capacity 8 and packed
  capacity 32. Each exercises multiple live counts and balanced/hot/imbalanced
  routes with compilation forbidden, then mutates inputs, routes and router
  weights during replay. Zero router weights and entirely nonlocal expert maps
  must overwrite the output with zero. These routed tests use synthetic GGUF
  blocks and independent GGUF dequantization; they do not establish model quality.
- 24 actual Unsloth layer20/layer39 gate/up projections passed independent
  GGUF-oracle checks at M=1/4/8/32/128/512, including poisoned output and
  scratch during replay. Fixture bytes are SHA256-checked against pinned
  GGUF samples before wrapping them in safetensors for the benchmark.

Reproduce the checkpoint fixture with
`benchmarks/prepare_unsloth_iq2_fixture.py --recipe iq3_xxs --manifest ... --output ...`,
then use `benchmarks/benchmark_dense_gemm.py --dtype checkpoint-a16
--checkpoint-recipe iq3_xxs --model-path ... --evidence ... --check --flush-l2`.
The evidence records selected configurations, source hashes, physical device,
raw cold-cache timings, correctness and memory usage. Projection results do
not establish routed MoE latency or full-model quality.

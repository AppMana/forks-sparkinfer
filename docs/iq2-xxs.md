# Compact IQ2_XXS execution

IQ2_XXS stores 66 bytes per 256 weights: an FP16 base and eight groups of
four grid-index bytes plus a packed sign/subscale word. Each grid byte
selects eight magnitudes from the original 256-entry GGML codebook. Each
32-weight group shares a four-bit subscale and contains four seven-bit sign
codes with parity. Decode uses `base * (subscale + 0.5) * 0.25 * magnitude`,
then the sign, rounding once to BF16. IQ2_XS has a different codebook and
cannot represent every IQ2_XXS block losslessly.

`blockscaled.pack_weight(blocks, recipe="iq2_xxs")` accepts CUDA UINT8
`[N,K/256,66]` blocks. Preparation separates tiled grid bytes `[N,K/8]`
from 34 metadata bytes per block. The latter use the same base/byte-plane
layout as IQ3_XXS. N must be divisible by eight; only the last metadata N128
tile is padded. At N128 alignment the resident weight payload remains
exactly 66 bytes per block. No expanded weight matrix is retained.

The tensor-core path stages compact bytes, uses a 4 KiB shared BF16 lookup
table, reconstructs register fragments and accumulates in FP32. TMA requires
at least 16 packed bytes per row, so IQ2_XXS admits K128/K256 tiles. K64
would produce eight-byte rows and caused an illegal-instruction launch fault
in the first integration probe; planning, candidate enumeration and direct
kernel construction now reject it. Query schema is 7 and the candidate
contract is 18. The K256 bulk-copy path also derives its transfer size and
offsets from the format's 32-byte grid payload; retaining the IQ2_XS/IQ3_XXS
64-byte assumption mismatched the pipeline transaction count and stalled
the initial K256 probe. There is no small-row SIMT candidate for this format yet.

The register decoder passed 27 GB10 GPU cases against independent GGUF
0.18.0 decoding: all grid/sign/subscale combinations, nine FP16 base edge
values, global/shared lookup and tensor-core fragment layouts, with changed
source bytes, poisoned output and stable-allocation graph replay. Run
`python -m pytest tests/moe/test_iq2_xxs_decode.py` to reproduce.

Six dense numerical/replay cases and a K64 rejection case passed in
`tests/gemm/test_iq2_xxs.py`. These cover N128/N136, K768, K128/K256
tiles, split-K, and changing M=1/3/8/17/65. The combined IQ2_XXS/IQ3_XXS/
IQ2_XS run reached 64 passes before an existing policy error-message
assertion failed; restoring format-specific diagnostics then passed all 14
selected descriptor/policy/API regression cases.
Routed MoE integration, actual checkpoint projections and workload performance
selection remain separate requirements; decoder tests do not prove those.

The corrected N128/K256-tile case passed Compute Sanitizer memcheck with
zero errors, using `--report-api-errors no`. The unsuppressed run passed the
numerical test but reported 41 `cuGetProcAddress_v2` invalid-value API probes
and no device memory errors; both logs are retained. API-error suppression
was limited to that diagnostic rerun and does not disable device memcheck.

## Routed experts

`PackedSource(format="iq2_xxs")` with `IQ2XXSWeights(w13, w2)` prepares
same-format experts. Each projection uses native compact grid/sign/scale
planes; IQ2_XXS stages half as many grid bytes as IQ2_XS or IQ3_XXS.
Offsets and the expert-count calculation use that smaller width. Both H and
I must be divisible by 256. Use expert ownership maps for rank-local full
experts; reduction of rank contributions remains the caller's responsibility.

For mixed projections, use one of these source and bundle combinations:

| Gate/up | Down | `fc2_format` | Bundle |
|---|---|---|---|
| IQ2_XXS | MXFP4 | `"fp4_e8m0_k32"` | `IQMXFP4Weights` |
| IQ2_XXS | IQ3_XXS | `"iq3_xxs"` | `IQMixedWeights` |

The IQ2_XXS→IQ3_XXS pair occurs in Q2 candidate layers 5–14. Its fused
kernel reuses the 4 KiB shared lookup region: after the grid barrier has
finished all FC1 readers, it loads the 2 KiB IQ3 table and waits for local
completion before FC2. This avoids an additional shared allocation. The
copy is part of execution and must be included in performance comparisons.
Prepared metadata describes each projection separately. MoE query schema
13 and candidate contract 16 separate these plans from earlier contracts.

Twelve low-level same-format/MXFP4 cases and six mixed-IQ cases passed at
H/I=256/256 and 5120/2304, with direct and packed routes, frozen replay,
changed inputs and poisoned output/scratch. The expanded public suite passed 28 cases: 24 execution cases for all three
down formats and four contract checks. Execution covers mapped/unmapped
ownership and multiple live counts under frozen plans.

Actual pinned Unsloth layer-2/layer-5 samples passed twelve expert-contribution cases
across the Q2 and IQ3 candidate variants at M=1/8/32/128, top-k 6. The Q2
layer-2 sample has IQ2_XXS down and layer 5 has IQ3_XXS down; the IQ3
candidate uses MXFP4 down. The probe retains
one sampled expert and marks other global experts nonlocal. It uses unclamped
SiLU and does not establish full-model quality or full-layer performance.

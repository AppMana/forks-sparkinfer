# GB10 GGUF dense prefill

The planned Q6_K and Q8_0 BF16-activation path retains the checkpoint's codes
and scales. On NVIDIA GB10 (SM121, 48 SMs), capacities of at least 128 rows
with K>=1280 and N>=1024 select M64/N64/K64 for Q6_K and M32/N64/K64
for Q8_0. Other device/geometry cases retain their previous defaults.

Dynamic capacity plans also prepare the previous M16/N64/K64 kernel. Live
counts below 128 use that callable; counts of at least 128 use the prefill
callable. Dispatch allocates no GPU memory and performs no compilation.
The typed `short_rows` config records this behavior. Explicit overrides default
to `short_rows=False`, so they continue to force the requested schedule.
Config schema5 and candidate contract21 invalidate earlier tuning records.

## Component evidence

2026-10-27, CUTLASS DSL4.6.2, Torch2.13.0+cu130, NVIDIA GB10 UUID
c775c34d-9e0c-f7c7-b950-918680d667a3, isolated from model serving.
Checkpoint: Unsloth DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL revision
60ad3651641b5c97679c807e064ab1d7a3ca1455.

Seven actual TP2 projection geometries include shared gate/down, attention
Q-A/Q-B, grouped WO-A, WO-B, and Engram N12800/K6144. At M1/8/127/128/512/1024,
all42 policy outputs were bitwise equal to the previous kernel, and each
passed an independent GGUF-dequantization FP32-matmul oracle. Frozen graph
replay retained allocation counts. Paired cold timings evict128MiB before
the timed work and alternate candidate order for ten trials:

| Recipe | M>=128 baseline/policy latency ratio | M<128 ratio |
|---|---|---|
| Q6_K | 2.14–2.79 (median2.54) | 0.992–1.004 |
| Q8_0 | 1.20–1.67 (median1.50) | 0.977–1.018 |

Ratios above1 mean the policy is faster. Larger K256 tiles reduced redundant
metadata transfers but lost in actual timings; they are not selected.
These are component results, not end-to-end serving throughput or a global
optimum claim. Raw timings and nvidia-smi snapshots remain external evidence.

Reproduce against the already-verified checkpoint:

```sh
.venv/bin/python benchmarks/benchmark_gguf_dense_prefill.py \
  --checkpoint /path/to/checkpoint-parent --source-commit <tested-commit> \
  --output /path/to/results.jsonl
```

Regression checks in `tests/gemm/test_q6_k.py` exercise both recipes across the
127/128 boundary, changed inputs, independent oracles, frozen kernel
resolution, and allocation-free graph replay. `tests/preparation/test_defaults.py`
covers device qualification, short capacities, serialization, and overrides.

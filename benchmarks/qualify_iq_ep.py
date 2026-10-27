"""Two-rank public IQ expert execution and captured NCCL sum qualification.

Uses independent synthetic GGUF blocks at DSV4.1 geometry. This is a distributed
contract test, not full-model quality or a kernel performance comparison.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

import gguf
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from benchmarks.moe_preparation import request_for_capacity
from benchmarks.benchmark_iq2_xs_moe import capture_lifetime, check


def make_weight(e, n, k, seed, device, recipe):
    rng = np.random.default_rng(seed)
    size = {"iq2_s": 82, "iq3_s": 110}[recipe]
    raw = rng.integers(0, 256, (e, n, k//256, size), dtype=np.uint8)
    raw[..., :2] = rng.normal(0, .001, (e, n, k//256, 1)).astype(np.float16).view(np.uint8)
    decoded = gguf.dequantize(raw.reshape(e*n, -1), getattr(gguf.GGMLQuantizationType, recipe.upper()))
    return torch.from_numpy(raw).to(device), torch.from_numpy(decoded).reshape(e,n,k).bfloat16().to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--master', required=True)
    parser.add_argument('--port', type=int, default=29661)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(MASTER_ADDR=args.master, MASTER_PORT=str(args.port),
                      RANK=str(args.rank), WORLD_SIZE='2', LOCAL_RANK='0')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=10), device_id=device)
    h, i, topk, global_e = 5120, 2304, 6, 384
    # Distinct native weights on each rank; the shared activation/routes are
    # identical. Only IDs 0 and 192 are resident in this sampled EP test.
    raw13, ref13 = make_weight(1, 2*i, h, 45+args.rank, device, 'iq2_s')
    raw2, ref2 = make_weight(1, h, i, 75+args.rank, device, 'iq3_s')
    wp = moe.plan_weights(
        source=moe.PackedSource(format='iq2_s', fc2_format='iq3_s', w13_layout='w31'),
        activation=moe.ActivationSpec(mode='a16', nonlinearity='silu',
                                      io_dtype=torch.bfloat16, swiglu_limit=10.0),
        geometry=moe.MoEGeometry(num_experts=1, hidden_size=h, intermediate_size=i),
    )
    experts = moe.prepare_weights(plan=wp, weights=moe.IQMixedWeights(raw13, raw2))
    del raw13, raw2
    mapping = torch.full((global_e,), -1, dtype=torch.int32, device=device)
    mapping[args.rank*192] = 0
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/'b12x/moe').rglob('*.py')) + [Path(__file__)]
    identity = {'command': sys.argv, 'rank': args.rank,
                'scope': 'synthetic sampled EP2 public MoE plus NCCL sum',
                'gpu_uuid': str(torch.cuda.get_device_properties(device).uuid),
                'source_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest(),
                'torch': torch.__version__, 'nccl': torch.cuda.nccl.version()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as output:
        for mode, capacity, counts in [('direct', 8, (1, 3, 8)), ('packed', 32, (1, 17, 32))]:
            x = torch.empty((capacity,h), dtype=torch.bfloat16, device=device)
            ids = torch.empty((capacity,topk), dtype=torch.int32, device=device)
            probs = torch.empty((capacity,topk), dtype=torch.float32, device=device)
            out = torch.empty_like(x)

            def inputs(m, step):
                torch.manual_seed(140+step)
                x[:m].normal_(std=2)
                ids[:m].copy_(torch.tensor([0,192,4,5,6,7], dtype=torch.int32, device=device))
                if step:
                    ids[:m:2,0] = 9
                    ids[1:m:2,1] = 10
                probs[:m].fill_(1/topk)

            def oracle(m):
                gate, up = (x[:m].float() @ ref13[0].float().T).bfloat16().chunk(2, dim=-1)
                mid = (F.silu(gate.float().clamp(max=10)) * up.float().clamp(-10,10)).bfloat16()
                weight = torch.where(ids[:m] == args.rank*192, probs[:m], 0).sum(-1)
                result = (mid.float() @ ref2[0].float().T) * weight[:,None]
                dist.all_reduce(result)
                return result

            inputs(capacity, 0)
            decl = moe.plan_execution(experts=experts,
                capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=topk, route_num_experts=global_e),
                override=moe.MoeDecodeConfig(backend='w4a16', route_planner='internal',
                                            max_active_clusters=None, w4a16_route_mode=mode))

            def factory(state):
                scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device)
                                for s in state.scratch.scratch_specs())
                bound = state.bind(scratch=scratch, a=x, topk_ids=ids, topk_weights=probs,
                                   output=out, route_expert_map=mapping)
                return PreparedCall(run=bound.run, output=out, owners=(scratch,bound))

            with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
                session.prepare((request_for_capacity(decl, name=f'ep-{mode}', calls={capacity: factory}),))
                state = require_prepared(decl, 'moe.decode')
                if hasattr(state, 'variants'):
                    state = state.variants[capacity]
                scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device)
                                for s in state.scratch.scratch_specs())
                for m in counts:
                    inputs(m, 0)
                    bound = moe.bind(decl, scratch=scratch, a=x[:m], topk_ids=ids[:m],
                                     topk_weights=probs[:m], output=out[:m], route_expert_map=mapping)
                    def run():
                        moe.run(binding=bound)
                        dist.all_reduce(out[:m])
                    expected = oracle(m)
                    for _ in range(3):
                        run()
                    metrics = check(out[:m], expected)
                    dist.barrier()
                    with kernel_resolution_guard('EP2 graph replay'):
                        graph = torch.cuda.CUDAGraph()
                        with capture_lifetime(), session.capture(), torch.cuda.graph(graph):
                            run()
                        inputs(m, 1)
                        expected = oracle(m)
                        out.fill_(float('nan'))
                        allocated = torch.cuda.memory_stats(device)['allocation.all.allocated']
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == allocated
                        metrics.update({'replay_'+k:v for k,v in check(out[:m], expected).items()})
                        probs.zero_()
                        out.fill_(float('nan'))
                        graph.replay()
                        torch.cuda.synchronize()
                        check(out[:m], torch.zeros_like(out[:m]))
                        graph.reset()
                    record = dict(identity, mode=mode, m=m, capacity=capacity,
                                  h=h, i=i, topk=topk, global_experts=global_e,
                                  resident_experts_per_rank=1, **metrics)
                    output.write(json.dumps(record)+'\n')
                    output.flush()
                    print(json.dumps(record), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

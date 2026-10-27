"""Actual pinned GGUF Q6_K projections through the public planned GEMM path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import gguf
import numpy as np
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from tests.gemm._blockscaled import prepared
from benchmarks.benchmark_iq2_xs_moe import check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads(args.manifest.read_text())
    root=Path(__file__).resolve().parents[1]
    sources=sorted((root/'b12x/gemm/blockscaled').glob('*.py')) + [root/'b12x/_lib/dense_gemm.py',root/'b12x/_lib/quant/gguf_int_decode.py',Path(__file__)]
    identity={'command':sys.argv,'repo':manifest['repo'],'revision':manifest['revision'],
              'scope':'full sampled Q6_K projection correctness, not latency selection',
              'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
              'source_sha256':hashlib.sha256(b''.join(s.read_bytes() for s in sources)).hexdigest()}
    with args.output.open('w') as output:
        for sample in manifest['samples']:
            data=(args.manifest.parent/Path(sample['path']).name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==sample['sha256']
            k,n=sample['dims']
            raw=np.frombuffer(data,dtype=np.uint8).copy().reshape(n,k//256,210)
            ref=torch.from_numpy(gguf.dequantize(raw.reshape(n,-1),gguf.GGMLQuantizationType.Q6_K)).bfloat16().cuda()
            weight=blockscaled.pack_weight(torch.from_numpy(raw).cuda(),recipe='q6_k')
            x=torch.randn(128,k,dtype=torch.bfloat16,device='cuda')*.1
            out=torch.empty(128,n,dtype=torch.bfloat16,device='cuda')
            override=blockscaled.BlockscaledConfig(mode='a16',tile_m=32,tile_n=128,tile_k=128,split_k=1)
            with prepared(x,weight,out=out,override=override) as plan, kernel_resolution_guard('actualQ6replay'):
                for m in (1,8,32,128):
                    expected=x[:m].float()@ref.float().T
                    blockscaled.mm(x[:m],weight,out=out[:m],plan=plan)
                    metrics=check(out[:m],expected)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        blockscaled.mm(x[:m],weight,out=out[:m],plan=plan)
                    x[:m].neg_()
                    out.fill_(float('nan'))
                    allocated=torch.cuda.memory_stats()['allocation.all.allocated']
                    graph.replay()
                    torch.cuda.synchronize()
                    assert allocated==torch.cuda.memory_stats()['allocation.all.allocated']
                    metrics.update({'replay_'+k:v for k,v in check(out[:m],-expected).items()})
                    graph.reset()
                    row=dict(identity,tensor=sample['name'],m=m,n=n,k=k,
                             sample_sha256=sample['sha256'],resident_bytes=weight.values.numel()+weight.metadata.numel(),**metrics)
                    output.write(json.dumps(row)+'\n')
                    output.flush()
                    print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()

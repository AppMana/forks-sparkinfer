"""Lossless GGUF Q8_0 planes for native BF16 tensor-core GEMM."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Q8_0LinearWeight:
    values: torch.Tensor
    metadata: torch.Tensor
    in_features: int
    out_features: int

    @property
    def padded_in_features(self):
        return self.in_features


def pack_q8_0_weight(blocks):
    from ._a16 import _check_tensor

    if blocks.device.type != "cuda":
        raise ValueError("Q8_0 weights must be on CUDA")
    _check_tensor("Q8_0 blocks", blocks, blocks.device, torch.uint8)
    if blocks.ndim != 3 or blocks.shape[-1] != 34 or min(blocks.shape) <= 0:
        raise ValueError("Q8_0 requires [N,K/32,34] blocks")
    n, groups, _ = blocks.shape
    if n % 8 or groups % 8:
        raise ValueError("Q8_0 requires N divisible by 8 and K divisible by 256")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q8_0 packing must precede graph capture")
    bases = blocks[..., :2].contiguous()
    if not torch.isfinite(bases.view(torch.float16)).all().item():
        raise ValueError("Q8_0 block bases must be finite")
    nt = (n+127)//128
    # One K256 tile: eight planes of 128 original FP16 bases.
    rows = torch.zeros((nt*128,groups,2), dtype=torch.uint8, device=blocks.device)
    rows[:n].copy_(bases)
    metadata = rows.reshape(nt,128,groups//8,8,2).permute(0,2,3,1,4).contiguous().reshape(nt,groups//8,2048)
    values = blocks[...,2:].contiguous().reshape(n,groups*32)
    return Q8_0LinearWeight(values,metadata,groups*32,n)

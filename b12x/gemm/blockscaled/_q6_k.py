"""Lossless Q6_K low-nibble values and high-bit/scale metadata planes."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Q6KLinearWeight:
    values: torch.Tensor
    metadata: torch.Tensor
    in_features: int
    out_features: int

    @property
    def padded_in_features(self):
        return self.in_features


def pack_q6_k_weight(blocks):
    from ._a16 import _check_tensor

    if blocks.device.type != "cuda":
        raise ValueError("Q6_K weights must be on CUDA")
    _check_tensor("Q6_K blocks", blocks, blocks.device, torch.uint8)
    if blocks.ndim != 3 or blocks.shape[-1] != 210 or min(blocks.shape) <= 0:
        raise ValueError("Q6_K requires [N,K/256,210] blocks")
    n, groups, _ = blocks.shape
    if n % 8:
        raise ValueError("Q6_K requires N divisible by 8")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q6_K packing must precede graph capture")
    if not torch.isfinite(blocks[..., 208:].contiguous().view(torch.float16)).all().item():
        raise ValueError("Q6_K block bases must be finite")
    nt = (n + 127) // 128
    values = torch.empty((n, groups * 128), dtype=torch.uint8, device=blocks.device)
    metadata = torch.zeros((nt, groups, 82, 128), dtype=torch.uint8, device=blocks.device)
    # Bound preparation temporaries to one N128 tile. Values stay four bits;
    # metadata holds the remaining two bits, signed K16 scales and FP16 bases.
    for start in range(0, n, 128):
        rows = min(128, n - start)
        raw = blocks[start:start + rows]
        low = raw[..., :128].reshape(rows, groups, 2, 1, 64)
        lo = ((low >> torch.tensor([0, 4], dtype=torch.uint8, device=blocks.device).view(1, 1, 1, 2, 1)) & 15).reshape(rows, groups, 256)
        values[start:start + rows].copy_((lo[..., 0::2] | (lo[..., 1::2] << 4)).reshape(rows, -1))
        high = raw[..., 128:192].reshape(rows, groups, 2, 1, 32)
        hi = ((high >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=blocks.device).view(1, 1, 1, 4, 1)) & 3).reshape(rows, groups, 256)
        h = hi[..., 0::4] | (hi[..., 1::4] << 2) | (hi[..., 2::4] << 4) | (hi[..., 3::4] << 6)
        tile = metadata[start // 128, :, :, :rows]
        tile[:, :2].copy_(raw[..., 208:210].permute(1, 2, 0))
        tile[:, 2:18].copy_(raw[..., 192:208].permute(1, 2, 0))
        tile[:, 18:].copy_(h.permute(1, 2, 0))
    return Q6KLinearWeight(values, metadata.reshape(nt, groups, 10496), groups * 256, n)

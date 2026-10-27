"""Lossless dense IQ3_XXS storage prepared from safetensors block payloads."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IQ3XXSLinearWeight:
    """Grid-index tiles and metadata [ceil(N/128),K/256,4352], in bytes.

    The [N,K/4] descriptor byte view stores [N/B,K/256,B,64] tiles,
    where B is 128 for N divisible by 128, and 8 otherwise.
    Each metadata tile holds 128 FP16 block bases followed by 32 byte planes
    of 128 packed sign/subscale words. Only the final N metadata tile is padded.
    Tensors are owned by this object and must remain unmodified during use.
    """

    values: torch.Tensor
    metadata: torch.Tensor
    in_features: int
    out_features: int

    @property
    def padded_in_features(self) -> int:
        return self.in_features


def pack_iq3_xxs_weight(blocks: torch.Tensor) -> IQ3XXSLinearWeight:
    """Rearrange uint8 [N,K/256,98] payloads without dequantizing weights."""
    from ._a16 import _check_tensor

    if blocks.device.type != "cuda":
        raise ValueError("IQ3_XXS weights must be on CUDA")
    _check_tensor("IQ3_XXS blocks", blocks, blocks.device, torch.uint8)
    if blocks.ndim != 3 or blocks.shape[-1] != 98 or min(blocks.shape) <= 0:
        raise ValueError("IQ3_XXS blocks must have positive shape [N,K/256,98]")
    n, kb, _ = blocks.shape
    if n % 8:
        raise ValueError("IQ3_XXS dense weights require N divisible by 8")
    with torch.cuda.device(blocks.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("IQ3_XXS weight packing must precede CUDA graph capture")
    bases = blocks[..., :2].contiguous()
    if not torch.isfinite(bases.view(torch.float16)).all().item():
        raise ValueError("IQ3_XXS block bases must be finite")
    nt = (n + 127) // 128
    metadata = torch.zeros((nt, kb, 4352), dtype=torch.uint8, device=blocks.device)
    # Row padding is confined to metadata, preserving the descriptor matrix.
    base_rows = torch.zeros((nt * 128, kb, 2), dtype=torch.uint8, device=blocks.device)
    scale_rows = torch.zeros(
        (nt * 128, kb, 32), dtype=torch.uint8, device=blocks.device
    )
    base_rows[:n].copy_(bases)
    scale_rows[:n].copy_(blocks[..., 66:])
    metadata[..., :256].copy_(
        base_rows.view(nt, 128, kb, 2).permute(0, 2, 1, 3).reshape(nt, kb, 256)
    )
    metadata[..., 256:].copy_(
        scale_rows.view(nt, 128, kb, 32).permute(0, 2, 3, 1).reshape(nt, kb, 4096)
    )
    tile_n = 128 if n % 128 == 0 else 8
    values = (
        blocks[..., 2:66]
        .reshape(n // tile_n, tile_n, kb, 64)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n, kb * 64)
    )
    return IQ3XXSLinearWeight(values, metadata, kb * 256, n)

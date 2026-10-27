"""Lossless GGML MXFP4 byte-layout conversion for native E2M1 kernels."""

from __future__ import annotations

import torch


def split_gguf_mxfp4(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [E,N,K/32,17] blocks to adjacent E2M1 nibbles and E8M0 bytes.

    GGML byte j holds K positions j and j+16. Native packed E2M1 byte j
    holds positions 2*j and 2*j+1. This is only a permutation of nibbles;
    every scale byte and code (including signed zero) is retained. Output
    payload is exactly the input payload size. Preparation temporaries are
    bounded to 256 rows of one expert; no floating-point weights are created.
    """
    if blocks.dtype != torch.uint8:
        raise TypeError("GGUF MXFP4 blocks must be uint8")
    if blocks.ndim != 4 or blocks.shape[-1] != 17 or min(blocks.shape) <= 0:
        raise ValueError("GGUF MXFP4 blocks must have shape [E,N,K/32,17]")
    experts, rows, groups, _ = blocks.shape
    values = torch.empty(
        (experts, rows, groups, 16), dtype=torch.uint8, device=blocks.device
    )
    scales = torch.empty(
        (experts, rows, groups), dtype=torch.uint8, device=blocks.device
    )
    for expert in range(experts):
        for row in range(0, rows, 256):
            source = blocks[expert, row : row + 256]
            scales[expert, row : row + 256].copy_(source[..., 0])
            codes = source[..., 1:]
            even, odd = codes[..., ::2], codes[..., 1::2]
            values[expert, row : row + 256, :, :8].copy_(
                (even & 15) | ((odd & 15) << 4)
            )
            values[expert, row : row + 256, :, 8:].copy_((even >> 4) | (odd & 240))
    return values.reshape(experts, rows, groups * 16), scales


def prepare_gguf_mxfp4_fc2(blocks: torch.Tensor):
    """Prepare a native down projection without silently saturating source scales.

    The current BF16 consumer compensates scales in its register decode and
    supports bytes through 247. Reject larger source scales before calling
    its generic preparation, which otherwise saturates them.
    """
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_w4a16_fc2_e8m0_weights

    if not blocks.is_cuda:
        raise ValueError("native MXFP4 preparation requires CUDA blocks")
    values, scales = split_gguf_mxfp4(blocks)
    if bool((scales > 247).any()):
        raise ValueError(
            "GGUF MXFP4 scale exceeds the native BF16 consumer range; refusing saturation"
        )
    return prepare_w4a16_fc2_e8m0_weights(values, scales)

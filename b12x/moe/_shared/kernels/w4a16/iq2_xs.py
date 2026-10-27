"""Lossless preparation of GGML IQ2_XS, IQ2_XXS and IQ3_XXS expert blocks."""

from __future__ import annotations

import torch

from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut
from .prepare import PreparedW4A16MoeWeights, _make_workspace


def _pack_iq_matrix(
    blocks: torch.Tensor,
    *,
    swap_halves: bool = False,
    tile_descriptors: bool = False,
    tile_scales: bool = False,
    block_bytes: int = 74,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack uint8[E,N,K/256,block_bytes] into compact execution planes.

    Descriptors are uint16[E,K/16,N/16,8,2,2], exposed as int32 words.
    Each row pair joins output channels r and r+8 within an N16 tile.
    With tile_descriptors, words use [E,N/64,K/128,8,4,8,2] int32 tiles
    so each K128/N64 descriptor block is contiguous for bulk DMA.
    Metadata holds contiguous FP16[E,K/256,N/16,8,2] bases followed by
    uint8[E,K/256,N/16,8,8,2] scale pairs. With tile_scales, scale pairs use
    uint8[E,N/64,K/128,4,4,8,2] so each stage reads a contiguous 256 bytes.
    IQ3_XXS uses the same grid plane with uint32 rather than uint8 metadata
    words. The total is exactly 74 (IQ2_XS) or 98 (IQ3_XXS) bytes per 256
    weights. IQ2_XXS uses two-byte grid pairs in the same logical tiles,
    exposed as int32 words, with 34 metadata bytes for 66 total bytes.
    IQ2_S keeps 64 grid/sign bytes plus 18 metadata bytes. IQ3_S keeps
    64 grid bytes plus 46 metadata bytes, tiled as K64 groups of eleven
    byte planes (high bits, signs and subscales) plus the base plane.
    Temporary copies contain at most 256 output rows of one expert.
    """
    if block_bytes not in {66, 74, 82, 98, 110}:
        raise ValueError("unsupported IQ block size")
    recipe = "IQ3_S" if block_bytes == 110 else "IQ2_S" if block_bytes == 82 else "IQ2_XXS" if block_bytes == 66 else "IQ3_XXS" if block_bytes == 98 else "IQ2_XS"
    if blocks.dtype != torch.uint8:
        raise TypeError(f"{recipe} blocks must be uint8")
    if blocks.ndim != 4 or blocks.shape[-1] != block_bytes:
        raise ValueError(f"{recipe} blocks must have shape [E,N,K/256,{block_bytes}]")
    e, n, kb, _ = blocks.shape
    if min(e, n, kb) <= 0 or n % 16:
        raise ValueError(f"{recipe} requires positive geometry and N divisible by 16")
    if swap_halves and n % 32:
        raise ValueError(f"{recipe} projection halves must be divisible by 16")
    if (tile_descriptors or tile_scales) and (n % 64 or (swap_halves and n % 128)):
        raise ValueError(
            f"tiled {recipe} requires N and projection halves divisible by 64"
        )
    words = torch.empty(
        (e, n // 64, kb * 2, 8, 4, 8, 2)
        if tile_descriptors
        else (e, kb * 16, n // 16, 8, 2),
        dtype=torch.int16 if block_bytes == 66 else torch.int32,
        device=blocks.device,
    )
    metadata = torch.empty(
        e * kb * n * (block_bytes - (32 if block_bytes == 66 else 64)), dtype=torch.uint8, device=blocks.device
    )
    base_bytes = e * kb * n * 2
    bases = metadata[:base_bytes].view(torch.float16).reshape(e, kb, n // 16, 8, 2)
    if block_bytes == 110:
        if not tile_scales:
            raise ValueError("IQ3_S requires tiled K64 metadata planes")
        scales = metadata[base_bytes:].reshape(e, n // 64, kb * 4, 11, 4, 8, 2)
    else:
        scales = (
            metadata[base_bytes:]
            .view(torch.int16 if block_bytes == 82 else torch.int32 if block_bytes in {66, 98} else torch.uint8)
            .reshape(
                (e, n // 64, kb, 2, 4, 4, 8, 2)
                if tile_scales
                else (e, kb, n // 16, 8, 8, 2)
            )
        )
    for expert in range(e):
        boundaries = (0, n // 2, n) if swap_halves else (0, n)
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            for row in range(begin, end, 256):
                stop = min(row + 256, end)
                source_row = (row + n // 2) % n if swap_halves else row
                chunk = blocks[expert, source_row : source_row + stop - row]
                d = (
                    chunk[..., :2]
                    .contiguous()
                    .view(torch.float16)
                    .reshape(stop - row, kb)
                )
                if not bool(torch.isfinite(d).all()):
                    raise ValueError(f"{recipe} block bases must be finite")
                if block_bytes == 66:
                    payload = chunk[..., 2:].reshape(stop - row, kb, 8, 8)
                    qs = payload[..., :4].contiguous().view(torch.int16)
                    descriptors = (
                        qs.reshape(stop - row, kb * 16)
                        .T.contiguous().reshape(kb * 16, (stop - row) // 16, 2, 8)
                        .transpose(-2, -1)
                    )
                    gas = payload[..., 4:].contiguous()
                else:
                    if block_bytes == 82:
                        qs = torch.stack((chunk[..., 2:34], chunk[..., 34:66]), dim=-1).contiguous().view(torch.uint16)
                        gas = torch.stack((chunk[..., 66:74], chunk[..., 74:82]), dim=-1)
                    else:
                        qs = chunk[..., 2:66].contiguous().view(torch.uint16)
                        gas = chunk[..., 66:]
                    q = qs.reshape(stop - row, kb * 16, 2).permute(1, 0, 2).contiguous()
                    descriptors = (
                        q.view(torch.int32)
                        .reshape(kb * 16, (stop - row) // 16, 2, 8)
                        .transpose(-2, -1)
                    )
                if tile_descriptors:
                    words[expert, row // 64 : stop // 64].copy_(
                        descriptors.reshape(
                            kb * 2, 8, (stop - row) // 64, 4, 8, 2
                        ).permute(2, 0, 1, 3, 4, 5)
                    )
                else:
                    words[expert, :, row // 16 : stop // 16].copy_(descriptors)
                bases[expert, :, row // 16 : stop // 16].copy_(
                    d.T.reshape(kb, (stop - row) // 16, 2, 8).transpose(-2, -1)
                )
                if block_bytes == 110:
                    # K64 metadata: two qh bytes, eight sign bytes, one scale.
                    planes = torch.cat((chunk[..., 66:74].reshape(stop-row, kb, 4, 2),
                                        chunk[..., 74:106].reshape(stop-row, kb, 4, 8),
                                        chunk[..., 106:110].reshape(stop-row, kb, 4, 1)), dim=-1)
                    scales[expert, row // 64:stop // 64].copy_(
                        planes.reshape((stop-row)//64, 4, 2, 8, kb, 4, 11)
                        .permute(0, 4, 5, 6, 1, 3, 2)
                        .reshape((stop-row)//64, kb*4, 11, 4, 8, 2)
                    )
                elif tile_scales:
                    scales[expert, row // 64 : stop // 64].copy_(
                        gas
                        .contiguous()
                        .view(torch.int16 if block_bytes == 82 else torch.int32 if block_bytes in {66, 98} else torch.uint8)
                        .reshape((stop - row) // 64, 4, 2, 8, kb, 2, 4)
                        .permute(0, 4, 5, 6, 1, 3, 2)
                    )
                else:
                    scales[expert, :, row // 16 : stop // 16].copy_(
                        gas
                        .contiguous()
                        .view(torch.int16 if block_bytes == 82 else torch.int32 if block_bytes in {66, 98} else torch.uint8)
                        .reshape((stop - row) // 16, 2, 8, kb, 8)
                        .permute(3, 0, 4, 2, 1)
                    )
    return words.reshape(-1).view(torch.int32), metadata


def pack_iq2_xs_matrix(
    blocks, *, swap_halves=False, tile_descriptors=False, tile_scales=False
):
    return _pack_iq_matrix(
        blocks,
        swap_halves=swap_halves,
        tile_descriptors=tile_descriptors,
        tile_scales=tile_scales,
    )


def pack_iq3_xxs_matrix(
    blocks, *, swap_halves=False, tile_descriptors=False, tile_scales=False
):
    return _pack_iq_matrix(
        blocks,
        swap_halves=swap_halves,
        tile_descriptors=tile_descriptors,
        tile_scales=tile_scales,
        block_bytes=98,
    )


def _prepare_iq_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    w13_layout: str,
    recipe: str = "iq2_xs",
    fc2_format: str | None = None,
) -> PreparedW4A16MoeWeights:
    """Prepare compact IQ storage and device tables before execution."""
    if recipe not in {"iq2_xs", "iq3_xxs", "iq2_xxs", "iq2_s", "iq3_s"}:
        raise ValueError(f"unsupported IQ recipe {recipe!r}")
    if fc2_format not in {None, "mxfp4", "iq3_xxs", "iq3_s"}:
        raise ValueError(f"unsupported FC2 format {fc2_format!r}")
    if fc2_format == "iq3_xxs" and recipe != "iq2_xxs":
        raise ValueError("IQ3_XXS mixed down requires IQ2_XXS gate/up")
    if fc2_format == "iq3_s" and recipe not in {"iq2_s", "iq3_xxs"}:
        raise ValueError("IQ3_S mixed down requires IQ2_S or IQ3_XXS gate/up")
    if activation not in {"silu", "relu2"}:
        raise ValueError(f"{recipe} supports SiLU and ReLU²")
    if w13_layout not in {"w13", "w31"}:
        raise ValueError(f"{recipe} requires w13 or w31 projection order")
    if (
        min(hidden_size, intermediate_size, num_experts) <= 0
        or hidden_size % 256
        or intermediate_size % 256
    ):
        raise ValueError(
            f"{recipe} requires positive geometry and H/I divisible by 256"
        )
    if not w13.is_cuda or w2.device != w13.device:
        raise ValueError(f"{recipe} preparation requires tensors on one CUDA device")
    block_bytes = 110 if recipe == "iq3_s" else 82 if recipe == "iq2_s" else 66 if recipe == "iq2_xxs" else 98 if recipe == "iq3_xxs" else 74
    pack_matrix = pack_iq3_s_matrix if recipe == "iq3_s" else pack_iq2_s_matrix if recipe == "iq2_s" else pack_iq2_xxs_matrix if recipe == "iq2_xxs" else pack_iq3_xxs_matrix if recipe == "iq3_xxs" else pack_iq2_xs_matrix
    gated = activation == "silu"
    expected13 = (
        num_experts,
        intermediate_size * (2 if gated else 1),
        hidden_size // 256,
        block_bytes,
    )
    expected2 = (num_experts, hidden_size, intermediate_size // 256, block_bytes)
    if fc2_format == "iq3_s":
        expected2 = (num_experts, hidden_size, intermediate_size // 256, 110)
    if fc2_format == "iq3_xxs":
        expected2 = (num_experts, hidden_size, intermediate_size // 256, 98)
    if fc2_format == "mxfp4":
        expected2 = (num_experts, hidden_size, intermediate_size // 32, 17)
    if tuple(w13.shape) != expected13 or tuple(w2.shape) != expected2:
        raise ValueError(
            f"{recipe} blocks must have shapes {expected13} and {expected2}"
        )
    q13, s13 = pack_matrix(
        w13,
        swap_halves=gated and w13_layout == "w13",
        tile_descriptors=True,
        tile_scales=True,
    )
    if fc2_format == "mxfp4":
        from b12x.loader.gguf_mxfp4 import split_gguf_mxfp4
        from .prepare import _repack_weight, _pack_e8m0_k32_scales

        values, scales = split_gguf_mxfp4(w2)
        if bool((scales > 247).any()):
            raise ValueError(
                "MXFP4 scale exceeds native BF16 range; refusing saturation"
            )
        q2 = _repack_weight(values, size_k=intermediate_size, size_n=hidden_size)
        s2 = _pack_e8m0_k32_scales(scales, size_k=intermediate_size, size_n=hidden_size)
    else:
        down_packer = pack_iq3_s_matrix if fc2_format == "iq3_s" else pack_iq3_xxs_matrix if fc2_format == "iq3_xxs" else pack_matrix
        q2, s2 = down_packer(w2, tile_descriptors=True, tile_scales=True)
    if recipe == "iq3_s":
        from b12x._lib.quant.iq3_s import iq3_s_execution_lut

        iq3_s_execution_lut(w13.device, prepare=True)
    elif recipe == "iq2_s":
        from b12x._lib.quant.iq2_s import iq2_s_execution_lut

        iq2_s_execution_lut(w13.device, prepare=True)
    elif recipe == "iq2_xxs":
        from b12x._lib.quant.iq2_xxs import iq2_xxs_execution_lut

        iq2_xxs_execution_lut(w13.device, prepare=True)
    elif recipe == "iq3_xxs":
        from b12x._lib.quant.iq3_xxs import iq3_xxs_execution_lut

        iq3_xxs_execution_lut(w13.device, prepare=True)
    else:
        iq2_xs_execution_lut(w13.device, prepare=True, selectors=True)
    if fc2_format == "iq3_xxs":
        from b12x._lib.quant.iq3_xxs import iq3_xxs_execution_lut

        iq3_xxs_execution_lut(w13.device, prepare=True)
    if fc2_format == "iq3_s":
        from b12x._lib.quant.iq3_s import iq3_s_execution_lut

        iq3_s_execution_lut(w13.device, prepare=True)
    unit = torch.ones(num_experts, dtype=torch.float32, device=w13.device)
    return PreparedW4A16MoeWeights(
        w13=q13,
        w13_scale=s13,
        w13_global_scale=unit,
        w2=q2,
        w2_scale=s2,
        w2_global_scale=unit,
        workspace=_make_workspace(w13.device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=gated,
        params_dtype=torch.bfloat16,
        fc1_tile_n=64,
        fc2_tile_n=64,
        source_format=recipe,
        fc2_format=fc2_format,
        w13_layout="packed",
        weight_layout=recipe,
        scale_format=recipe,
    )


def prepare_iq2_xs_moe_weights(w13, w2, **kwargs):
    return _prepare_iq_moe_weights(w13, w2, recipe="iq2_xs", **kwargs)


def prepare_iq3_xxs_moe_weights(w13, w2, **kwargs):
    return _prepare_iq_moe_weights(w13, w2, recipe="iq3_xxs", **kwargs)


def pack_iq2_xxs_matrix(blocks, *, swap_halves=False, tile_descriptors=False, tile_scales=False):
    return _pack_iq_matrix(blocks, swap_halves=swap_halves,
                          tile_descriptors=tile_descriptors, tile_scales=tile_scales,
                          block_bytes=66)


def prepare_iq2_xxs_moe_weights(w13, w2, **kwargs):
    return _prepare_iq_moe_weights(w13, w2, recipe="iq2_xxs", **kwargs)


def pack_iq2_s_matrix(blocks, *, swap_halves=False, tile_descriptors=False, tile_scales=False):
    return _pack_iq_matrix(blocks, swap_halves=swap_halves,
                          tile_descriptors=tile_descriptors, tile_scales=tile_scales,
                          block_bytes=82)


def prepare_iq2_s_moe_weights(w13, w2, **kwargs):
    return _prepare_iq_moe_weights(w13, w2, recipe="iq2_s", **kwargs)


def pack_iq3_s_matrix(blocks, *, swap_halves=False, tile_descriptors=True, tile_scales=True):
    return _pack_iq_matrix(blocks, swap_halves=swap_halves,
                          tile_descriptors=tile_descriptors, tile_scales=tile_scales,
                          block_bytes=110)


def prepare_iq3_s_moe_weights(w13, w2, **kwargs):
    return _prepare_iq_moe_weights(w13, w2, recipe="iq3_s", **kwargs)

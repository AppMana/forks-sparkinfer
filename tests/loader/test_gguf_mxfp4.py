"""Exact source-code preservation across GGML/native MXFP4 layouts."""

import numpy as np
import pytest
import torch

from b12x.loader.gguf_mxfp4 import split_gguf_mxfp4


def test_all_codes_scales_and_row_chunk_boundary():
    gguf = pytest.importorskip("gguf")
    raw = np.empty((2, 257, 1, 17), dtype=np.uint8)
    raw[..., 0] = np.arange(257, dtype=np.uint16).astype(np.uint8)[None, :, None]
    raw[..., 1:] = np.arange(16, dtype=np.uint8) * 17
    blocks = torch.from_numpy(raw)
    before = blocks.clone()
    values, scales = split_gguf_mxfp4(blocks)
    assert values.numel() + scales.numel() == blocks.numel()
    assert torch.equal(blocks, before)
    assert torch.equal(scales, blocks[..., 0])
    adjacent = torch.stack((values & 15, values >> 4), dim=-1).reshape(2, 257, 32)
    codes = torch.cat((blocks[..., 1:] & 15, blocks[..., 1:] >> 4), dim=-1).reshape_as(
        adjacent
    )
    assert torch.equal(adjacent, codes)
    # Use GGUF's independent floating-point decoder to check the numeric mapping.
    lut = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float32,
    )
    scale = np.exp2(scales.numpy().astype(np.float64) - 127)
    actual = (lut[adjacent.numpy()] * scale).astype(np.float32)
    expected = gguf.dequantize(
        raw.reshape(2 * 257, 17), gguf.GGMLQuantizationType.MXFP4
    ).reshape(actual.shape)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 128, 8, 16), torch.uint8),
        ((1, 128, 8, 17), torch.float32),
        ((0, 128, 8, 17), torch.uint8),
    ],
)
def test_invalid_blocks(shape, dtype):
    with pytest.raises((TypeError, ValueError)):
        split_gguf_mxfp4(torch.empty(shape, dtype=dtype))


@pytest.mark.parametrize("scale", [248, 255])
def test_native_preparation_refuses_scale_saturation(scale):
    from b12x.loader.gguf_mxfp4 import prepare_gguf_mxfp4_fc2
    from tests._reference.helpers import require_b12x

    device = require_b12x()
    blocks = torch.zeros((1, 128, 8, 17), dtype=torch.uint8, device=device)
    blocks[..., 0] = scale
    with pytest.raises(ValueError, match="refusing saturation"):
        prepare_gguf_mxfp4_fc2(blocks)

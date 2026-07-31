"""GPU correctness coverage for the vLLM DSV4 INT8 indexer contract."""

from __future__ import annotations

import pytest
import torch

from sparkinfer.attention.nsa_indexer.kernel import run_paged_logits_kernel
from sparkinfer.attention.nsa_indexer.contiguous_kernel import (
    run_contiguous_logits_kernel,
)


_PAGE_SIZE = 64
_HEAD_DIM = 128
_HEADS = 64
_SCALE_BYTES = 4
_TOKEN_BYTES = _HEAD_DIM + _SCALE_BYTES
_CACHE_PAGES = 6_650
_PHYSICAL_PAGE_STRIDE = 438_784
_LOW_PAGE = 1
_HIGH_PAGE = 5_712
_LIVE_CONTEXT = 10_769
_MAX_MODEL_LEN = 16_384
_PAGE_TABLE_WIDTH = _MAX_MODEL_LEN // _PAGE_SIZE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1),
    reason="SM121 CUDA GPU required",
)


def _make_int8_contract() -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    q = torch.full((1, _HEADS, _HEAD_DIM), -1, dtype=torch.int8, device=device)
    weights = (
        torch.arange(1, _HEADS + 1, dtype=torch.float32, device=device) / _HEADS
    ).reshape(1, _HEADS)
    cache = torch.empty_strided(
        (_CACHE_PAGES, _PAGE_SIZE, 1, _TOKEN_BYTES),
        (_PHYSICAL_PAGE_STRIDE, _TOKEN_BYTES, _TOKEN_BYTES, 1),
        dtype=torch.uint8,
        device=device,
    )
    for physical_page, scale in ((_LOW_PAGE, 0.01), (_HIGH_PAGE, 0.02)):
        cache[physical_page, ..., :_HEAD_DIM].view(torch.int8).fill_(-1)
        cache[physical_page, ..., _HEAD_DIM:].view(torch.float32).fill_(scale)

    live_pages = (_LIVE_CONTEXT + _PAGE_SIZE - 1) // _PAGE_SIZE
    page_table = torch.full(
        (1, _PAGE_TABLE_WIDTH), -1, dtype=torch.int32, device=device
    )
    page_table[0, :live_pages:2] = _LOW_PAGE
    page_table[0, 1:live_pages:2] = _HIGH_PAGE
    seqlens = torch.tensor([_LIVE_CONTEXT], dtype=torch.int32, device=device)
    active_width = torch.tensor([_LIVE_CONTEXT], dtype=torch.int32, device=device)
    return q, weights, cache, page_table, seqlens, active_width


def test_int8_interleaved_vllm_contract_and_high_page_addressing() -> None:
    """Match the live 10,769-token launch and cross the signed-32-bit boundary."""
    q, weights, cache, page_table, seqlens, active_width = _make_int8_contract()
    assert q.shape == (1, 64, 128)
    assert q.stride() == (8_192, 128, 1)
    assert cache.shape == (6_650, 64, 1, 132)
    assert cache.stride() == (438_784, 132, 132, 1)
    assert page_table.shape == (1, 256)
    live_pages = (_LIVE_CONTEXT + _PAGE_SIZE - 1) // _PAGE_SIZE
    assert live_pages == 169
    assert int(page_table[0, :live_pages].min().item()) == _LOW_PAGE
    assert int(page_table[0, :live_pages].max().item()) == _HIGH_PAGE
    assert (_HIGH_PAGE * cache.stride(0)) > (2**31 - 1)

    logits = run_paged_logits_kernel(
        q_fp8=q,
        weights=weights,
        index_k_cache=cache,
        real_page_table=page_table,
        seqlens_per_query=seqlens,
        active_width=active_width,
        page_size=_PAGE_SIZE,
    )
    torch.cuda.synchronize()

    weight_sum = float(weights.sum().item())
    low_expected = _HEAD_DIM * 0.01 * weight_sum
    high_expected = _HEAD_DIM * 0.02 * weight_sum
    expected_page = torch.cat(
        (
            torch.full((_PAGE_SIZE,), low_expected, device=q.device),
            torch.full((_PAGE_SIZE,), high_expected, device=q.device),
        )
    )
    repeats = (_LIVE_CONTEXT + expected_page.numel() - 1) // expected_page.numel()
    expected = expected_page.repeat(repeats)[:_LIVE_CONTEXT]
    torch.testing.assert_close(
        logits[0, :_LIVE_CONTEXT],
        expected,
        atol=2e-4,
        rtol=2e-4,
    )
    assert torch.isneginf(logits[0, _LIVE_CONTEXT:]).all()


def _contiguous_eager_reference(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
) -> torch.Tensor:
    dots = torch.einsum("mhd,nd->mhn", q.to(torch.float32), k.to(torch.float32))
    logits = (dots.relu() * weights[:, :, None]).sum(dim=1)
    logits *= scales[None, :]
    cols = torch.arange(k.shape[0], device=q.device)
    valid = (cols[None, :] >= k_start[:, None]) & (cols[None, :] < k_end[:, None])
    return logits.masked_fill(~valid, float("-inf"))


def test_int8_contiguous_prefill_matches_eager_oracle() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randint(
        -17,
        18,
        (33, _HEADS, _HEAD_DIM),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    k = torch.randint(
        -19,
        20,
        (321, _HEAD_DIM),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    weights = torch.randn(
        (33, _HEADS), dtype=torch.float32, device="cuda", generator=generator
    )
    scales = (
        torch.rand((321,), dtype=torch.float32, device="cuda", generator=generator)
        * 0.03
    )
    k_start = torch.arange(33, dtype=torch.int32, device="cuda") % 29
    k_end = 321 - (torch.arange(33, dtype=torch.int32, device="cuda") % 31)

    actual = run_contiguous_logits_kernel(
        q_fp8=q,
        weights=weights,
        k_quant=k,
        k_scale=scales,
        k_start=k_start,
        k_end=k_end,
    )
    expected = _contiguous_eager_reference(q, weights, k, scales, k_start, k_end)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-5)


@pytest.mark.parametrize("rows,context", [(8_192, 2_048), (1_024, 16_384)])
def test_int8_contiguous_prefill_target_shapes(rows: int, context: int) -> None:
    q = torch.full((rows, _HEADS, _HEAD_DIM), -1, dtype=torch.int8, device="cuda")
    k = torch.full((context, _HEAD_DIM), -1, dtype=torch.int8, device="cuda")
    weights = torch.full(
        (rows, _HEADS), 1.0 / _HEADS, dtype=torch.float32, device="cuda"
    )
    scales = torch.linspace(0.005, 0.02, context, dtype=torch.float32, device="cuda")
    k_start = torch.zeros(rows, dtype=torch.int32, device="cuda")
    k_end = torch.full((rows,), context, dtype=torch.int32, device="cuda")

    actual = run_contiguous_logits_kernel(
        q_fp8=q,
        weights=weights,
        k_quant=k,
        k_scale=scales,
        k_start=k_start,
        k_end=k_end,
        preinitialize_invalid_logits=False,
    )
    expected_row = _HEAD_DIM * scales
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected_row.expand(rows, context),
        atol=2e-4,
        rtol=2e-4,
    )

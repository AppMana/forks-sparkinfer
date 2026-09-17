"""Capture exact DSV4 mHC inputs and compare SparkInfer with eager math.

This module is mounted as ``sitecustomize.py`` only in a disposable diagnostic
JobSet.  It is inactive unless ``SPARKINFER_MHC_CAPTURE_DIR`` is set and the
``capture.enable`` sentinel exists in that directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from sparkinfer.norm import mhc


_CAPTURE_DIR = Path(os.environ.get("SPARKINFER_MHC_CAPTURE_DIR", "/nonexistent"))
_RANK = os.environ.get("RANK", "unknown")
_ORIGINAL_PRE = mhc.run_pre
_ORIGINAL_POST_PRE = mhc.run_post_pre
_CALL_INDEX = 0
_SAVED_DIVERGENCE = False


def _enabled() -> bool:
    return bool(os.environ.get("SPARKINFER_MHC_CAPTURE_DIR")) and (
        _CAPTURE_DIR / "capture.enable"
    ).exists()


def _pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
    if norm_weight is not None:
        y_float = y.float()
        y = (
            y_float
            * torch.rsqrt(y_float.square().mean(dim=-1, keepdim=True) + norm_eps)
            * norm_weight.float()
        ).to(torch.bfloat16)
    return post, comb, y


def _post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if post.ndim == 3:
        post = post.squeeze(-1)
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(torch.bfloat16)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = actual.float() - expected.float()
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "nonfinite": int((~torch.isfinite(actual)).sum().item()),
        "different": int((actual != expected).sum().item()),
    }


def _capture(
    kind: str,
    inputs: dict[str, object],
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    global _CALL_INDEX, _SAVED_DIVERGENCE
    call_index = _CALL_INDEX
    _CALL_INDEX += 1
    metrics = [_metrics(a, e) for a, e in zip(actual, expected, strict=True)]
    limits = (2e-2, 2e-4, 2e-4, 2e-2)
    divergent = any(row["nonfinite"] or row["max_abs"] > limit for row, limit in zip(metrics, limits, strict=True))
    print(
        "SPARKINFER_MHC_REAL_PARITY "
        + json.dumps(
            {
                "rank": _RANK,
                "call": call_index,
                "kind": kind,
                "tokens": int(actual[0].shape[0]),
                "metrics": metrics,
                "divergent": divergent,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    if call_index == 0:
        paths.append(_CAPTURE_DIR / f"mhc-first-call-rank{_RANK}.pt")
    if divergent and not _SAVED_DIVERGENCE:
        _SAVED_DIVERGENCE = True
        paths.append(
            _CAPTURE_DIR / f"mhc-first-divergence-rank{_RANK}-call{call_index}.pt"
        )
    if not paths:
        return
    payload = {
        "rank": _RANK,
        "call": call_index,
        "kind": kind,
        "inputs": {
            name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        },
        "actual": tuple(t.detach().cpu() for t in actual),
        "expected": tuple(t.detach().cpu() for t in expected),
        "metrics": metrics,
    }
    for path in paths:
        torch.save(payload, path)


def _run_pre(*args, **kwargs):
    actual = _ORIGINAL_PRE(*args, **kwargs)
    if not _enabled():
        return actual
    residual, fn, scale, bias = args[:4]
    residual_ref = residual[:, None, :].expand(-1, 4, -1).contiguous()
    post, comb, y = _pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=float(kwargs["rms_eps"]),
        hc_eps=float(kwargs["hc_eps"]),
        sinkhorn_iters=int(kwargs["sinkhorn_iters"]),
        norm_weight=kwargs.get("norm_weight"),
        norm_eps=float(kwargs.get("norm_eps", 0.0)),
    )
    expected = (residual_ref, post, comb, y)
    _capture(
        "pre",
        {
            "residual": residual,
            "fn": fn,
            "scale": scale,
            "bias": bias,
            "norm_weight": kwargs.get("norm_weight"),
            "rms_eps": float(kwargs["rms_eps"]),
            "hc_eps": float(kwargs["hc_eps"]),
            "sinkhorn_iters": int(kwargs["sinkhorn_iters"]),
            "norm_eps": float(kwargs.get("norm_eps", 0.0)),
            "split_k": int(kwargs.get("split_k", mhc.DEFAULT_SPLIT_K)),
        },
        actual,
        expected,
    )
    return actual


def _run_post_pre(*args, **kwargs):
    actual = _ORIGINAL_POST_PRE(*args, **kwargs)
    if not _enabled():
        return actual
    x, residual, prev_post, prev_comb, fn, scale, bias = args[:7]
    residual_ref = _post_reference(x, residual, prev_post, prev_comb)
    post, comb, y = _pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=float(kwargs["rms_eps"]),
        hc_eps=float(kwargs["hc_eps"]),
        sinkhorn_iters=int(kwargs["sinkhorn_iters"]),
        norm_weight=kwargs.get("norm_weight"),
        norm_eps=float(kwargs.get("norm_eps", 0.0)),
    )
    expected = (residual_ref, post, comb, y)
    _capture(
        "post_pre",
        {
            "x": x,
            "residual": residual,
            "prev_post": prev_post,
            "prev_comb": prev_comb,
            "fn": fn,
            "scale": scale,
            "bias": bias,
            "norm_weight": kwargs.get("norm_weight"),
            "rms_eps": float(kwargs["rms_eps"]),
            "hc_eps": float(kwargs["hc_eps"]),
            "sinkhorn_iters": int(kwargs["sinkhorn_iters"]),
            "norm_eps": float(kwargs.get("norm_eps", 0.0)),
            "split_k": int(kwargs.get("split_k", mhc.DEFAULT_SPLIT_K)),
        },
        actual,
        expected,
    )
    return actual


if os.environ.get("SPARKINFER_MHC_CAPTURE_DIR"):
    mhc.run_pre = _run_pre
    mhc.run_post_pre = _run_post_pre

"""Typed checkpoint tensor bundles accepted by fused-MoE preparation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from b12x.loader.gguf_experts import ExpertTensor
    from ._impl import B12XFP4ExpertWeights
    from .planning import WeightPlan


class WeightEncoding(str, Enum):
    """Numeric encoding of prepared weight elements."""

    FP4_E2M1 = "fp4_e2m1"
    FP6_E2M3 = "fp6_e2m3"
    TRELLIS = "trellis"
    IQ2_XS = "iq2_xs"
    IQ3_XXS = "iq3_xxs"
    IQ2_XXS = "iq2_xxs"
    IQ2_S = "iq2_s"
    IQ3_S = "iq3_s"


class ScaleEncoding(str, Enum):
    """Scale encoding consumed with the prepared weights."""

    E4M3_K16 = "e4m3_k16"
    E4M3_K32 = "e4m3_k32"
    E8M0_K32 = "e8m0_k32"
    E8M0_K32_E4M3_RESIDUAL = "e8m0_k32_x_e4m3_k16_residual"
    TRELLIS_SCALES = "trellis_scales"
    IQ2_XS = "iq2_xs"
    IQ3_XXS = "iq3_xxs"
    IQ2_XXS = "iq2_xxs"
    IQ2_S = "iq2_s"
    IQ3_S = "iq3_s"


class WeightPacking(str, Enum):
    """In-memory packing or zero-copy view exposed after preparation."""

    SOURCE_NATIVE = "source_native"
    MMA_VIEW = "mma_view"
    MMA_PACKED = "mma_packed"
    QMMA_REPACKED = "qmma_repacked"
    TRELLIS_NATIVE = "trellis_native"
    IQ2_XS_COMPACT = "iq2_xs_compact"
    IQ3_XXS_COMPACT = "iq3_xxs_compact"
    IQ2_XXS_COMPACT = "iq2_xxs_compact"
    IQ2_S_COMPACT = "iq2_s_compact"
    IQ3_S_COMPACT = "iq3_s_compact"


@dataclass(frozen=True, kw_only=True)
class PreparedWeightFormat:
    """Numeric and physical contract produced by weight preparation."""

    weights: WeightEncoding
    scales: ScaleEncoding
    packing: WeightPacking
    available_packings: frozenset[WeightPacking]
    fc2_weights: WeightEncoding | None = None
    fc2_scales: ScaleEncoding | None = None
    fc2_packing: WeightPacking | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", WeightEncoding(self.weights))
        object.__setattr__(self, "scales", ScaleEncoding(self.scales))
        object.__setattr__(self, "packing", WeightPacking(self.packing))
        available = frozenset(WeightPacking(value) for value in self.available_packings)
        if self.packing not in available:
            raise ValueError("packing must be present in available_packings")
        object.__setattr__(self, "available_packings", available)
        down = (self.fc2_weights, self.fc2_scales, self.fc2_packing)
        if any(value is not None for value in down):
            if any(value is None for value in down):
                raise ValueError("FC2 representation must specify encoding, scales and packing")
            object.__setattr__(self, "fc2_weights", WeightEncoding(self.fc2_weights))
            object.__setattr__(self, "fc2_scales", ScaleEncoding(self.fc2_scales))
            object.__setattr__(self, "fc2_packing", WeightPacking(self.fc2_packing))


@dataclass(frozen=True)
class ScaleFactors:
    """One scale boundary represented as vectors times optional gains."""

    vectors: torch.Tensor
    gains: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.vectors, torch.Tensor):
            raise TypeError("ScaleFactors.vectors must be a torch.Tensor")
        if self.gains is not None and not isinstance(self.gains, torch.Tensor):
            raise TypeError("ScaleFactors.gains must be a torch.Tensor or None")


@dataclass(frozen=True)
class TrellisWeights:
    """Layer-local views of the canonical ``b12x_trellis`` tensors.

    ``codes`` is the rank-local ``[I_local/32, row_stride]`` uint8 payload.
    ``rate`` is a view selected from the single model-level uint8 rate tensor;
    it is never copied merely to give each layer its own rate parameter.
    """

    codes: torch.Tensor
    rate: torch.Tensor
    input_scales: ScaleFactors
    intermediate_scales: ScaleFactors
    output_scales: ScaleFactors
    expert_sign_patterns: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.codes, torch.Tensor):
            raise TypeError("TrellisWeights.codes must be a torch.Tensor")
        if not isinstance(self.rate, torch.Tensor):
            raise TypeError("TrellisWeights.rate must be a torch.Tensor")
        for name in (
            "input_scales",
            "intermediate_scales",
            "output_scales",
        ):
            if not isinstance(getattr(self, name), ScaleFactors):
                raise TypeError(f"TrellisWeights.{name} must be ScaleFactors")
        if self.expert_sign_patterns is not None and not isinstance(
            self.expert_sign_patterns, torch.Tensor
        ):
            raise TypeError(
                "TrellisWeights.expert_sign_patterns must be a tensor or None"
            )


@dataclass(frozen=True)
class GGUFExpertWeights:
    """File-backed gate/up/down sources and ordered EP ownership for streaming.

    Expert IDs map local slots to global IDs. Supply the same mapping when
    binding execution; preparation does not perform routing or collectives.
    """

    gate: "ExpertTensor"
    up: "ExpertTensor"
    down: "ExpertTensor"
    expert_ids: tuple[int, ...]
    device: torch.device

    def __post_init__(self):
        from b12x.loader.gguf_experts import ExpertTensor

        if not all(isinstance(source, ExpertTensor) for source in (self.gate, self.up, self.down)):
            raise TypeError("GGUFExpertWeights requires ExpertTensor sources")
        object.__setattr__(self, "expert_ids", tuple(self.expert_ids))
        object.__setattr__(self, "device", torch.device(self.device))


@dataclass(frozen=True)
class IQ2XSWeights:
    """Safetensors uint8[E,N,K/256,74] blocks in PackedSource.w13_layout order."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("w13", "w2"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
                raise TypeError(f"IQ2XSWeights.{name} must be a uint8 tensor")
            if tensor.ndim != 4 or tensor.shape[-1] != 74:
                raise ValueError(f"IQ2XSWeights.{name} must have shape [E,N,K/256,74]")


@dataclass(frozen=True)
class IQ3XXSWeights:
    """Safetensors uint8[E,N,K/256,98] blocks in PackedSource.w13_layout order."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("w13", "w2"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
                raise TypeError(f"IQ3XXSWeights.{name} must be a uint8 tensor")
            if tensor.ndim != 4 or tensor.shape[-1] != 98:
                raise ValueError(f"IQ3XXSWeights.{name} must have shape [E,N,K/256,98]")


@dataclass(frozen=True)
class IQ2XXSWeights:
    """Safetensors uint8[E,N,K/256,66] blocks in PackedSource.w13_layout order."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("w13", "w2"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
                raise TypeError(f"IQ2XXSWeights.{name} must be a uint8 tensor")
            if tensor.ndim != 4 or tensor.shape[-1] != 66:
                raise ValueError(f"IQ2XXSWeights.{name} must have shape [E,N,K/256,66]")


@dataclass(frozen=True)
class IQ2SWeights:
    """Safetensors uint8[E,N,K/256,82] blocks in PackedSource.w13_layout order."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("w13", "w2"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
                raise TypeError(f"IQ2SWeights.{name} must be a uint8 tensor")
            if tensor.ndim != 4 or tensor.shape[-1] != 82:
                raise ValueError(f"IQ2SWeights.{name} must have shape [E,N,K/256,82]")


@dataclass(frozen=True)
class IQ3SWeights:
    """Safetensors uint8[E,N,K/256,110] blocks in PackedSource.w13_layout order."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("w13", "w2"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
                raise TypeError(f"IQ3SWeights.{name} must be a uint8 tensor")
            if tensor.ndim != 4 or tensor.shape[-1] != 110:
                raise ValueError(f"IQ3SWeights.{name} must have shape [E,N,K/256,110]")


@dataclass(frozen=True)
class IQMXFP4Weights:
    """Raw GGUF IQ gate/up blocks and independent MXFP4 down blocks."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        for name, sizes in (("w13", {66, 74, 82, 98}), ("w2", {17})):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
                raise TypeError(f"IQMXFP4Weights.{name} must be uint8")
            if tensor.ndim != 4 or tensor.shape[-1] not in sizes:
                raise ValueError(f"IQMXFP4Weights.{name} requires rank-4 GGUF blocks with payload size in {sizes}")


@dataclass(frozen=True)
class IQMixedWeights:
    """Compact GGUF blocks for supported mixed IQ projection pairs."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        for name, sizes in (("w13", {66, 82, 98}), ("w2", {98, 110})):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
                raise TypeError(f"IQMixedWeights.{name} must be uint8")
            if tensor.ndim != 4 or tensor.shape[-1] not in sizes:
                raise ValueError(f"IQMixedWeights.{name} requires rank-4 GGUF blocks with payload size in {sizes}")


@dataclass(frozen=True)
class PackedWeights:
    """Ordinary packed MoE checkpoint tensors, without runtime policy fields.

    Activation scales are optional source metadata. A16 preparation uses unit
    scales; ModelOpt NVFP4 A4/A8 preparation requires both scale tensors.
    ``immutable_input_scales`` promises that input scale values remain unchanged
    throughout prepared bindings and graph replay; reprepare after mutation.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_block_scales: torch.Tensor
    w2_block_scales: torch.Tensor
    w13_global_scales: torch.Tensor
    w2_global_scales: torch.Tensor
    input_scale: torch.Tensor | None = None
    intermediate_scale: torch.Tensor | None = None
    immutable_input_scales: bool = False

    def __post_init__(self) -> None:
        if type(self.immutable_input_scales) is not bool:
            raise TypeError("immutable_input_scales must be boolean")
        for name in (
            "w13",
            "w2",
            "w13_block_scales",
            "w2_block_scales",
            "w13_global_scales",
            "w2_global_scales",
        ):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise TypeError(f"PackedWeights.{name} must be a torch.Tensor")
        for name in ("input_scale", "intermediate_scale"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"PackedWeights.{name} must be a tensor or None")


@dataclass(frozen=True, kw_only=True)
class PreparedExperts:
    """Prepared expert tensors owned by a canonical weight plan."""

    plan: "WeightPlan"
    _impl: "B12XFP4ExpertWeights"

    def __post_init__(self) -> None:
        from ._impl import B12XFP4ExpertWeights
        from .planning import WeightPlan

        if not isinstance(self.plan, WeightPlan):
            raise TypeError("plan must be a canonical WeightPlan")
        if not isinstance(self._impl, B12XFP4ExpertWeights):
            raise TypeError("_impl must be prepared B12X expert weights")
        if self._impl.plan != self.plan._impl:
            raise ValueError("prepared experts do not match the canonical plan")

    @property
    def num_experts(self) -> int:
        return self._impl.num_experts

    @property
    def hidden_size(self) -> int:
        return self._impl.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self._impl.intermediate_size

    @property
    def device(self) -> torch.device:
        return self._impl.w1_fp4.device


__all__ = [
    "PackedWeights",
    "GGUFExpertWeights",
    "IQ2XSWeights",
    "IQ3XXSWeights",
    "IQ2XXSWeights",
    "IQ2SWeights",
    "IQ3SWeights",
    "IQMXFP4Weights",
    "IQMixedWeights",
    "PreparedExperts",
    "PreparedWeightFormat",
    "ScaleEncoding",
    "ScaleFactors",
    "TrellisWeights",
    "WeightEncoding",
    "WeightPacking",
]

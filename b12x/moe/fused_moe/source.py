"""Checkpoint-side fused-MoE weight representations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .config import TrellisConfig


class PackedSourceFormat(str, Enum):
    """Packed checkpoint encoding, including its scale-grid contract."""

    MXFP4_E8M0_K32 = "fp4_e8m0_k32"
    MODELOPT_NVFP4 = "modelopt_nvfp4"
    COMPRESSED_TENSORS_FP4 = "compressed_tensors"
    MXFP6_E8M0_K32 = "mxfp6_e2m3"
    IQ2_XS = "iq2_xs"
    IQ3_XXS = "iq3_xxs"
    IQ2_XXS = "iq2_xxs"
    IQ2_S = "iq2_s"
    IQ3_S = "iq3_s"


class W13Layout(str, Enum):
    """Logical order of the two gated FC1 projections."""

    W13 = "w13"
    W31 = "w31"


@dataclass(frozen=True, kw_only=True)
class PackedSource:
    """Source encoding and packing of one ordinary MoE checkpoint."""

    format: PackedSourceFormat
    w13_layout: W13Layout = W13Layout.W13
    fc2_format: PackedSourceFormat | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", PackedSourceFormat(self.format))
        object.__setattr__(self, "w13_layout", W13Layout(self.w13_layout))
        if self.fc2_format is not None:
            object.__setattr__(self, "fc2_format", PackedSourceFormat(self.fc2_format))
            supported = (
                self.format in {PackedSourceFormat.IQ2_XS, PackedSourceFormat.IQ3_XXS, PackedSourceFormat.IQ2_XXS, PackedSourceFormat.IQ2_S}
                and self.fc2_format is PackedSourceFormat.MXFP4_E8M0_K32
            ) or (
                self.format is PackedSourceFormat.IQ2_XXS
                and self.fc2_format is PackedSourceFormat.IQ3_XXS
            )
            supported = supported or (
                self.format in {PackedSourceFormat.IQ2_S, PackedSourceFormat.IQ3_XXS}
                and self.fc2_format is PackedSourceFormat.IQ3_S
            )
            if not supported:
                raise ValueError("unsupported mixed source projection formats")


WeightSource: TypeAlias = PackedSource | TrellisConfig


__all__ = [
    "PackedSource",
    "PackedSourceFormat",
    "W13Layout",
    "WeightSource",
]

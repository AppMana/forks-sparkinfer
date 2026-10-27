"""Bounded reads of unchanged GGUF expert blocks for native preparation.

Container support does not imply execution support for a codec. Consumers must
select a matching native preparation/kernel; this loader never requantizes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Sequence

import torch


@dataclass(frozen=True)
class ExpertTensor:
    path: Path
    name: str
    quant_type: str
    offset: int
    experts: int
    rows: int
    columns: int
    block_weights: int
    block_bytes: int

    def __post_init__(self):
        if (
            min(
                self.experts,
                self.rows,
                self.columns,
                self.block_weights,
                self.block_bytes,
            )
            <= 0
        ):
            raise ValueError("Expert dimensions and block geometry must be positive")
        if self.offset < 0 or self.columns % self.block_weights:
            raise ValueError("Invalid offset or unaligned GGUF row")

    @property
    def expert_bytes(self) -> int:
        return self.rows * (self.columns // self.block_weights) * self.block_bytes

    def read_expert(self, expert: int) -> torch.Tensor:
        """Return owned CPU uint8 [N,K/block,bytes] without decoding or mmap."""
        if not 0 <= expert < self.experts:
            raise ValueError("Expert index outside the tensor")
        offset = self.offset + expert * self.expert_bytes
        # Python integers and pread offsets remain 64-bit-safe past 2 GiB.
        with self.path.open("rb", buffering=0) as stream:
            if offset + self.expert_bytes > os.fstat(stream.fileno()).st_size:
                raise ValueError(f"Truncated GGUF payload for {self.name}")
            storage = bytearray(self.expert_bytes)
            target = memoryview(storage)
            stream.seek(offset)
            done = 0
            while done < len(target):
                count = stream.readinto(target[done : done + (8 << 20)])
                if not count:
                    raise ValueError(f"Short read for {self.name}")
                done += count
            # The owned buffer is the staging copy. Retaining read-once file
            # pages as well competes with CUDA weights on unified-memory GPUs.
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    stream.fileno(), offset, self.expert_bytes, os.POSIX_FADV_DONTNEED
                )
        return torch.frombuffer(storage, dtype=torch.uint8).reshape(
            self.rows, self.columns // self.block_weights, self.block_bytes
        )

    def iter_experts(
        self, expert_ids: Sequence[int]
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Preserve explicit EP ownership/order, with one expert per CPU buffer."""
        ids = tuple(expert_ids)
        if len(set(ids)) != len(ids) or any(not 0 <= e < self.experts for e in ids):
            raise ValueError("Expert IDs must be unique and in range")
        for expert in ids:
            yield expert, self.read_expert(expert)

    def tp_slice(
        self, blocks: torch.Tensor, *, rank: int, size: int, axis: str
    ) -> torch.Tensor:
        """Slice whole rows/blocks only; reject lossy cuts through a codebook block."""
        if size <= 0 or not 0 <= rank < size or axis not in {"rows", "columns"}:
            raise ValueError("Invalid TP partition")
        if blocks.dtype != torch.uint8 or tuple(blocks.shape) != (
            self.rows,
            self.columns // self.block_weights,
            self.block_bytes,
        ):
            raise ValueError("Blocks disagree with source tensor geometry")
        width = self.rows if axis == "rows" else self.columns
        if width % size or (axis == "columns" and (width // size) % self.block_weights):
            raise ValueError(
                "TP split cuts a quantization block; use whole-expert ownership"
            )
        dim = 0 if axis == "rows" else 1
        count = blocks.shape[dim] // size
        return blocks.narrow(dim, rank * count, count)


def index_experts(paths: Sequence[Path]) -> dict[str, ExpertTensor]:
    """Index local shards; GGUF is an optional container-reading dependency.

    GGUFReader maps files while parsing their headers, but this function never
    reads or decodes tensor data, including the separate Engram table.
    """
    from gguf import GGML_QUANT_SIZES, GGUFReader

    result = {}
    for path in paths:
        path = Path(path)
        reader = GGUFReader(path, mode="r")
        if reader.byte_order != "I":
            raise ValueError("Only native little-endian GGUF payloads are supported")
        for tensor in reader.tensors:
            if not tensor.name.endswith("_exps.weight"):
                continue
            if tensor.name in result or len(tensor.shape) != 3:
                raise ValueError(f"Duplicate or invalid expert tensor: {tensor.name}")
            k, n, e = map(int, tensor.shape)
            block, byte_count = GGML_QUANT_SIZES[tensor.tensor_type]
            entry = ExpertTensor(
                path,
                tensor.name,
                tensor.tensor_type.name,
                int(tensor.data_offset),
                e,
                n,
                k,
                block,
                byte_count,
            )
            if entry.expert_bytes * e != tensor.n_bytes:
                raise ValueError(f"Invalid tensor payload size: {tensor.name}")
            result[tensor.name] = entry
        del reader
    return result


def prepare_iq_expert_partition(
    gate: ExpertTensor,
    up: ExpertTensor,
    down: ExpertTensor,
    expert_ids: Sequence[int],
    *,
    device: torch.device | str,
):
    """Stream an EP partition into native compact storage, one expert at a time.

    Returns (prepared_weights, global_to_local_map). No full raw partition or
    dequantized matrix is allocated. Temporary storage is bounded by one expert
    triple plus its prepared representation, independent of partition size.
    The caller owns activation clamping, routing and the final EP reduction.
    """
    from dataclasses import replace

    from b12x.moe._shared.kernels.w4a16.iq2_xs import _prepare_iq_moe_weights

    ids = tuple(expert_ids)
    if not ids or len(set(ids)) != len(ids) or any(
        not 0 <= e < gate.experts for e in ids
    ):
        raise ValueError("Expert IDs must be nonempty, unique and in range")
    h, i = gate.columns, gate.rows
    if (up.experts, up.rows, up.columns, up.quant_type) != (
        gate.experts, i, h, gate.quant_type
    ) or (down.experts, down.rows, down.columns) != (gate.experts, h, i):
        raise ValueError("Gate/up/down expert geometry or format disagrees")
    sizes = {"IQ2_XS": 74, "IQ2_XXS": 66, "IQ2_S": 82,
             "IQ3_XXS": 98, "IQ3_S": 110}
    if gate.quant_type not in sizes or down.quant_type not in {*sizes, "MXFP4"}:
        raise ValueError("Unsupported native IQ expert format")
    for source in (gate, up, down):
        expected = (32, 17) if source.quant_type == "MXFP4" else (256, sizes[source.quant_type])
        if (source.block_weights, source.block_bytes) != expected:
            raise ValueError("Source block geometry disagrees with its format")
    pair = (gate.quant_type, down.quant_type)
    if down.quant_type not in {gate.quant_type, "MXFP4"} and pair not in {
        ("IQ2_XXS", "IQ3_XXS"), ("IQ2_S", "IQ3_S"), ("IQ3_XXS", "IQ3_S")
    }:
        raise ValueError("Unsupported mixed expert pair")
    if h % 256 or i % 256:
        raise ValueError("Native IQ experts require H/I divisible by 256")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native IQ preparation requires CUDA")
    count = len(ids)
    prepared = None
    for local, global_id in enumerate(ids):
        # Concatenate on the CPU: only the combined gate/up and down raw blocks
        # cross to the device. GGUF's separate gate and up tensors are w31.
        raw13 = torch.cat((gate.read_expert(global_id), up.read_expert(global_id)))
        raw2 = down.read_expert(global_id)
        one = _prepare_iq_moe_weights(
            raw13.unsqueeze(0).to(device), raw2.unsqueeze(0).to(device),
            hidden_size=h, intermediate_size=i, num_experts=1,
            activation="silu", w13_layout="w31", recipe=gate.quant_type.lower(),
            fc2_format=None if gate.quant_type == down.quant_type else down.quant_type.lower(),
        )
        del raw13, raw2
        if prepared is None:
            storage = {}
            for name in ("w13", "w13_scale", "w2", "w2_scale"):
                src = getattr(one, name)
                shape = (src.shape[0] * count, *src.shape[1:])
                storage[name] = torch.empty(shape, dtype=src.dtype, device=device)
            unit = torch.ones(count, dtype=torch.float32, device=device)
            prepared = replace(one, **storage, num_experts=count,
                               w13_global_scale=unit, w2_global_scale=unit)
        for name in ("w13", "w13_scale", "w2", "w2_scale"):
            src = getattr(one, name).reshape(-1)
            dst = getattr(prepared, name).reshape(-1)
            # IQ metadata is all experts' bases followed by all experts' other
            # metadata. Concatenating one-expert buffers would corrupt E > 1.
            base = (2 * i * (h // 256) * 2 if name == "w13_scale" else
                    h * (i // 256) * 2 if name == "w2_scale" and down.quant_type != "MXFP4" else 0)
            if base:
                dst[local * base:(local + 1) * base].copy_(src[:base])
                tail = src.numel() - base
                start = count * base + local * tail
                dst[start:start + tail].copy_(src[base:])
            else:
                start = local * src.numel()
                dst[start:start + src.numel()].copy_(src)
        del one
    expert_map = torch.full((gate.experts,), -1, dtype=torch.int32, device=device)
    expert_map[torch.tensor(ids, device=device)] = torch.arange(count, dtype=torch.int32, device=device)
    return prepared, expert_map

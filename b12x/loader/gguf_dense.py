"""Bounded reads and native compact preparation for GGUF dense matrices."""
from dataclasses import dataclass
import os
from pathlib import Path

import torch

_FORMATS = {"Q6_K": (256, 210), "Q8_0": (32, 34)}


@dataclass(frozen=True)
class DenseTensor:
    path: Path
    name: str
    quant_type: str
    offset: int
    rows: int
    columns: int

    def __post_init__(self):
        if self.quant_type not in _FORMATS:
            raise ValueError("Native dense source requires Q6_K or Q8_0")
        if self.offset < 0 or min(self.rows, self.columns) <= 0:
            raise ValueError("Invalid dense source geometry or offset")
        if self.columns % _FORMATS[self.quant_type][0]:
            raise ValueError("Dense source row cuts a quantization block")

    def read_rows(self, start, count, *, column_start=0, columns=None):
        """Owned CPU blocks, with zero padding beyond the source dimensions.

        Bounds refer to a conceptual zero-padded matrix. Every column boundary
        must preserve complete original quantization blocks. File positions use
        Python integers; no 32-bit products or full-matrix mmap/read is involved.
        """
        columns = self.columns if columns is None else columns
        block, size = _FORMATS[self.quant_type]
        if min(start, column_start) < 0 or min(count, columns) <= 0:
            raise ValueError("Invalid dense slice")
        if column_start % block or columns % block:
            raise ValueError("Dense slice cuts a quantization block")
        storage = bytearray(count * (columns // block) * size)
        target = memoryview(storage)
        source_stride = self.columns // block * size
        target_stride = columns // block * size
        live_rows = max(0, min(count, self.rows - start))
        live_columns = max(0, min(columns, self.columns - column_start))
        with Path(self.path).open("rb", buffering=0) as stream:
            if self.offset + self.rows * source_stride > os.fstat(stream.fileno()).st_size:
                raise ValueError(f"Truncated dense payload: {self.name}")
            for row in range(live_rows):
                if live_columns == 0:
                    break
                stream.seek(self.offset + (start + row) * source_stride + column_start // block * size)
                dest = target[row * target_stride:row * target_stride + live_columns // block * size]
                done = 0
                while done < len(dest):
                    count_read = stream.readinto(dest[done:])
                    if not count_read:
                        raise ValueError(f"Short dense payload read: {self.name}")
                    done += count_read
            # Keep the owned staging tile, not a second copy in the page cache.
            if live_rows and live_columns and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    stream.fileno(), self.offset + start * source_stride,
                    live_rows * source_stride, os.POSIX_FADV_DONTNEED,
                )
        return torch.frombuffer(storage, dtype=torch.uint8).reshape(count, columns // block, size)


def index_dense(paths):
    """Index supported 2D matrices without reading their payloads."""
    from gguf import GGUFReader

    result = {}
    for path in paths:
        reader = GGUFReader(path, mode="r")
        if reader.byte_order != "I":
            raise ValueError("Only native little-endian GGUF payloads are supported")
        for tensor in reader.tensors:
            if tensor.tensor_type.name not in _FORMATS or len(tensor.shape) != 2:
                continue
            if tensor.name in result:
                raise ValueError(f"Duplicate dense tensor: {tensor.name}")
            k, n = map(int, tensor.shape)
            source = DenseTensor(Path(path), tensor.name, tensor.tensor_type.name, int(tensor.data_offset), n, k)
            block, size = _FORMATS[source.quant_type]
            if n * (k // block) * size != tensor.n_bytes:
                raise ValueError(f"Invalid dense tensor size: {tensor.name}")
            result[tensor.name] = source
        del reader
    return result


def prepare_dense_partition(source, *, device, row_start=0, rows=None,
                            column_start=0, columns=None):
    """Stream N128 tiles into final compact storage, including padded TP slices.

    Extra GPU/CPU memory is bounded by one N128 source tile and its packed
    representation. No full raw or BF16 matrix is allocated during preparation.
    """
    from b12x.gemm.blockscaled import pack_weight
    from b12x.gemm.blockscaled._q6_k import Q6KLinearWeight
    from b12x.gemm.blockscaled._q8_0 import Q8_0LinearWeight

    rows = source.rows if rows is None else rows
    columns = source.columns if columns is None else columns
    if rows <= 0 or columns <= 0 or rows % 8 or columns % 256:
        raise ValueError("Native dense partition requires N divisible by 8 and K by 256")
    block, _ = _FORMATS[source.quant_type]
    if min(row_start, column_start) < 0 or column_start % block:
        raise ValueError("Dense partition cuts a quantization block or has negative offset")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native dense preparation requires CUDA")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dense preparation must precede graph capture")
        q6 = source.quant_type == "Q6_K"
        values = torch.empty((rows, columns // (2 if q6 else 1)), dtype=torch.uint8, device=device)
        metadata = torch.empty(((rows + 127) // 128, columns // 256, 10496 if q6 else 2048), dtype=torch.uint8, device=device)
        for start in range(0, rows, 128):
            count = min(128, rows - start)
            raw = source.read_rows(row_start + start, count, column_start=column_start, columns=columns)
            packed = pack_weight(raw.to(device), recipe=source.quant_type.lower())
            values[start:start + count].copy_(packed.values)
            metadata[start // 128:start // 128 + 1].copy_(packed.metadata)
            del raw, packed
    return (Q6KLinearWeight if q6 else Q8_0LinearWeight)(values, metadata, columns, rows)

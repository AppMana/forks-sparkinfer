"""Container reads must preserve codebook bytes, EP ordering and TP boundaries."""

import numpy as np
import pytest
import torch

from b12x.loader.gguf_experts import ExpertTensor, index_experts


def test_sparse_file_large_offset_and_expert_order(tmp_path):
    path = tmp_path / "experts.bin"
    offset = (1 << 32) + 32
    data = bytes(range(148)) * 3
    with path.open("wb") as stream:
        stream.seek(offset)
        stream.write(data)
    tensor = ExpertTensor(
        path, "blk.0.ffn_gate_exps.weight", "IQ2_XS", offset, 3, 2, 256, 256, 74
    )
    values = list(tensor.iter_experts([2, 0]))
    assert [e for e, _ in values] == [2, 0]
    assert values[0][1].numpy().tobytes() == data[296:444]
    values[0][1].zero_()
    assert values[1][1].numpy().tobytes() == data[:148]
    with pytest.raises(ValueError, match="unique"):
        list(tensor.iter_experts([0, 0]))
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Truncated"):
        tensor.read_expert(0)


def test_dsv41_tp2_rejects_half_block_but_preserves_row_partition(tmp_path):
    tensor = ExpertTensor(
        tmp_path / "unused", "down", "IQ3_XXS", 0, 384, 16, 2304, 256, 98
    )
    blocks = torch.arange(16 * 9 * 98, dtype=torch.int64).byte().reshape(16, 9, 98)
    with pytest.raises(ValueError, match="cuts a quantization block"):
        tensor.tp_slice(blocks, rank=1, size=2, axis="columns")
    parts = [tensor.tp_slice(blocks, rank=r, size=2, axis="rows") for r in (0, 1)]
    assert torch.equal(torch.cat(parts), blocks)


def test_real_gguf_index_and_byte_exact_payload(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "test.gguf"
    packed = np.arange(2 * 16 * 74, dtype=np.int64).astype(np.uint8).reshape(2, 16, 74)
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_tensor(
        "blk.0.ffn_gate_exps.weight", packed, raw_dtype=gguf.GGMLQuantizationType.IQ2_XS
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    indexed = index_experts([path])
    entry = indexed["blk.0.ffn_gate_exps.weight"]
    assert (entry.experts, entry.rows, entry.columns) == (2, 16, 256)
    assert entry.read_expert(1).numpy().tobytes() == packed[1].tobytes()
    with pytest.raises(ValueError, match="Duplicate"):
        index_experts([path, path])

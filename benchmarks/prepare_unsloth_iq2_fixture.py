"""Wrap exact sampled GGUF blocks for the existing native A16 benchmark.

This is a projection qualification fixture, not a converted serving checkpoint.
No dequantization or requantization is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", choices=("iq2_xs", "iq3_xxs"), default="iq2_xs")
    args = parser.parse_args()
    block_bytes = 74 if args.recipe == "iq2_xs" else 98
    manifest = json.loads(args.manifest.read_text())
    tensors, recipes, provenance = {}, {}, []
    for sample in manifest["samples"]:
        if sample["type"] != args.recipe.upper():
            continue
        if sample["experts"] != 1:
            raise ValueError("Fixture expects one sampled expert per projection")
        payload = (args.manifest.parent / Path(sample["path"]).name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sample["sha256"]:
            raise ValueError("Source sample SHA256 mismatch")
        k, n, _ = sample["dims"]
        name = sample["name"]
        tensors[name] = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
            n, k // 256, block_bytes
        )
        recipes[name.removesuffix(".weight")] = dict(
            quant_algo=args.recipe.upper(), packing="ggml", group_size=256, block_payload_bytes=block_bytes
        )
        provenance.append(sample)
    if not tensors:
        raise ValueError("No matching IQ sample found")
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output / "samples.safetensors")
    documents = {
        "config.json": {
            "fixture_only": True,
            "model_type": "dsv41_projection_qualification",
        },
        "model.safetensors.index.json": {
            "weight_map": {k: "samples.safetensors" for k in tensors}
        },
        "hf_quant_config.json": {"quantization": {"quantized_layers": recipes}},
        "provenance.json": {
            "repo": manifest["repo"],
            "revision": manifest["revision"],
            "samples": provenance,
        },
    }
    for name, data in documents.items():
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

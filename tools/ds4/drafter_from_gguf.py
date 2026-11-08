"""Use an existing DeepSeek-V4.1-Flash DSpark GGUF as a Maya drafter, without copying its weight file.

    python tools/ds4/drafter_from_gguf.py --gguf <dspark.gguf> --out <dspark.json>
    python tools/ds4/pack_gguf.py --model <model-ggufs> --drafter <dspark.json>

The JSON has the same ``tensors`` and ``config`` entries as drafter_pack.py's output. Its sibling .bin is a
relative symlink to the GGUF; tensor offsets point into that file, past its GGUF header. The GGUF must hold the
three-block, single-pass DeepSeek-V4.1-Flash drafter in F32, BF16 and MXFP4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from pack_gguf import Gguf, LAYER, TYPES


SPECIAL = {
    "enc.output_norm.weight": "mtp.0.main_norm.weight",
    "fc.weight": "mtp.0.main_proj.weight",
    "output_norm.weight": "mtp.2.norm.weight",
    "conf_proj.weight": "mtp.2.confidence_head.proj.weight",
    "markov_w1.weight": "mtp.2.markov_head.embed.weight",
    "markov_w2.weight": "mtp.2.markov_head.head.weight",
}
BLOCK_TENSORS = {
    "attn_kv.weight", "attn_kv_a_norm.weight", "attn_norm.weight", "attn_output_a.weight",
    "attn_output_b.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
    "attn_sinks.weight", "exp_probs_b.bias", "ffn_down_exps.weight", "ffn_down_shexp.weight",
    "ffn_gate_exps.weight", "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_norm.weight",
    "ffn_up_exps.weight", "ffn_up_shexp.weight", "hc_attn_base.weight", "hc_attn_fn.weight",
    "hc_attn_scale.weight", "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight",
}
PRECISIONS = {0, 30, 39}  # F32, BF16, MXFP4


def manifest(source: Path) -> dict:
    """Map GGUF tensor names and byte ranges to Maya's drafter manifest; read no weight payload."""
    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        raise SystemExit(f"{source}: not a file")
    g = Gguf(source)
    kv = g.kv
    if kv.get("general.architecture") != "dflash" or kv.get("dflash.block_count") != 3:
        raise SystemExit("Expected the three-block DeepSeek-V4.1-Flash DSpark GGUF")
    if kv.get("dflash.target_layers") != [37, 38, 39] or not kv.get("dflash.hyper_connection.single_pass"):
        raise SystemExit("Unexpected DSpark target stages or hyper-connection layout")
    try:
        block_size = int(kv["dflash.block_size"])
        noise_token = int(kv["tokenizer.ggml.mask_token_id"])
        expert_count = int(kv["dflash.expert_count"])
        experts_per_token = int(kv["dflash.expert_used_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Missing or invalid DSpark geometry: {exc}") from exc
    if block_size < 1 or noise_token < 0 or expert_count < 1 or not 0 < experts_per_token <= expert_count:
        raise SystemExit("Invalid DSpark block, noise token or expert geometry")

    entries, covered, seen_specials, names, ranges = [], [], set(), set(), []
    file_bytes = source.stat().st_size
    if g.data_start > file_bytes:
        raise SystemExit("GGUF header extends beyond the file")
    for name, typ, dims, relative in g.tensors:
        if name in names:
            raise SystemExit(f"Duplicate GGUF tensor name: {name}")
        names.add(name)
        if typ not in PRECISIONS:
            raise SystemExit(f"Unexpected precision for {name}: GGML type {typ}")
        dtype, block, block_bytes = TYPES[typ]
        if not dims or any(d <= 0 for d in dims) or dims[0] % block:
            raise SystemExit(f"Invalid block geometry for {name}")
        byte_count = math.prod(dims) // block * block_bytes
        offset = g.data_start + relative
        if offset < g.data_start or offset + byte_count > file_bytes:
            raise SystemExit(f"Invalid source byte range for {name}")
        ranges.append((offset, offset + byte_count, name))
        shape = list(reversed(dims))
        layer = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if layer and int(layer[1]) < 3 and layer[2] in LAYER:
            target = f"mtp.{layer[1]}.{LAYER[layer[2]]}"
            if target.endswith(".attn.wo_a.slice"):
                target = target[:-len(".slice")] + ".weight"
            if ".ffn.experts." in target:
                if dtype != "MXFP4" or len(shape) != 3 or shape[0] != expert_count:
                    raise SystemExit(f"Unexpected expert geometry for {name}")
                per_expert = byte_count // expert_count
                if per_expert * expert_count != byte_count:
                    raise SystemExit(f"Uneven expert byte ranges for {name}")
                prefix, role = target.rsplit(".", 1)
                for expert in range(expert_count):
                    entries.append([f"{prefix}.{expert}.{role}.weight", dtype,
                                    offset + expert * per_expert, per_expert, shape[1:]])
            else:
                entries.append([target, dtype, offset, byte_count, shape])
        elif name in SPECIAL:
            entries.append([SPECIAL[name], dtype, offset, byte_count, shape])
            seen_specials.add(name)
        else:
            raise SystemExit(f"Unmapped drafter tensor: {name}")
        covered.append([name, dtype, offset, byte_count, shape])
    if seen_specials != set(SPECIAL):
        raise SystemExit(f"Missing special tensors: {', '.join(sorted(set(SPECIAL) - seen_specials))}")
    required = {f"blk.{block}.{tensor}" for block in range(3) for tensor in BLOCK_TENSORS}
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"Missing DSpark block tensors ({len(missing)}): {', '.join(missing[:8])}")
    destinations = [entry[0] for entry in entries]
    if len(set(destinations)) != len(destinations):
        raise SystemExit("Duplicated destination tensor name")
    ordered_ranges = sorted(ranges)
    for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
        if current[0] < previous[1]:
            raise SystemExit(f"Overlapping GGUF tensor byte ranges: {previous[2]}, {current[2]}")

    markov = next(entry for entry in entries if entry[0] == "mtp.2.markov_head.embed.weight")
    config = {
        "dspark_block_size": block_size,
        "dspark_noise_token_id": noise_token,
        "dspark_markov_rank": markov[4][-1],
        "dspark_n_routed_experts": expert_count,
        "dspark_num_experts_per_tok": experts_per_token,
        "dspark_target_layer_ids": kv["dflash.target_layers"],
    }
    with source.open("rb") as f:
        header_sha = hashlib.sha256(f.read(g.data_start)).hexdigest()
    return {
        "from": source.name,
        "format": "existing-gguf-byte-ranges",
        "config": config,
        "source_size": file_bytes,
        "source_header_sha256": header_sha,
        "source_tensors": covered,
        "tensors": entries,
    }


def convert(source: Path, out: Path) -> dict:
    """Write only metadata and a link; leave the GGUF untouched."""
    source = source.expanduser().resolve(strict=True)
    payload = manifest(source)
    out = out.expanduser()
    if out.suffix != ".json":
        raise SystemExit("--out must name a .json file")
    out.parent.mkdir(parents=True, exist_ok=True)
    output = out.parent.resolve() / out.name  # use the physical directory when a mount has another alias
    binary = output.with_suffix(".bin")
    temporary = output.with_suffix(output.suffix + ".part")
    if source in (output, binary, temporary):
        raise SystemExit("Refusing to overwrite the source GGUF")
    if binary.exists() or binary.is_symlink():
        if not binary.is_symlink() or binary.resolve() != source:
            raise SystemExit(f"Refusing to replace {binary}")
        new_link = False
    else:
        new_link = True
    created_link = False
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if new_link:
            binary.symlink_to(os.path.relpath(source, binary.parent))
            created_link = True
        temporary.replace(output)
    except OSError:
        temporary.unlink(missing_ok=True)
        if created_link:
            binary.unlink(missing_ok=True)
        raise
    return {"json": str(output), "binary_symlink": str(binary), "source_tensors": len(payload["source_tensors"]),
            "mapped_entries": len(payload["tensors"]), "config": payload["config"], "payload_bytes_written": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True, help="the existing DSpark GGUF")
    ap.add_argument("--out", type=Path, required=True, help="output JSON; a sibling .bin links to the GGUF")
    args = ap.parse_args()
    print(json.dumps(convert(args.gguf, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

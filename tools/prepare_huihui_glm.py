#!/usr/bin/env python3
"""Prepare manually downloaded Huihui GLM-5.3-Flash GGUF shards for Maya.

A complete source set is SHA-256 verified before creating links/copies/moves.
GGUF bytes are never rewritten; metadata is extracted from the headers only.

  python tools/prepare_huihui_glm.py --model glm5.3-flash-ablit-Q3_K_XL \
      --source /downloads/UD-Q3_K_XL --models-dir /models --mode hardlink
  python tools/prepare_huihui_glm.py --model UD-IQ4_XS \
      --source /models/glm5.3-flash-ablit-IQ4_XS --metadata-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import maya  # noqa: E402 - reuse exactly the same catalog as the setup
from gguf_reader import GGUFFile, BLOCK_GEOMETRY  # noqa: E402

METADATA_FILE = "huihui-gguf-metadata.json"


def choose_model(name: str) -> tuple[str, dict]:
    """Accept the setup ID, the HF quant folder, or its short quant name."""
    for ident, folder, _, _, _ in maya.HUIHUI_QUANTS:
        if name.lower() in (ident.lower(), folder.lower(), folder.removeprefix("UD-").lower()):
            return ident, maya.MODELS[ident]
    raise ValueError(f"unknown Huihui quant {name!r}: use " + ", ".join(x[0] for x in maya.HUIHUI_QUANTS))


def source_files(source: Path, spec: dict, source_prefix: str | None = None) -> list[Path]:
    """Resolve published filename or explicitly renamed prefix; don't guess a shard order."""
    names = maya.shard_names(spec)
    folder = source / spec["folder"] if (source / spec["folder"]).is_dir() else source
    if source_prefix:
        if not source_prefix or Path(source_prefix).name != source_prefix or source_prefix.endswith(".gguf"):
            raise ValueError("--source-prefix must be a filename prefix without a path or .gguf")
        names = [f"{source_prefix}-{i:05d}-of-{spec['shards']:05d}.gguf"
                 for i in range(1, spec["shards"] + 1)]
    return [folder / name for name in names]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(32 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def inspect_architecture(metadata: dict) -> dict:
    """Compare the real GGUF geometry with Glm5Geometry::from_gguf's hard constraints.

    Missing optional metadata is not evidence of compatibility: the C++ runner has
    defaults, so absence is shown explicitly. Tensor names/types are checked later
    by iq_pack.py, whose fail-closed logic remains the authority.
    """
    arch = metadata.get("general.architecture", "")
    errors, unknown, facts = [], [], {}
    if arch not in maya.GLM_ARCHS:
        errors.append(f"general.architecture {arch!r} is not glm5-next/glm5next")
    prefix = ("glm5next." if arch == "glm5next" else "glm5-next.")

    def get(key, default=None):
        full = prefix + key
        if full not in metadata:
            unknown.append(full)
        value = metadata.get(full, default)
        facts[key] = value
        return value

    block = get("block_count")
    nextn = get("nextn_predict_layers", 0)
    explicit = get("trunk_block_count", 0)
    try:
        layers = int(explicit or (int(block) - int(nextn))) if block is not None else None
        facts["trunk_layers"] = layers
        if layers is not None and layers < 1:
            errors.append(f"block_count - nextn_predict_layers must leave a trunk; got {layers}")
    except (ValueError, TypeError):
        layers = None
        errors.append("block_count / nextn_predict_layers is not a valid integer")

    hc = get("hyper_connection.count", 4)
    if hc != 4:
        errors.append(f"hyper_connection.count must be 4 (Maya mHC kernels), got {hc!r}")
    qk = get("attention.key_length_mla", 256)
    v = get("attention.value_length_mla", 256)
    if qk != v:
        errors.append(f"attention.key_length_mla ({qk}) != value_length_mla ({v})")
    kv = get("attention.kv_lora_rank", 512)
    key_length = get("attention.key_length")
    if key_length is not None:
        if isinstance(key_length, list):
            key_length = key_length[0] if key_length else None
        if key_length != kv:
            errors.append(f"attention.key_length ({key_length}) != kv_lora_rank ({kv})")
    kpool = get("attention.indexer.kpool", 4)
    top_k = get("attention.indexer.top_k", 2048)
    try:
        if int(kpool) <= 1 or int(top_k) % int(kpool):
            errors.append(f"indexer requires kpool>1 and top_k divisible by kpool, got {kpool}/{top_k}")
    except (TypeError, ValueError, ZeroDivisionError):
        errors.append("indexer kpool/top_k are not valid integers")
    tail = get("attention.indexer.kpool_select_tail", 1)
    if not tail:
        errors.append("attention.indexer.kpool_select_tail=false is unsupported by Maya")
    kinds = get("attention.indexer.types")
    heads = get("attention.head_count_kv")
    if isinstance(kinds, list) and isinstance(heads, list) and layers:
        if len(kinds) >= layers and len(heads) >= layers:
            unsupported = [i for i in range(layers) if heads[i] != 0 and kinds[i] == 0]
            if unsupported:
                errors.append(f"shared (non-full) DSA indexer at layer(s): {unsupported[:8]}")
    return {
        "status": "incompatible" if errors else ("inconclusive" if block is None else "passes_known_geometry_constraints"),
        "checked_against": "src/core/glm_model.cu:Glm5Geometry::from_gguf",
        "metadata_prefix": prefix, "observed": facts,
        "errors": errors, "defaulted_or_absent_keys": sorted(set(unknown)),
        "limits": "A metadata check cannot establish tensor-name/shape or inference correctness without all GGUF shards and a GPU run.",
    }


def validate_files(paths: list[Path], spec: dict, *, verify_hash: bool = True) -> list[dict]:
    expected = maya.shard_names(spec)
    if len(paths) != len(expected):
        raise ValueError(f"expected {len(expected)} GGUF shards, got {len(paths)}")
    result = []
    first_meta = None
    seen_tensors = set()
    for i, (src, canonical) in enumerate(zip(paths, expected), 1):
        if not src.is_file():
            raise FileNotFoundError(f"shard {i}/{len(expected)} missing: {src}")
        # Read every header without touching tensor data. Catches a truncated header or
        # mismatched architecture before anything is placed in the destination.
        try:
            gguf = GGUFFile(src)
        except (OSError, ValueError, KeyError, IndexError, EOFError) as e:
            raise ValueError(f"{src}: invalid GGUF header: {e}") from e
        if gguf.metadata.get("general.architecture") not in maya.GLM_ARCHS:
            raise ValueError(f"{src}: not a GLM-5.3 GGUF: {gguf.metadata.get('general.architecture')!r}")
        if first_meta is None:
            first_meta = gguf.metadata
            arch_result = inspect_architecture(first_meta)
            if arch_result["errors"]:
                raise ValueError(f"{src}: unsupported Maya GLM geometry: " + "; ".join(arch_result["errors"]))
        for t in gguf.tensors:
            if t.name in seen_tensors:
                raise ValueError(f"{src}: duplicate GGUF tensor across shards: {t.name}")
            seen_tensors.add(t.name)
            if t.type_name not in BLOCK_GEOMETRY or t.expected_bytes() is None:
                raise ValueError(f"{src}: unsupported or malformed GGML type {t.type_name} of {t.name}")
        need = (gguf.data_start + max(t.offset + (t.expected_bytes() or 0) for t in gguf.tensors)
                if gguf.tensors else gguf.header_end)
        size = src.stat().st_size
        if size < need:
            raise ValueError(f"{src}: truncated: {size} bytes, at least {need} required")
        want = spec["sha256"][canonical]
        actual = digest(src) if verify_hash else None
        if verify_hash and actual != want:
            raise ValueError(f"{src}: SHA-256 mismatch: got {actual}; expected {want}")
        result.append({"source": str(src.resolve()), "filename": canonical, "size": size,
                       "sha256": actual or want, "sha256_verified": verify_hash,
                       "tensor_count": len(gguf.tensors), "gguf_version": gguf.version,
                       "architecture": gguf.metadata["general.architecture"]})
    return result


def place(src: Path, dst: Path, mode: str, sha: str) -> str:
    if src.resolve() == dst.resolve():
        return "already in place"
    if dst.exists() or dst.is_symlink():
        if dst.is_file() and digest(dst) == sha:
            return "already present (SHA-256 matches)"
        raise FileExistsError(f"{dst} already exists and does not match: refusing to overwrite")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "move":
        os.rename(src, dst)  # same filesystem only; no silent 50-GB cross-drive copy
    elif mode == "copy":
        tmp = dst.with_name(dst.name + ".maya-partial")
        if tmp.exists() or tmp.is_symlink():
            raise FileExistsError(f"{tmp} exists: remove it only after checking it is a partial copy")
        try:
            shutil.copyfile(src, tmp)
            if digest(tmp) != sha:
                raise ValueError(f"{dst}: copied data has a different SHA-256")
            os.rename(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink()
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return mode


def extract_metadata(paths: list[Path], spec: dict, ident: str, records: list[dict], dest: Path) -> Path:
    """Preserve complete GGUF metadata, shard statistics, and provenance in JSON."""
    headers = [GGUFFile(p) for p in paths]
    types = Counter(t.type_name for g in headers for t in g.tensors)
    first = headers[0]
    data = {
        "model_id": ident, "repository": spec["repo"], "revision": spec["revision"],
        "quant_folder": spec["folder"], "gguf_architecture": first.metadata.get("general.architecture"),
        "gguf_version": first.version, "metadata": first.metadata,
        "maya_architecture_check": inspect_architecture(first.metadata),
        "tensor_types": dict(sorted(types.items())), "total_tensors": sum(types.values()),
        "shards": records,
        "note": "Model GGUF bytes are not modified; metadata is read from each GGUF v3 header.",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".maya-partial")
    if tmp.exists():
        raise FileExistsError(f"{tmp} exists")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, dest)  # metadata output only; never overwrite a GGUF
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Maya setup ID, or UD-Q3_K_XL / UD-IQ4_XS")
    ap.add_argument("--source", required=True, type=Path, help="folder holding the GGUF shards (or its UD-* parent)")
    ap.add_argument("--models-dir", type=Path, help="same models root as maya.py --models-dir")
    ap.add_argument("--mode", choices=("hardlink", "symlink", "copy", "move"), default="hardlink",
                    help="how to place existing shards: hardlink avoids copying 148-157 GB (same filesystem)")
    ap.add_argument("--source-prefix", help="old shard filename prefix (explicit rename to published names)")
    ap.add_argument("--metadata-only", action="store_true", help="only extract GGUF metadata, no placement")
    ap.add_argument("--metadata-out", type=Path, help="JSON output path (default: destination/" + METADATA_FILE + ")")
    ap.add_argument("--dry-run", action="store_true", help="show filenames and destinations without SHA hashing or changes")
    args = ap.parse_args(argv)
    try:
        ident, spec = choose_model(args.model)
        if not args.metadata_only and not args.models_dir:
            ap.error("--models-dir is required unless --metadata-only is used")
        src = args.source.expanduser().resolve()
        sources = source_files(src, spec, args.source_prefix)
        target = ((args.models_dir.expanduser().resolve() / ident) if not args.metadata_only else
                  sources[0].parent)
        names = maya.shard_names(spec)
        for path, name in zip(sources, names):
            print(f"{path}  ->  {target / name}" if not args.metadata_only else str(path), flush=True)
        if args.dry_run:
            if not all(p.is_file() for p in sources):
                raise FileNotFoundError("one or more source files are missing (see paths above)")
            print("dry run: SHA-256 and metadata extraction not performed")
            return 0
        print("Verifying all GGUF headers and published SHA-256 digests before placement ...", flush=True)
        rows = validate_files(sources, spec)
        if not args.metadata_only:
            for src_file, name, row in zip(sources, names, rows):
                dest = target / name
                print(f"{name}: {place(src_file, dest, args.mode, row['sha256'])}", flush=True)
            sources = [target / n for n in names]
        out = args.metadata_out.expanduser().resolve() if args.metadata_out else target / METADATA_FILE
        extract_metadata(sources, spec, ident, rows, out)
        print(f"Metadata: {out}")
        if not args.metadata_only:
            print(f"Ready: ./setup.sh --setup --model {ident} --models-dir {args.models_dir} --no-start")
            print("Windows: START-MAYA.bat --setup --model " + ident + " --models-dir " + str(args.models_dir))
        return 0
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

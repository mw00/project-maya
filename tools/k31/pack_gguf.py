"""tools/k31/pack_gguf.py - the engine's pack for one Kimi K3.1 GGUF (llama.cpp's kimi-k31 layout).

Writes a directory of text beside the weights. No weights are copied.

    python tools/k31/pack_gguf.py --model model.gguf
    python tools/k31/pack_gguf.py --model model.gguf --out /tmp/pack
    python tools/k31/pack_gguf.py --model header.gguf --header-only     (a header with no tensor data)

The directory holds (the record grammar: docs/K31-ENGINE.md §3):
  k31.txt       geometry, one record a line
  tensors.txt   every tensor: `file <i> <path>`, then `t <name> <dtype> <file> <offset> <bytes> <dims...>`
                dims are outermost first; offset is absolute in that file
src/k31/k31_pack.hpp reads both and refuses again what this tool refuses.

The packer refuses, before it writes, a file this engine cannot run:
  another architecture;
  an indexer that pools keys, shares a selection, rotates, or skips layers;
  a missing indexer LayerNorm epsilon;
  an A_log (blk.N.ssm_a) whose length is not the KDA head count (it is sliced at conversion, as k31::read requires);
  MoE layers that keep different expert counts, or per-layer expert widths (thin experts are not packed yet);
  an expert slab whose shape is not the geometry's, or whose roles are not one format;
  an expert width that is not a multiple of its format's block (256 values for the IQ and K-quant formats);
  expert ids (the file's kimi-k31.expert_prune keys or the <file>.prune.json sidecar) that k31::read would refuse;
  tensor data that runs past the end of the file (a download or copy still running), unless --header-only.
The sidecar is looked up beside the path as given, so a link to the weights finds the sidecar beside the link, as
maya-k31 does.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "ref"))

from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402
import kimi_k31 as K  # noqa: E402

ARCH = K.ARCH
INDEXER_KEYS = {"head_count", "key_length", "top_k", "rope_dim", "topk_freq", "kpool", "types"}
EXPERT_ROLES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
MAX_SOURCE = 1 << 20   # k31::read's plausibility bound on a source expert count


class PackRefused(Exception):
    """One refusal, already a full message."""


def _no(why: str) -> PackRefused:
    return PackRefused("maya-k31: pack: " + why)


def _meta(meta: dict, key: str, default=None):
    full = key if key.startswith("general.") or key.startswith(ARCH + ".") else f"{ARCH}.{key}"
    return meta.get(full, default)


def _int_list(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)]
    return [int(x) for x in value]


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _scalar_per_layer(meta: dict, key: str, what: str) -> dict:
    """A per-layer array whose entries are all equal becomes that one value; differing entries are refused."""
    full = f"{ARCH}.{key}"
    value = meta.get(full)
    if value is None or isinstance(value, (int, float)):
        return meta
    values = _int_list(value)
    if len(set(values)) != 1:
        raise _no("%s differs per layer (%s): %s" % (key, ", ".join(str(x) for x in sorted(set(values))), what))
    meta = dict(meta)
    meta[full] = values[0]
    return meta


def assess(meta: dict, tensors: list):
    """(KimiK31Config, KDA heads) when the file may be packed; raises PackRefused otherwise."""
    arch = meta.get("general.architecture")
    if arch != ARCH:
        raise _no("architecture %r, kimi-k31 only" % arch)

    pre = f"{ARCH}.attention.indexer."
    for key in meta:
        if key.startswith(pre) and key[len(pre):] not in INDEXER_KEYS:
            raise _no("%s is not a per-token indexer setting" % key)

    rope = _meta(meta, "attention.indexer.rope_dim", 0)
    if int(rope) != 0:
        raise _no("index_rope_dim %s (K3.1's indexer rotates nothing)" % rope)
    freq = _meta(meta, "attention.indexer.topk_freq", 1)
    if int(freq) != 1:
        raise _no("index_topk_freq %s (every MLA layer runs its own indexer)" % freq)
    kpool = _meta(meta, "attention.indexer.kpool", None)
    if kpool is not None and int(kpool) != 0:
        raise _no("index kpool %s (a pooled indexer; K3.1 scores every key)" % kpool)
    if _meta(meta, "attention.layer_norm_epsilon", None) is None:
        raise _no("indexer LayerNorm eps is missing (kimi-k31.attention.layer_norm_epsilon)")

    hkv = _int_list(_meta(meta, "attention.head_count_kv", None))
    types = _int_list(_meta(meta, "attention.indexer.types", None))

    counts = _int_list(_meta(meta, "expert_count", None))
    if counts is None:
        raise _no("expert_count is missing")
    if len(set(counts)) != 1:
        raise _no("MoE layers keep different expert counts (%s)" % ", ".join(str(x) for x in sorted(set(counts))))
    meta = dict(meta)
    meta[f"{ARCH}.expert_count"] = counts[0]
    meta = _scalar_per_layer(meta, "expert_feed_forward_length",
                             "per-layer expert widths (thin experts) are not packed yet")
    # checked above; the oracle's config reads neither key (a NoPE, every-layer indexer is its only one)
    meta.pop(f"{ARCH}.attention.indexer.rope_dim", None)
    meta.pop(f"{ARCH}.attention.indexer.topk_freq", None)
    # the indexer types, whole, before KimiK31Config: it indexes them per MLA layer and folds a shared selection into
    # a different sentence
    if types is not None:
        n_layers = int(_meta(meta, "block_count", 93))
        if len(types) == 1:
            types = types * n_layers
        if len(types) != n_layers:
            raise _no("attention.indexer.types has %d entries for %d layers" % (len(types), n_layers))
        if hkv is None:
            mla = K.released_mla_layers(n_layers) if n_layers == 93 else []
        else:
            mla = [il for il, heads in enumerate(hkv) if heads != 0] if len(hkv) == n_layers else []
        for il in mla:
            if types[il] != 1:
                raise _no("layer %d shares another layer's indexer selection" % il)
        meta[f"{ARCH}.attention.indexer.types"] = types

    try:
        cfg = K.KimiK31Config(meta)
    except (ValueError, NotImplementedError, TypeError) as exc:
        raise _no(str(exc)) from None

    by_name = {t.name: t for t in tensors}
    kda_heads = int(_meta(meta, "kda.head_count", cfg.n_head))
    E, D, L = cfg.n_expert, cfg.n_ff_exp, cfg.n_expert_latent
    for il in range(cfg.n_layer_dense_lead, cfg.n_layers):
        gate = by_name.get("blk.%d.ffn_gate_exps.weight" % il)
        if gate is None:
            raise _no("blk.%d.ffn_gate_exps.weight is missing" % il)
        if len(gate.shape) != 3:
            raise _no("blk.%d.ffn_gate_exps.weight is not a stacked expert tensor" % il)
        if int(gate.shape[-1]) != E:
            raise _no("MoE layers keep different expert counts (%s)"
                      % ", ".join(str(x) for x in sorted({int(gate.shape[-1]), E})))
        for role, want in zip(EXPERT_ROLES, ([L, D, E], [L, D, E], [D, L, E])):
            tensor = by_name.get("blk.%d.%s" % (il, role))
            if tensor is None:
                raise _no("blk.%d.%s is missing" % (il, role))
            block = BLOCK_GEOMETRY.get(tensor.type_name)
            if block is None:
                raise _no("blk.%d.%s has ggml type %s, which the pack does not read" % (il, role, tensor.type_name))
            if [int(d) for d in tensor.shape] != want:
                raise _no("blk.%d.%s is %s, the geometry says %s" % (il, role, list(tensor.shape), want))
            for dim in tensor.shape[:-1]:
                if int(dim) % block[0]:
                    raise _no("layer %d expert width %d is not a multiple of %d (%s)"
                              % (il, int(dim), block[0], tensor.type_name))

    for il, heads in enumerate(cfg.n_head_kv):
        if heads != 0:
            continue
        alog = by_name.get("blk.%d.ssm_a" % il)
        if alog is None:
            raise _no("blk.%d.ssm_a is missing" % il)
        if [int(d) for d in alog.shape] != [kda_heads]:
            raise _no("blk.%d.ssm_a is %s for %d KDA heads (A_log is sliced to the head count at conversion)"
                      % (il, list(alog.shape), kda_heads))
    return cfg, kda_heads


def _check_ids(where: str, ids: list, source: int):
    seen = set()
    for j, x in enumerate(ids):
        if not 0 <= x < source:
            raise _no("%s entry %d = %d, outside the source's %d experts" % (where, j, x, source))
        if x in seen:
            raise _no("%s keeps expert %d twice" % (where, x))
        seen.add(x)


def _read_sidecar(side: str, cfg: K.KimiK31Config, moe: list):
    """None when nothing is at the path (not even a link); else (source, tool, {layer: ids}). k31::read's rules."""
    if not os.path.lexists(side):
        return None
    name = "the prune sidecar %s" % side
    if not os.path.isfile(side):
        raise _no("%s is not a regular file (a directory, a link to nothing, or a path that cannot be read)" % name)
    try:
        text = pathlib.Path(side).read_text()
    except OSError as exc:
        raise _no("%s cannot be read (%s)" % (name, exc.strerror)) from None
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise _no("%s is not valid JSON (%s)" % (name, exc)) from None
    if not isinstance(data, dict):
        raise _no("%s is not a JSON object" % name)
    arch = data.get("arch")
    if not isinstance(arch, str):
        raise _no('%s has no "arch" string' % name)
    if arch != ARCH:
        raise _no("%s is for '%s', not '%s'" % (name, arch, ARCH))
    for key in ("source_expert_count", "expert_count"):
        if key not in data:
            raise _no('%s has no "%s"' % (name, key))
        if not _is_int(data[key]):
            raise _no('%s: "%s" is not an integer' % (name, key))
    if data["expert_count"] != cfg.n_expert:
        raise _no("%s keeps %d experts a layer, the file has %d" % (name, data["expert_count"], cfg.n_expert))
    source = data["source_expert_count"]
    if not cfg.n_expert <= source <= MAX_SOURCE:
        raise _no('%s: "source_expert_count" %d is below the file\'s %d experts or implausible'
                  % (name, source, cfg.n_expert))
    layers = data.get("layers")
    if not isinstance(layers, dict):
        raise _no('%s has no "layers" object' % name)
    per = {}
    for key, ids in layers.items():
        il = int(key) if key.isascii() and key.isdigit() and len(key) <= 6 else -1
        if il not in moe or str(il) != key:
            raise _no('%s lists layer "%s", which is not a MoE layer of this file' % (name, key))
        if not isinstance(ids, list):
            raise _no("%s: layer %s is not an array" % (name, key))
        if len(ids) != cfg.n_expert:
            raise _no("%s: layer %s lists %d experts, the file has %d" % (name, key, len(ids), cfg.n_expert))
        if not all(_is_int(x) for x in ids):
            raise _no("%s: layer %s has an id that is not an integer" % (name, key))
        _check_ids("%s: layer %s" % (name, key), ids, source)
        per[il] = list(ids)
    for il in moe:
        if il not in per:
            raise _no("%s has no ids for layer %d" % (name, il))
    tool = data.get("tool")
    return source, tool if isinstance(tool, str) else "", per


def _keep(path: pathlib.Path, meta: dict, cfg: K.KimiK31Config):
    """(source_count, method, {layer: ids in file order}, note). The sources and rules of k31::read: the file's
    expert_prune keys, else the <path>.prune.json sidecar beside the path as given, else the file's own 0..E-1;
    both present must agree."""
    moe = list(range(cfg.n_layer_dense_lead, cfg.n_layers))
    keep = _meta(meta, "expert_prune.keep", None)
    src_key = _meta(meta, "expert_prune.source_count", None)
    in_file = None
    if keep is not None:
        if not isinstance(keep, list) or not all(_is_int(x) for x in keep):
            raise _no("kimi-k31.expert_prune.keep is not an integer array")
        if src_key is None:
            raise _no("kimi-k31.expert_prune.keep without kimi-k31.expert_prune.source_count")
        if not _is_int(src_key):
            raise _no("kimi-k31.expert_prune.source_count is not an integer")
        if not cfg.n_expert <= src_key <= MAX_SOURCE:
            raise _no("kimi-k31.expert_prune.source_count = %d (below the file's %d experts, or implausible)"
                      % (src_key, cfg.n_expert))
        if len(keep) != len(moe) * cfg.n_expert:
            raise _no("kimi-k31.expert_prune.keep has %d ids for %d MoE layers x %d experts"
                      % (len(keep), len(moe), cfg.n_expert))
        per = {il: [int(x) for x in keep[j * cfg.n_expert:(j + 1) * cfg.n_expert]] for j, il in enumerate(moe)}
        for il, ids in per.items():
            _check_ids("kimi-k31.expert_prune.keep layer %d" % il, ids, src_key)
        method = _meta(meta, "expert_prune.method", None)
        if method is not None and not isinstance(method, str):
            raise _no("kimi-k31.expert_prune.method is not a string")
        in_file = (src_key, method or "file", per)
    elif src_key is not None:
        raise _no("kimi-k31.expert_prune.source_count without kimi-k31.expert_prune.keep")

    side_path = str(path) + ".prune.json"
    side = _read_sidecar(side_path, cfg, moe)
    if in_file and side:
        if side[2] != in_file[2]:
            raise _no("the prune sidecar %s disagrees with kimi-k31.expert_prune.keep" % side_path)
        if side[0] != in_file[0]:
            raise _no('the prune sidecar %s says "source_expert_count" %d, kimi-k31.expert_prune.source_count %d'
                      % (side_path, side[0], in_file[0]))
    if in_file:
        return in_file[0], in_file[1], in_file[2], None
    if side:
        return side[0], side[1] or "sidecar", side[2], None
    note = ("no expert_prune keys and no prune sidecar (%s): the file's experts are its own ids 0..%d"
            % (side_path, cfg.n_expert - 1))
    return cfg.n_expert, "none", {il: list(range(cfg.n_expert)) for il in moe}, note


def _sff(cfg: K.KimiK31Config, by_name: dict, il: int) -> int:
    tensor = by_name.get("blk.%d.ffn_gate_shexp.weight" % il)
    if tensor is None:
        raise _no("blk.%d.ffn_gate_shexp.weight is missing" % il)
    others = [int(d) for d in tensor.shape if int(d) != cfg.n_embd]
    if len(others) != 1:
        raise _no("blk.%d.ffn_gate_shexp.weight shape %s is not embedding x shared width" % (il, list(tensor.shape)))
    return others[0]


def _fmt(value) -> str:
    if isinstance(value, float):
        return "%.9g" % value
    return str(value)


def build_lines(path: pathlib.Path, *, header_only: bool = False):
    """Return (k31_lines, tensor_lines, summary) or raise PackRefused. `path` is used as given (not resolved): the
    prune sidecar is "<path>.prune.json"."""
    gguf = GGUFFile(path)
    file_size = os.stat(path).st_size
    cfg, kda_heads = assess(gguf.metadata, gguf.tensors)
    by_name = {t.name: t for t in gguf.tensors}
    source, method, keep, note = _keep(path, gguf.metadata, cfg)

    tensor_lines = []
    data_end = gguf.data_start
    for tensor in gguf.tensors:
        block = BLOCK_GEOMETRY.get(tensor.type_name)
        if block is None:
            raise _no("%s has ggml type %s, which the pack does not read" % (tensor.name, tensor.type_name))
        nbytes = tensor.expected_bytes()
        if nbytes is None or not tensor.shape or int(tensor.shape[0]) % block[0]:
            raise _no("%s rows of %s are not whole %s blocks"
                      % (tensor.name, tensor.shape[0] if tensor.shape else 0, tensor.type_name))
        absolute = gguf.data_start + int(tensor.offset)
        data_end = max(data_end, absolute + nbytes)
        dims = " ".join(str(int(d)) for d in reversed(tensor.shape))
        tensor_lines.append((tensor.name, tensor.type_name, absolute, nbytes, dims))
    if data_end <= file_size:
        data = "whole"
    elif header_only:
        data = "header-only"
    else:
        raise _no("%s is not whole: its tensor data ends at byte %d, the file has %d (a download or copy still "
                  "running?  --header-only packs a header for --pack-info and tests)" % (path, data_end, file_size))

    context = int(_meta(gguf.metadata, "context_length", 1048576))
    rope = int(_meta(gguf.metadata, "attention.indexer.rope_dim", 0))
    freq = int(_meta(gguf.metadata, "attention.indexer.topk_freq", 1))
    lines = [
        "format gguf",
        "arch %s" % ARCH,
        "data %s" % data,
        "n_layers %d" % cfg.n_layers,
        "n_embd %d" % cfg.n_embd,
        "vocab %d" % cfg.n_vocab,
        "n_head %d" % cfg.n_head,
        "context %d" % context,
        "rms_eps %s" % _fmt(float(cfg.eps_rms)),
        "lora_eps %s" % _fmt(float(cfg.eps_mla_latent)),
        "kda_l2_eps %s" % _fmt(float(cfg.eps_kda_l2)),
        "idx_ln_eps %s" % _fmt(float(cfg.eps_indexer_ln)),
        "router_renorm %s" % _fmt(float(cfg.eps_router_renorm)),
        "attn_res_block %d" % cfg.attn_res_block_size,
        "kda_heads %d" % kda_heads,
        "kda_head_dim %d" % cfg.kda_head_dim,
        "d_conv %d" % cfg.ssm_d_conv,
        "gate_lower_bound %s" % _fmt(float(cfg.kda_gate_lower_bound)),
        "mla_heads %d" % cfg.n_head,
        "kv_lora %d" % cfg.kv_lora_rank,
        "qk_nope %d" % cfg.qk_nope,
        "qk_rope %d" % cfg.n_rot,
        "v_head %d" % cfg.n_head_v_mla,
        "q_lora %d" % cfg.q_lora_rank,
        "idx_heads %d" % cfg.idx_n_head,
        "idx_dim %d" % cfg.idx_key_dim,
        "idx_topk %d" % cfg.idx_top_k,
        "idx_rope_dim %d" % rope,
        "idx_topk_freq %d" % freq,
    ]
    classes = []
    for il in range(cfg.n_layers):
        attn = "kda" if cfg.is_recr(il) else "mla"
        if il < cfg.n_layer_dense_lead:
            lines.append("layer %d attn %s ffn dense n_exp 0 k 0 edim 0 eff %d sff 0 norm_div %d"
                         % (il, attn, cfg.n_ff, cfg.n_expert_latent))
            continue
        roles = []
        formats = []
        for role in EXPERT_ROLES:
            tensor = by_name["blk.%d.%s" % (il, role)]
            roles.append(tensor.name)
            formats.append(tensor.type_name)
        if len(set(formats)) != 1:
            raise _no("layer %d experts are not one format (gate %s, up %s, down %s): a class holds one format"
                      % (il, *formats))
        sff = _sff(cfg, by_name, il)
        lines.append("layer %d attn %s ffn moe n_exp %d k %d edim %d eff %d sff %d norm_div %d"
                     % (il, attn, cfg.n_expert, cfg.n_expert_used, cfg.n_expert_latent, cfg.n_ff_exp,
                        sff, cfg.n_expert_latent))
        classes.append("class %d 0 %s %d %s" % (il, formats[0], cfg.n_expert, " ".join(roles)))
    lines.extend(classes)
    for il in range(cfg.n_layer_dense_lead, cfg.n_layers):
        lines.append("keep %d %s" % (il, " ".join(str(x) for x in keep[il])))
    lines.append("prune source_count %d method %s" % (source, method))

    summary = {
        "n_layers": cfg.n_layers,
        "kda": sum(1 for h in cfg.n_head_kv if h == 0),
        "mla": sum(1 for h in cfg.n_head_kv if h != 0),
        "n_expert": cfg.n_expert,
        "source": source,
        "data": data,
        "note": note,
    }
    return lines, tensor_lines, summary


def write_pack(model: pathlib.Path, out: pathlib.Path | None, *, header_only: bool = False) -> pathlib.Path:
    # the path as given (absolute, its last component not resolved): a link to the weights keeps its sidecar and its
    # default pack beside the link; the directories are resolved, so the relative model_file holds through links
    model = pathlib.Path(model).expanduser().absolute()
    if not model.is_file():
        raise _no("%s is not a file" % model)
    model = model.parent.resolve() / model.name
    lines, tensors, summary = build_lines(model, header_only=header_only)
    if summary["note"]:
        print("maya-k31: pack: note: %s" % summary["note"], file=sys.stderr)
    dest = out.expanduser().resolve() if out else model.parent / (model.stem + ".maya-k31")
    dest.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(model, dest).replace(os.sep, "/")
    if any(c.isspace() for c in rel):
        raise _no("the model path relative to the pack has a space: %s" % rel)
    lines.append("model_file %s" % rel)
    (dest / "k31.txt").write_text("\n".join(lines) + "\n", newline="\n")
    body = ["file 0 %s" % rel]
    for name, dtype, offset, nbytes, dims in tensors:
        body.append("t %s %s 0 %d %d %s" % (name, dtype, offset, nbytes, dims))
    (dest / "tensors.txt").write_text("\n".join(body) + "\n", newline="\n")
    print("%s: %d layers (%d KDA / %d MLA), %d of %d experts, %d tensors, data %s -> %s" % (
        model.name, summary["n_layers"], summary["kda"], summary["mla"], summary["n_expert"],
        summary["source"], len(tensors), summary["data"], dest))
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=pathlib.Path, help="one kimi-k31 GGUF")
    ap.add_argument("--out", type=pathlib.Path, help="default: <model stem>.maya-k31 beside the file")
    ap.add_argument("--header-only", action="store_true",
                    help="pack a file without (all of) its tensor data, recorded as `data header-only`: the engine "
                         "reads such a pack for --pack-info and tests only")
    args = ap.parse_args()
    try:
        write_pack(args.model, args.out, header_only=args.header_only)
    except PackRefused as exc:
        print(exc, file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print("maya-k31: pack: cannot read %s: %s" % (args.model, exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

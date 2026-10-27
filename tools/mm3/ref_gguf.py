"""tools/mm3/ref_gguf.py - tools/mm3/ref.py's oracle over a GGUF pack (tools/mm3/pack_gguf.py's): the same forward
(ref.py's layer: transformers' MiniMax-M3.1 arithmetic in F32), each weight dequantised by llama.cpp's gguf-py from the
GGUF's own blocks - a check of maya-mm3's GGUF path that shares none of its code.

    python tools/mm3/ref_gguf.py --pack <model>/maya-mm3 --ids 200019,1234,... [--dump ref.npz] [--layers N]

gguf-py: GGUF_PY=<llama.cpp>/gguf-py (or an installed `gguf`).  The GGUF's norms hold 1 + w already (llama.cpp's
converter): w is taken back from them for ref.py's Gemma norm.  --dump writes what tools/mm3/compare.py and
compare_hs (below, --engine-hs) read: the ids, the logits and every layer's residual stream at the last --dump-last
positions.

    python tools/mm3/ref_gguf.py --pack P --ids @ids.txt --dump ref.npz --engine-hs hs.bin
        (the engine's MM3_DUMP_HS file of the same ids: each layer's relative L2 against this forward's)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ref as R  # noqa: E402

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

PFX, LM = R.PFX, R.LM
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight",
         "index_q_norm.weight", "index_k_norm.weight", "model.norm.weight")


class GgufPack:
    """pack_gguf.py's tensors.txt over the GGUF's files (memory-mapped), and mm3.txt's geometry."""

    def __init__(self, d: pathlib.Path):
        self.kv = {}
        for line in (d / "mm3.txt").read_text().splitlines():
            k, _, v = line.partition(" ")
            self.kv[k] = v
        if self.kv.get("format") != "gguf":
            raise SystemExit("%s: not a GGUF pack (tools/mm3/pack_gguf.py)" % d)
        model = pathlib.Path(self.kv["model_dir"])
        self.mm, self.t = {}, {}
        for line in (d / "tensors.txt").read_text().splitlines():
            f = line.split()
            if f[0] == "file":
                self.mm[int(f[1])] = np.memmap(model / f[2], dtype=np.uint8, mode="r")
            elif f[0] == "t":
                self.t[f[1]] = (f[2], int(f[3]), int(f[4]), int(f[5]), tuple(int(x) for x in f[6:]))

    def tc(self) -> dict:
        """config.json's text_config from mm3.txt (the keys ref.Geo reads)."""
        def num(v):
            return float(v) if "." in v or "e" in v else int(v)
        tc, sa = {}, {}
        for k, v in self.kv.items():
            try:
                val = [num(x) for x in v.split(",")] if "," in v else num(v)
            except ValueError:
                continue
            if k.startswith("sparse_attention_config."):
                sa[k.split(".", 1)[1]] = val
            else:
                tc[k] = val
        for k in ("moe_layer_freq",):
            if not isinstance(tc.get(k), list):
                tc[k] = [tc[k]]
        for k in ("sparse_attention_freq", "sparse_disable_index_value"):
            if k in sa and not isinstance(sa[k], list):
                sa[k] = [sa[k]]
        tc["sparse_attention_config"] = sa
        return tc

    def tensor(self, name: str) -> np.ndarray:
        """The tensor as F32 [dims] (dequantised by gguf-py)."""
        dt, fi, off, nb, shape = self.t[name]
        raw = np.asarray(self.mm[fi][off:off + nb])
        qt = gguf.GGMLQuantizationType[dt]
        if dt == "F32":
            return raw.view(np.float32).reshape(shape).copy()
        if dt in ("F16", "BF16"):
            v = raw.view(np.uint16).astype(np.uint32)
            return (v << 16).view(np.float32).reshape(shape) if dt == "BF16" else raw.view(np.float16).astype(
                np.float32).reshape(shape)
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        return dequantize(raw.reshape(rows, -1), qt).reshape(shape).astype(np.float32)


class GgufRef:
    """ref.Ref's interface (g, tc, dev, T, W, layer_weights, experts) over a GGUF pack."""

    def __init__(self, pack: pathlib.Path, dev: str = "cuda:0"):
        self.pk = GgufPack(pack)
        self.tc = self.pk.tc()
        self.g = R.Geo(self.tc)
        self.g.emulate = False
        self.dev = dev

    def T(self, name: str) -> torch.Tensor:
        x = torch.from_numpy(self.pk.tensor(name)).to(self.dev)
        if name.endswith(NORMS):
            x = x - 1.0   # (the GGUF holds 1 + w; ref's Gemma norm adds the 1)
        return x

    def W(self, prefix: str, x16: bool = True) -> torch.Tensor:
        return self.T(prefix + ".weight").float()

    layer_weights = R.Ref.layer_weights

    def experts(self, i: int, ids: list, out_dev: str | None = None) -> dict:
        m = f"{PFX}layers.{i}.block_sparse_moe.experts."
        out = {}
        for e in ids:
            ws = tuple(self.T(f"{m}{e}.{n}.weight") for n in ("w1", "w3", "w2"))
            out[e] = ws if out_dev is None else tuple(w.to(out_dev) for w in ws)
        return out


def forward(ref: GgufRef, ids: list, last: int, log=print):
    g, dev = ref.g, ref.dev
    T = len(ids)
    emb = ref.pk
    dt, fi, off, nb, shape = emb.t[PFX + "embed_tokens.weight"]
    rb = nb // shape[0]
    raw = np.asarray(emb.mm[fi][off:off + nb]).reshape(shape[0], rb)[ids]
    rows = raw.view(np.float32) if dt == "F32" else dequantize(raw, gguf.GGMLQuantizationType[dt])
    h = torch.from_numpy(np.asarray(rows, dtype=np.float32).reshape(T, g.dim)).to(dev)
    pos = torch.arange(T, device=dev)
    hs = []
    for i in range(g.n_layers):
        t0 = time.time()
        w = ref.layer_weights(i)
        h = R.layer(ref, g, i, w, h, pos)
        hs.append(h[T - last:].cpu())
        del w
        torch.cuda.empty_cache()
        log(f"layer {i:2d} {time.time() - t0:5.1f}s  |h| {h.norm(dim=-1).mean().item():.1f}")
    h = R.gemma_rms(h, ref.T(PFX + "norm.weight"), g.eps)
    head = ref.W(LM + "lm_head")
    logits = torch.cat([h[a:a + 256] @ head.T for a in range(0, T, 256)])
    return logits, hs


def compare_hs(path: str, hs: list, T: int, dim: int, last: int) -> None:
    """The engine's MM3_DUMP_HS (every layer's residual stream, F32 [T][dim] a layer, one chunk) against ours."""
    e = np.fromfile(path, dtype="<f4")
    n = len(hs)
    if e.size != n * T * dim:
        print(f"{path}: {e.size} floats, not {n} layers x {T} x {dim} (one chunk of the same ids?)")
        return
    e = e.reshape(n, T, dim)[:, T - last:]
    for i in range(n):
        r = hs[i].numpy().astype(np.float64)
        d = e[i].astype(np.float64) - r
        print(f"layer {i:2d}: rel L2 {np.linalg.norm(d) / np.linalg.norm(r):.3e}  max |d| {np.abs(d).max():.4f}  "
              f"|h| {np.linalg.norm(r, axis=-1).mean():.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated token ids, or @FILE")
    ap.add_argument("--dump", help="write the ids, logits and residual streams (.npz)")
    ap.add_argument("--dump-last", type=int, default=64)
    ap.add_argument("--engine-hs", help="the engine's MM3_DUMP_HS file of the same ids: compared layer by layer")
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    src = pathlib.Path(a.ids[1:]).read_text() if a.ids.startswith("@") else a.ids
    ids = [int(x) for x in src.replace(",", " ").split()]
    ref = GgufRef(pathlib.Path(a.pack).expanduser(), a.dev)
    T = len(ids)
    last = min(T, a.dump_last)
    with torch.no_grad():
        logits, hs = forward(ref, ids, last)
    print(f"T={T} next {int(logits[-1].argmax())} top5 {logits[-1].topk(5).indices.tolist()}")
    if a.engine_hs:
        compare_hs(a.engine_hs, hs, T, ref.g.dim, last)
    if a.dump:
        np.savez(a.dump, ids=np.array(ids), logits=logits[-last:].cpu().numpy(), pos=np.arange(T - last, T),
                 hs=torch.stack(hs).numpy())
        print("wrote", a.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())

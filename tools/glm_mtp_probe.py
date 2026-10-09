"""MTP (NextN, blk.45) acceptance probe for GLM-5.3-Flash, numpy reference.

    python mtp_ref.py <gguf_dir> <h.bin> <pack_test.out> <ids file> [n_pos] [order]

h.bin: the engine's final hidden states per position (GLM_DUMP_H, n x 4096 f32), pack_test.out: its "pos t tok X argmax Y"
lines (the main model's greedy prediction at every position).  For position p the draft block reads (h_p, emb(tok_{p+1}))
and predicts tok_{p+2}; that is compared with the main model's argmax at p+1 (what a greedy speculative decode would
accept) and with the text itself.  Positions < 2048 only (full causal attention, no indexer selection needed).
"""
import glob, os, sys, time
import numpy as np
import gguf
from gguf import quants

gdir, hpath, outpath, idspath = sys.argv[1:5]
NPOS = int(sys.argv[5]) if len(sys.argv) > 5 else 300
ORDER = sys.argv[6] if len(sys.argv) > 6 else "eh"      # concat order: "eh" = [enorm(emb), hnorm(h)], "he" = swapped
CACHE = os.path.expanduser("~/maya/mtp_cache")
os.makedirs(CACHE, exist_ok=True)

readers = [gguf.GGUFReader(p) for p in sorted(glob.glob(os.path.join(gdir, "*.gguf")))]
T = {}
meta = {}
for r in readers:
    for t in r.tensors:
        T[t.name] = t
    for k, f in r.fields.items():
        if k.startswith(("glm5next.", "glm5-next.")) and len(f.data) == 1:   # both spellings (gguf_fix_arch.py)
            try:
                meta["glm5next." + k.split(".", 1)[1]] = f.parts[f.data[0]].tolist()
            except Exception:
                pass
for k in sorted(meta):
    if any(s in k for s in ("scale", "norm", "swiglu", "eps", "gating", "expert")):
        print(k, meta[k])

def deq(name, cache=True):
    fn = os.path.join(CACHE, name + ".npy")
    if cache and os.path.exists(fn):
        return np.load(fn, mmap_mode="r")
    t = T[name]
    a = quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    if cache and a.size > 1 << 20:
        np.save(fn, a.astype(np.float16))
        return np.load(fn, mmap_mode="r")
    return a

def f32(name):
    return np.asarray(T[name].data, dtype=np.float32).reshape(-1)

def rms(x, w, eps=1e-5):
    return x / np.sqrt(np.mean(x * x) + eps) * w

P = "blk.45."
t0 = time.time()
emb = deq("token_embd.weight")          # (V, 4096) f16 memmap
head = deq("output.weight")             # (V, 4096)
eh = deq(P + "nextn.eh_proj.weight")    # (4096, 8192)
q_a = deq(P + "attn_q_a.weight"); q_b = deq(P + "attn_q_b.weight"); kv_a = deq(P + "attn_kv_a_mqa.weight")
k_b = np.asarray(deq(P + "attn_k_b.weight"), dtype=np.float32)   # (64, 512, 256)
v_b = np.asarray(deq(P + "attn_v_b.weight"), dtype=np.float32)   # (64, 256, 512)
o_w = deq(P + "attn_output.weight")     # (4096, 16384)
router = deq(P + "ffn_gate_inp.weight", cache=False).reshape(288, 4096)
bias = f32(P + "exp_probs_b.bias")
sg = deq(P + "ffn_gate_shexp.weight"); su = deq(P + "ffn_up_shexp.weight"); sd = deq(P + "ffn_down_shexp.weight")
gate_e = deq(P + "ffn_gate_exps.weight"); up_e = deq(P + "ffn_up_exps.weight"); down_e = deq(P + "ffn_down_exps.weight")
nrm = {n: f32(P + n) for n in ("attn_norm.weight", "ffn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                               "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight")}
print("weights ready in %.0f s; shapes eh %s q_a %s q_b %s kv_a %s k_b %s v_b %s o %s gate_e %s down_e %s" % (
    time.time() - t0, eh.shape, q_a.shape, q_b.shape, kv_a.shape, k_b.shape, v_b.shape, o_w.shape, gate_e.shape,
    down_e.shape))
w_scale = float(meta.get("glm5next.expert_weights_scale", [1.0])[0])
norm_w = bool(meta.get("glm5next.expert_weights_norm", [1])[0])
LIM = 10.0

def swiglu(g, u, lim=LIM):
    g = np.minimum(g, lim)
    u = np.clip(u, -lim, lim)
    return g / (1.0 + np.exp(-g)) * u

def mv(W, x):
    return np.asarray(W, dtype=np.float32) @ x

ids = [int(v) for v in open(idspath).read().replace(",", " ").split()]
H = np.fromfile(hpath, dtype=np.float32).reshape(-1, 4096)
main_am = {}
for line in open(outpath):
    a = line.split()
    if len(a) >= 6 and a[0] == "pos":
        main_am[int(a[1])] = int(a[5])
n = min(NPOS, len(H) - 2, 2040)
lat = np.zeros((n, 512), dtype=np.float32)
acc_main = acc_text = tot = 0
t1 = time.time()
for p in range(n):
    e = np.asarray(emb[ids[p + 1]], dtype=np.float32)
    a_in = rms(e, nrm["nextn.enorm.weight"]); b_in = rms(H[p], nrm["nextn.hnorm.weight"])
    cat = np.concatenate([a_in, b_in]) if ORDER == "eh" else np.concatenate([b_in, a_in])
    hid = mv(eh, cat)
    # attention (DSA/MLA, every earlier cell selected below 2048 positions)
    x = rms(hid, nrm["attn_norm.weight"])
    qr = rms(mv(q_a, x), nrm["attn_q_a_norm.weight"])
    q = mv(q_b, qr).reshape(64, 256)
    lat[p] = rms(mv(kv_a, x), nrm["attn_kv_a_norm.weight"])
    qabs = np.einsum("hcd,hd->hc", k_b, q)                 # (64, 512)
    sc = qabs @ lat[:p + 1].T / np.sqrt(256.0)              # (64, p+1)
    sc -= sc.max(axis=1, keepdims=True)
    pr = np.exp(sc); pr /= pr.sum(axis=1, keepdims=True)
    ctx = pr @ lat[:p + 1]                                  # (64, 512)
    o = np.einsum("hvc,hc->hv", v_b, ctx).reshape(-1)       # (16384,)
    hid = hid + mv(o_w, o)
    # MoE
    x = rms(hid, nrm["ffn_norm.weight"])
    logit = router @ x
    pe = 1.0 / (1.0 + np.exp(-logit))
    sel = np.argsort(-(pe + bias), kind="stable")[:8]
    w = pe[sel] / max(pe[sel].sum(), 6.103515625e-5) if norm_w else pe[sel]
    w = w * w_scale
    y = mv(sd, swiglu(mv(sg, x), mv(su, x)))
    for j, ex in enumerate(sel):
        hh = swiglu(mv(gate_e[ex], x), mv(up_e[ex], x))
        y = y + w[j] * mv(down_e[ex], hh)
    hid = hid + y
    hf = rms(hid, nrm["nextn.shared_head_norm.weight"])
    am = int(np.argmax(np.asarray(head, dtype=np.float32) @ hf))
    tot += 1
    acc_main += (p + 1) in main_am and am == main_am[p + 1]
    acc_text += p + 2 < len(ids) and am == ids[p + 2]
    if (p + 1) % 25 == 0:
        print("pos %d: draft == main argmax %.1f%%, == text %.1f%% (%.1f s/pos)" % (
            p + 1, 100.0 * acc_main / tot, 100.0 * acc_text / tot, (time.time() - t1) / (p + 1)), flush=True)
print("FINAL order=%s: %d positions, draft == main argmax %.1f%%, draft == text %.1f%%" % (
    ORDER, tot, 100.0 * acc_main / tot, 100.0 * acc_text / tot))

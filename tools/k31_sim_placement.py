"""k31_sim_placement.py: a placement-sim tier table -> a maya-k31 --placement file (v1).

  python3 tools/k31_sim_placement.py TIER OUT [--sim DIR] [--base FILE] [--gpu3 drop|fail]

TIER is a name in placement_sim.TIERS (today, pl, q4_pl, g3_pl, q4g3_pl, ...). Its table marks each (layer, expert):
1 = static VRAM on the layer's own card (0-2), 2 = static on GPU3 (a pool over all layers).

The engine runs each layer on one card and keeps that layer's VRAM experts on the same card. So mark 1 maps straight
to the layer's `slots`. Mark 2 (GPU3 holding experts of layers that run on cards 0-2) has no form in the file: with
--gpu3 fail (default) such a tier is refused; with --gpu3 drop those experts stay in RAM and the count is printed.

Base layout (gpus, each layer's card, dense in VRAM or host): from --base (a file the engine wrote with
--write-placement), else the KLX layout the engine chose on its own: gpus 0 1 2, layers 0-30 / 31-61 / 62-92, all
dense in VRAM. The sim's card map (layers 1-30 card 0, 31-61 card 1, 62-92 card 2) must agree with the base.

Slots are written hottest first by the sim's calibration picks-per-token. Per card, the slot bytes are printed next to
the base's (today's 2,523 slots), since the engine places what the file says and a card that cannot hold it fails at
load: a tier over today's bytes needs the VRAM it assumes (dense Q4_K for q4_*)."""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tier")
ap.add_argument("out")
ap.add_argument("--sim", default=os.path.expanduser("~/work/k31-notes/placement"))
ap.add_argument("--base")
ap.add_argument("--gpu3", choices=("drop", "fail"), default="fail")
a = ap.parse_args()

sys.path.insert(0, a.sim)
import placement_sim as S   # noqa: E402  (builds the tier tables and the byte sizes from the trace)

if a.tier not in S.TIERS:
    sys.exit(f"unknown tier {a.tier}; have {', '.join(S.TIERS)}")
t = S.TIERS[a.tier]   # [92][896], index li = layer - 1
NL = S.NL + 1         # the engine's layers: 0 (dense, no experts) .. 92

# base layout
gpus, dev, dense = [0, 1, 2], [], []
if a.base:
    gpus, dev, dense = None, [None] * NL, [None] * NL
    for ln in open(a.base):
        w = ln.split()
        if not w or w[0].startswith("#") or w[0] == "maya-k31-placement":
            continue
        if w[0] == "gpus":
            gpus = list(map(int, w[1:]))
        elif w[0] == "layer":
            il = int(w[1])
            if il < NL:
                dev[il], dense[il] = int(w[3]), w[5]
    if gpus is None or None in dev:
        sys.exit(f"{a.base}: no gpus line or a layer missing")
else:
    dev = [0 if il <= 30 else 1 if il <= 61 else 2 for il in range(NL)]
    dense = ["vram"] * NL
for il in range(1, NL):
    if gpus[dev[il]] != int(S.CARD[il - 1]):
        sys.exit(f"layer {il}: the base puts it on cuda {gpus[dev[il]]}, the sim on card {int(S.CARD[il - 1])}")

n3 = int((t == 2).sum())
if n3 and a.gpu3 == "fail":
    sys.exit(f"tier {a.tier} puts {n3} experts on GPU3 for layers on cards 0-2: no form in a placement file "
             "(--gpu3 drop keeps them in RAM)")

with open(a.out, "w") as f:
    f.write(f"maya-k31-placement 1\n# placement_sim tier {a.tier} ({n3} GPU3 experts dropped to RAM)\n")
    f.write("gpus " + " ".join(map(str, gpus)) + "\n")
    for il in range(NL):
        f.write(f"layer {il} dev {dev[il]} dense {dense[il]}")
        if il:
            li = il - 1
            es = np.where(t[li] == 1)[0]
            es = es[np.argsort(-S.PPT[li][es], kind="stable")]
            f.write(f" slots {len(es)} " + " ".join(map(str, es.tolist())))
        f.write("\n")

today = S.TIERS["today"]
for g in range(3):
    lis = np.where(S.CARD == g)[0]
    nb = float((S.SZ[lis, None] * (t[lis] == 1)).sum())
    tb = float((S.SZ[lis, None] * (today[lis] == 1)).sum())
    print(f"card {g}: {int((t[lis] == 1).sum())} slots, {nb / 1e9:.2f} GB (today {int((today[lis] == 1).sum())}, "
          f"{tb / 1e9:.2f} GB, {100 * (nb - tb) / tb:+.1f}%)")
print(f"wrote {a.out}" + (f"; {n3} GPU3 experts left in RAM" if n3 else ""))

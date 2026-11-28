"""tools/ling_quant/by_category.py - per_seq.jsonl rows broken down by the eval set's categories (prep_calib's order).

    python by_category.py per_seq.jsonl
"""
import json
import math
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
cats = [("chat", range(0, 6)), ("multiling", range(6, 9)), ("reasoning", range(9, 15)), ("web", range(15, 22)),
        ("code", range(22, 25)), ("tools", range(25, 28)), ("wikitext", range(28, 32))]
for title, key, f in (("PPL", "nll", math.exp), ("KL", "kl", lambda v: v), ("same top %", "same", lambda v: 100 * v)):
    print(title.ljust(20) + "".join(c.rjust(10) for c, _ in cats))
    for r in rows:
        ps = {p["seq"]: p for p in r["per_seq"]}
        print(r["label"][:19].ljust(20) + "".join(f"{f(sum(ps[s][key] for s in rg) / len(rg)):10.3f}" for _, rg in cats))
    print()

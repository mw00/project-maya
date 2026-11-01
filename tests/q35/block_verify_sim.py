"""tests/q35/block_verify_sim.py - dense_main.cpp's block verification (STRATA_DENSE_BLOCK_VERIFY) against the token test: the generated
tokens' joint distribution against the target's, Monte Carlo (python3 tests/q35/block_verify_sim.py)."""
# Monte Carlo: block verification (as dense_main.cpp implements it) leaves the generated tokens' joint distribution the
# target's.  Vocabulary 4, drafts of G tokens; target P(x | prefix) and draft Q(x | prefix) random per prefix.
import random, itertools, collections
V, G, N = 4, 3, 400000
rnd = random.Random(1)
cacheP, cacheQ = {}, {}
def dist(cache, pre):
    if pre not in cache:
        w = [rnd.random() ** 3 for _ in range(V)]; s = sum(w); cache[pre] = [x / s for x in w]
    return cache[pre]
P = lambda pre: dist(cacheP, pre); Q = lambda pre: dist(cacheQ, pre)
def draw(p, r):
    u, c = r.random(), 0.0
    for i, x in enumerate(p):
        c += x
        if u < c: return i
    return len(p) - 1
def round_(pre, r, block):
    xs, cur = [], pre
    for _ in range(G):
        x = draw(Q(cur), r); xs.append(x); cur = cur + (x,)
    pres = [pre + tuple(xs[:i]) for i in range(G + 1)]
    if block:
        pw = [1.0] * (G + 1); tau = 0
        for i in range(1, G + 1):
            x = xs[i - 1]; q = Q(pres[i - 1])[x]; pb = P(pres[i - 1])[x]
            pw[i] = min(pw[i - 1] * pb / q, 1.0) if q > 0 else 0.0
            h = pw[i]
            if i < G:
                res = sum(max(pw[i] * a - b, 0.0) for a, b in zip(P(pres[i]), Q(pres[i])))
                h = res / (res + 1.0 - pw[i]) if res + 1.0 - pw[i] > 0 else 1.0
            if r.random() <= h: tau = i
        if tau == G: y = draw(P(pres[G]), r)
        else:
            rr = [max(0.0, pw[tau] * a - b) for a, b in zip(P(pres[tau]), Q(pres[tau]))]; t = sum(rr)
            y = draw([x / t for x in rr], r) if t > 0 else draw(P(pres[tau]), r)
    else:
        tau = 0
        for i in range(1, G + 1):
            x = xs[i - 1]
            if r.random() < min(1.0, P(pres[i - 1])[x] / Q(pres[i - 1])[x]): tau = i
            else: break
        if tau == G: y = draw(P(pres[G]), r)
        else:
            rr = [max(0.0, a - b) for a, b in zip(P(pres[tau]), Q(pres[tau]))]; t = sum(rr)
            y = draw([x / t for x in rr], r) if t > 0 else draw(P(pres[tau]), r)
    return tuple(xs[:tau]) + (y,), tau
L = 3   # compare the distribution of the first L generated tokens
def run(block):
    r = random.Random(7); cnt = collections.Counter(); acc = 0; rounds = 0
    for _ in range(N):
        out = ()
        while len(out) < L:
            toks, tau = round_(out, r, block); out += toks; acc += tau; rounds += 1
        cnt[out[:L]] += 1
    return cnt, acc / rounds
def exact():
    e = {}
    for seq in itertools.product(range(V), repeat=L):
        p, pre = 1.0, ()
        for x in seq: p *= P(pre)[x]; pre += (x,)
        e[seq] = p
    return e
for block in (False, True):
    cnt, a = run(block); e = exact()
    tv = 0.5 * sum(abs(cnt[s] / N - e[s]) for s in e)
    print("block" if block else "token", "TV distance %.4f" % tv, "drafts kept a round %.3f" % a)

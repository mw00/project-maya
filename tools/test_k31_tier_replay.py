"""tools/test_k31_tier_replay.py - unit tests for tools/k31/tier_replay.py (numpy only, synthetic traces)."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parent / "k31" / "tier_replay.py"
_s = importlib.util.spec_from_file_location("tier_replay", _p)
tr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(tr)


def args(**kw) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    tr.add_rate_args(ap)
    a = ap.parse_args(["--n-kept", str(kw.pop("n_kept", 64))])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def brute_ms(r, v, w, d) -> float:
    """the same lane model over every assignment of the w warm + d cold picks (ds4_pcie split's way)."""
    zc_full = min(r.n_gpu * r.zc_gbs_gpu, r.bus_gbs)
    shrink = min(1.0, r.bus_gbs / (zc_full + r.cpu_gbs))
    disk = r.ms(r.nvme_gbs)
    picks = [0] * w + [1] * d
    best = math.inf
    for mask in itertools.product((0, 1), repeat=len(picks)):
        nz = sum(mask)
        nc = len(picks) - nz
        f = shrink if (nz and nc) else 1.0
        tc = sum(r.ms(r.cpu_gbs * f) + (disk if cold else 0) for m, cold in zip(mask, picks) if not m)
        tz = sum(r.ms(zc_full * f) + (disk if cold else 0) for m, cold in zip(mask, picks) if m)
        tc += r.oc if nc else 0
        tz += r.oz if nz else 0
        best = min(best, max(tc, tz))
    return best + v * r.ms(r.vram_gbs)


class TestCost(unittest.TestCase):
    def test_split_matches_brute_force(self):
        r = tr.Rates(args(nvme_gbs=1.4))
        for v, w, d in [(0, 8, 0), (2, 3, 3), (0, 0, 6), (4, 2, 1), (1, 5, 2)]:
            self.assertAlmostEqual(tr.request_ms(r, v, w, d), brute_ms(r, v, w, d), places=9)

    def test_equal_lanes_split_evenly(self):
        r = tr.Rates(args(cpu_gbs=20.0, zc_gbs=20.0, zc_gpus=1, bus_gbs=1e9, oc_ms=0, oz_ms=0))
        one = r.ms(20.0)
        self.assertAlmostEqual(tr.request_ms(r, 0, 16, 0), 8 * one, places=9)

    def test_no_zc_lane_is_cpu_only(self):
        r = tr.Rates(args(zc_gpus=0, oc_ms=0))
        self.assertAlmostEqual(tr.request_ms(r, 0, 4, 0), 4 * r.ms(r.cpu_gbs), places=9)

    def test_faster_nvme_never_slower(self):
        prev = math.inf
        for g in (0.5, 1.4, 3.5, 7.0):
            t = tr.request_ms(tr.Rates(args(nvme_gbs=g)), 2, 6, 8)
            self.assertLessEqual(t, prev + 1e-12)
            prev = t


class TestResidency(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.E, self.K, self.T = 64, 4, 400
        p = 1.0 / np.arange(1, self.E + 1)
        p /= p.sum()
        self.ids = np.stack([rng.choice(self.E, self.K, replace=False, p=p) for _ in range(self.T)]).astype(np.int16)
        self.kept = np.ones(self.E, dtype=bool)

    def test_everything_resident_no_nvme(self):
        c = tr.residency_static(self.ids, self.E, self.kept, 16, 48, 0.25)
        self.assertEqual(int(c[:, 2].sum()), 0)
        self.assertTrue((c.sum(1) == self.K).all())

    def test_lru_working_set_fits(self):
        ids = np.tile(np.arange(8, dtype=np.int16).reshape(2, 4), (50, 1))       # 8 experts repeating
        c = tr.residency_cache(ids, self.E, self.kept, 0, 8, 0.1, "lru")
        self.assertLessEqual(int(c[:, 2].sum()), 8)                               # only first touches miss

    def test_static_beats_nothing_and_kept_mask(self):
        kept = np.zeros(self.E, dtype=bool)
        kept[::2] = True
        c = tr.residency_static(self.ids, self.E, kept, 4, 4, 0.25)
        f = tr._freq(self.ids[:100], self.E).astype(float)
        f[~kept] = -1
        hot = np.argsort(-f, kind="stable")[:8]
        self.assertTrue(kept[hot].all())                                          # never places a pruned expert
        self.assertGreater(int(c[:, 0].sum() + c[:, 1].sum()), 0)

    def test_lfu_runs(self):
        c = tr.residency_cache(self.ids, self.E, self.kept, 4, 8, 0.25, "lfu")
        self.assertTrue((c.sum(1) == self.K).all())


class TestReplay(unittest.TestCase):
    def test_end_to_end_and_monotone_in_ram(self):
        rng = np.random.default_rng(1)
        E, K, T, L = 896, 16, 300, 3
        ids = np.stack([np.stack([rng.choice(E, K, replace=False) for _ in range(T)]) for _ in range(L)]).astype(np.int16)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.npz"
            np.savez(p, ids=ids, layers=np.array([1, 2, 3]), n_expert=E, top_k=K, stride=8, keep="null", source="{}")
            t = tr.load_trace(str(p))
            prev = 0.0
            for ram in (60.0, 123.0, 384.0, 768.0):
                res = tr.replay(t, args(n_kept=896, ram_gb=ram))
                self.assertTrue(math.isfinite(res["tok_s"]))
                self.assertGreaterEqual(res["tok_s"], prev - 1e-9)
                prev = res["tok_s"]
            self.assertAlmostEqual(res["frac_nvme"], 0.0)                         # 768 GB holds all 896


if __name__ == "__main__":
    unittest.main()


class TestReplayFreq(unittest.TestCase):
    def test_fit_means_no_nvme_and_spill_costs(self):
        rng = np.random.default_rng(2)
        counts = {l: list(rng.integers(1, 1000, 896)) for l in range(1, 5)}
        fit = tr.replay_freq(counts, args(n_kept=192, ram_gb=123.0), n_samples=400)
        self.assertAlmostEqual(fit["frac_nvme"], 0.0)                              # 192 x 92 x 6.45 MB fits
        spill = tr.replay_freq(counts, args(n_kept=448, ram_gb=123.0), n_samples=400)
        self.assertGreater(spill["frac_nvme"], 0.0)
        self.assertLess(spill["tok_s"], fit["tok_s"])

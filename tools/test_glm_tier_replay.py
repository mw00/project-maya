"""glm_tier_replay tests; run .venv/bin/python -m unittest discover -s tools -p test_glm_tier_replay.py."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import glm_tier_replay as gtr

TOP8 = list(range(8))


def usage_for(layers, n_expert=16):
    """Expert e of every layer used (n_expert - e) * 10 times: experts 0..7 are the most used."""
    return {il: {e: (n_expert - e) * 10 for e in range(n_expert)} for il in layers}


def route(il, ids, fetch=0, miss=0, cpu=0, near=None):
    line = f"{il} " + " ".join(map(str, ids)) + f" {fetch} {miss} {cpu}"
    return line + (" | " + " ".join(map(str, near)) if near else "")


class ParseTest(unittest.TestCase):
    def test_route_with_and_without_near_misses(self):
        a = gtr.parse_route(route(3, TOP8, fetch=1, cpu=6))
        b = gtr.parse_route(route(3, TOP8, fetch=1, cpu=6, near=[9, 12]))
        self.assertEqual(a, (3, TOP8, 7, []))
        self.assertEqual(b, (3, TOP8, 7, [9, 12]))
        self.assertIsNone(gtr.parse_route("3 1 2"))

    def test_tokens_start_where_the_layer_goes_back(self):
        rows = [gtr.parse_route(route(il, TOP8)) for il in (3, 4, 5, 3, 4, 5, 3)]
        self.assertEqual(gtr.count_tokens(rows), 3)


class ReplayTest(unittest.TestCase):
    def test_the_most_used_experts_stay_resident(self):
        # 8 residents + 3 spares, every route uses the 8 most-used experts: all hits, nothing off the card
        rows = [gtr.parse_route(route(il, TOP8)) for _ in range(20) for il in (3, 4)]
        r = gtr.replay(rows, usage_for((3, 4)), 11, "evict:lfu", [0, 1, 1, 2, 3, 3, 4, 5, 6])
        self.assertEqual((r["hit"], r["live"], r["fetch"], r["cpu"], r["promo"]), (100.0, 100.0, 0, 0, 0))

    def test_the_plan_sends_misses_to_the_cpu_lane(self):
        # no residents (3 slots = 3 spares): with a plan that gives the CPU every miss, nothing is fetched or promoted
        rows = [gtr.parse_route(route(il, list(range(8, 16)), cpu=255)) for _ in range(10) for il in (3, 4)]
        r = gtr.replay(rows, usage_for((3, 4)), 3, "evict:lfu", list(range(9)))
        self.assertEqual(r["toks"], 10)
        self.assertEqual((r["hit"], r["live"], r["fetch"], r["promo"]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(r["cpu"], 16.0)                    # 2 layers x 8 experts a token

    def test_fetched_misses_fill_the_spares(self):
        # 3 residents (experts 0-2) + 3 spares, the CPU takes none: each token promotes 3 of its 8 misses; under LRU
        # the boundary evicts the never-used residents first, so the promoted experts hit in the next token. Under
        # the engine's rule the seeded residents' higher counts keep them, and the newcomers go again.
        rows = [gtr.parse_route(route(3, list(range(8, 16)), fetch=255)) for _ in range(4)]
        lru = gtr.replay(rows, usage_for((3,)), 6, "evict:lru", [0] * 9)
        lfu = gtr.replay(rows, usage_for((3,)), 6, "evict:lfu", [0] * 9)
        self.assertEqual((lru["promo"], lfu["promo"]), (3.0, 3.0))
        self.assertGreater(lru["hit"], 0.0)
        self.assertEqual(lfu["hit"], 0.0)

    def test_the_draft_layer_is_not_in_the_hit_rate(self):
        # trunk layer 44 always hits, the draft layer 45 always misses: the hit rate counts the trunk only
        rows = [gtr.parse_route(route(il, TOP8 if il == 44 else list(range(8, 16)))) for _ in range(5)
                for il in (44, 45)]
        r = gtr.replay(rows, usage_for((44, 45)), 11, "evict:lfu", [0] * 9, nextn=45, keep=2)
        self.assertEqual(r["hit"], 100.0)
        # the draft brings in at most its 2 best-ranked misses a token (some stay resident between tokens)
        self.assertGreater(r["fetch"], 0.0)
        self.assertLessEqual(r["fetch"] + r["cpu"], 2.0)


class MainTest(unittest.TestCase):
    def test_prints_one_line_per_policy(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "routes")
            Path(prefix + ".0").write_text("\n".join(route(il, TOP8) for _ in range(5) for il in (3, 4)) + "\n")
            usage = Path(d) / "expert_usage.txt"
            usage.write_text("# test\n" + "\n".join(
                f"{il} " + " ".join(f"{e}:{(16 - e) * 10}" for e in range(16)) for il in (3, 4)) + "\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                gtr.main([prefix, str(usage), "30", "11", "11", "--policy", "evict:lfu", "--policy", "evict:belady"])
            lines = out.getvalue().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("dev0: hit 100.0 %", lines[0])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gtr.main([prefix, str(usage), "30", "11", "11", "0123"])


if __name__ == "__main__":
    unittest.main()

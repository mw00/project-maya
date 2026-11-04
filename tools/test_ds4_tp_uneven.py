"""CPU-only checks of MAYA_DS4_TP_GGUF_UNEVEN's split (no model or GPU): the engine's own splitb / split128, compiled from
src/ds4/ds4_model.cu - every member whole blocks, the parts covering n_ff once in order, at most a block apart, the odd
blocks going round the members with the rotation, split128 unchanged, and Maya's n_ff 2304 in 256s as 5 + 4."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

MAIN = r"""
#include <cstdio>
#include <cstdlib>
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
int main() {
    // every split: whole blocks, in order, covering n_ff once, at most one block apart; the extra blocks on the
    // members from rot on, cyclically
    const int cases[][2] = {{2304, 256}, {2304, 32}, {2048, 256}, {1536, 256}, {4096, 128}, {2560, 256}};
    for (const auto& c : cases)
        for (int np = 2; np <= 9 && np <= c[0] / c[1]; ++np)
            for (int rot = 0; rot < np; ++rot) {
                const int b = c[0] / c[1], r = b % np;
                int at = 0, lo = 1 << 30, hi = 0;
                for (int p = 0; p < np; ++p) {
                    int off = -1, n = -1;
                    splitb(c[0], c[1], np, p, off, n, rot);
                    EXPECT(off == at, "n_ff %d blk %d np %d rot %d: part %d at %d, not %d", c[0], c[1], np, rot, p, off, at);
                    EXPECT(n > 0 && n % c[1] == 0, "n_ff %d blk %d np %d rot %d: part %d is %d", c[0], c[1], np, rot, p, n);
                    const bool extra = ((p - rot) % np + np) % np < r;
                    EXPECT(n / c[1] == b / np + (extra ? 1 : 0), "np %d rot %d: part %d has %d blocks", np, rot, p, n / c[1]);
                    at += n;
                    lo = n < lo ? n : lo;
                    hi = n > hi ? n : hi;
                }
                EXPECT(at == c[0], "n_ff %d blk %d np %d rot %d: the parts cover %d", c[0], c[1], np, rot, at);
                EXPECT(hi - lo <= c[1], "n_ff %d blk %d np %d rot %d: parts %d and %d", c[0], c[1], np, rot, lo, hi);
            }
    // split128 is splitb in 128-blocks
    for (int np = 2; np <= 8; ++np)
        for (int rot = 0; rot < np; ++rot)
            for (int p = 0; p < np; ++p) {
                int o1, n1, o2, n2;
                split128(4096 + 128 * 3, np, p, o1, n1, rot);
                splitb(4096 + 128 * 3, 128, np, p, o2, n2, rot);
                EXPECT(o1 == o2 && n1 == n2, "split128 np %d rot %d part %d: %d+%d, splitb %d+%d", np, rot, p, o1, n1, o2, n2);
            }
    // Maya's quants at --tp 2: n_ff 2304 = 9 blocks of 256 - 5 + 4, the odd block on member (layer % 2)
    for (int rot = 0; rot < 2; ++rot) {
        int o0, n0, o1, n1;
        splitb(2304, 256, 2, 0, o0, n0, rot);
        splitb(2304, 256, 2, 1, o1, n1, rot);
        EXPECT(o0 == 0 && o1 == n0 && n0 + n1 == 2304, "Maya TP2 rot %d: %d+%d, %d+%d", rot, o0, n0, o1, n1);
        EXPECT((rot == 0 ? n0 : n1) == 1280 && (rot == 0 ? n1 : n0) == 1024, "Maya TP2 rot %d: %d and %d", rot, n0, n1);
    }
    std::printf(fails ? "FAILED (%d)\n" : "ALL OK\n", fails);
    return fails ? 1 : 0;
}
"""


def engine_split_source():
    """splitb and split128 as src/ds4/ds4_model.cu defines them."""
    src = (ROOT / "src/ds4/ds4_model.cu").read_text()
    body = re.search(r"^void splitb\(.*?^}$", src, re.M | re.S)        # (a body of several lines)
    one = re.search(r"^void split128\(.*\}$", src, re.M)                # (one line)
    assert body and one
    return body.group(0) + "\n" + one.group(0) + "\n"


@unittest.skipUnless(shutil.which("c++"), "C++ compiler required")
class TpUneven(unittest.TestCase):
    def test_splitb_and_split128(self):
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d) / "tp-uneven-test.cpp", Path(d) / "tp-uneven-test"
            cpp.write_text(engine_split_source() + MAIN)
            subprocess.run(["c++", "-std=c++17", "-O1", str(cpp), "-o", str(exe)], check=True, timeout=120)
            out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(out.returncode, 0, out.stdout)

    def test_the_settings_are_documented_and_read_both_ways(self):
        src = (ROOT / "src/ds4/ds4_model.cu").read_text()
        for name in ("TP_GGUF_UNEVEN", "TP_CHECK"):
            self.assertIn(f'std::getenv("STRATA_DS4_{name}")', src)
            self.assertIn(f'std::getenv("DS4_{name}")', src)
            self.assertIn(f"`MAYA_DS4_{name}`", (ROOT / "docs/SETTINGS.md").read_text())


if __name__ == "__main__":
    unittest.main()

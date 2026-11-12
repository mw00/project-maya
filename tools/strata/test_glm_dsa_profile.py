#!/usr/bin/env python3
"""tools/strata/test_glm_dsa_profile.py - glm-dsa W10: Maya v1.1.3's expert profile and drafting choice on a glm-dsa
pack (tools/strata/maya_sync.py, tools/strata/glm_dsa_profile.py, maya.py's hooks).  CPU only, no model files.

    python3 tools/strata/test_glm_dsa_profile.py

With MAYA_SYNC_BIN=<build>/glm_maya_sync_test (ctest sets it) the engine's own reader (read_expert_profile)
reads the profile this tool writes, at the glm-dsa layout (the older name STRATA_MAYA_SYNC_BIN still counts).
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import glm_dsa_profile as GP   # noqa: E402
import make_profile as MP      # noqa: E402
import maya_sync               # noqa: E402

TRUNK, LEAD, NE = 78, 3, 256


def dsa_pack(d: Path, trunk=TRUNK, lead=LEAD, ne=NE) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.txt").write_text("# strata index v3\n# rows\n# glm-dsa: trunk %d, dense lead %d, experts %d, mtp %d, "
                                 "indexer full 0,1,2\nrow 1 2 3\n" % (trunk, lead, ne, trunk), encoding="utf-8")
    return d


def flash_pack(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.txt").write_text("# strata index v3\n# rows\nrow 1 2 3\n", encoding="utf-8")
    return d


def header(p: Path):
    b = p.read_bytes()
    return b[:4], struct.unpack_from("<5I", b, 4)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shipped = self.root / "expert-profile-glm.bin"
        MP.write_profile(self.shipped, MP.rank_profile([], {}, 288, no_base=True, n_layer=46)[0], 288, 46)

    def test_pack_layout(self):
        self.assertEqual(maya_sync.pack_layout(dsa_pack(self.root / "dsa")), (79, 256))
        self.assertIsNone(maya_sync.pack_layout(flash_pack(self.root / "flash")))
        self.assertIsNone(maya_sync.pack_layout(self.root / "none"))

    def test_profile_for_flash_is_mayas(self):
        pack = flash_pack(self.root / "flash")
        self.assertEqual(maya_sync.profile_for(pack, self.shipped), self.shipped)
        self.assertIsNone(maya_sync.profile_for(pack, self.root / "absent.bin"))

    def test_profile_for_glm_dsa(self):
        pack = dsa_pack(self.root / "dsa")
        self.assertIsNone(maya_sync.profile_for(pack, self.shipped), "never the Flash profile on glm-dsa")
        # a Flash-layout file under the pack's name does not fit either
        (pack / maya_sync.PACK_PROFILE).write_bytes(self.shipped.read_bytes())
        self.assertIsNone(maya_sync.profile_for(pack, self.shipped))
        (pack / "expert_counts.txt").write_text("".join("%d %s\n" % (l, " ".join(["1"] * NE))
                                                        for l in range(LEAD, TRUNK + 1)), encoding="utf-8")
        self.assertEqual(GP.main([str(pack)]), 0)
        self.assertEqual(maya_sync.profile_for(pack, self.shipped), pack / maya_sync.PACK_PROFILE)

    def test_profile_for_glm_dsa_falls_back_to_glm53s_shipped_one(self):
        # S7: a glm-dsa pack without a profile of its own takes GLM-5.3's shipped one (data/expert-profile-glm53.bin)
        # when its layout fits; the pack's own still wins, and Flash's never applies
        pack = dsa_pack(self.root / "dsa53")
        g53 = self.root / "expert-profile-glm53.bin"
        MP.write_profile(g53, MP.rank_profile([], {}, NE, no_base=True, n_layer=TRUNK + 1)[0], NE, TRUNK + 1)
        self.assertEqual(maya_sync.profile_for(pack, self.shipped, g53), g53)
        self.assertIsNone(maya_sync.profile_for(pack, self.shipped, self.shipped), "a Flash-layout file never fits")
        self.assertIsNone(maya_sync.profile_for(pack, self.shipped, self.root / "absent.bin"))
        (pack / "expert_counts.txt").write_text("".join("%d %s\n" % (l, " ".join(["1"] * NE))
                                                        for l in range(LEAD, TRUNK + 1)), encoding="utf-8")
        self.assertEqual(GP.main([str(pack)]), 0)
        self.assertEqual(maya_sync.profile_for(pack, self.shipped, g53), pack / maya_sync.PACK_PROFILE)
        flash = flash_pack(self.root / "flash53")
        self.assertEqual(maya_sync.profile_for(flash, self.shipped, g53), self.shipped)

    def test_draft_for(self):
        said = []
        self.assertEqual(maya_sync.draft_for("dflash2", "glm-dsa", said.append), "mtp")
        self.assertTrue(said and "NextN" in said[0])
        self.assertEqual(maya_sync.draft_for("dflash2", "glm5next", said.append), "dflash2")
        self.assertEqual(maya_sync.draft_for("mtp", "glm-dsa", said.append), "mtp")


class ProfileToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pack = dsa_pack(Path(self.tmp.name) / "pack")

    def write_trace(self, path: Path, routes):
        with open(path, "wb") as f:
            for layer, ids in routes:
                f.write(struct.pack("<ii", layer, len(ids)))
                f.write(struct.pack("<%di" % len(ids), *ids))
                f.write(struct.pack("<%df" % len(ids), *([0.125] * len(ids))))
        return path

    def test_from_inputs(self):
        trace = self.write_trace(Path(self.tmp.name) / "routes.bin",
                                 [(78, [17, 18]), (78, [17]), (40, [9]), (78, [17]), (2, [5]), (90, [1])])
        usage = Path(self.tmp.name) / "usage.txt"
        usage.write_text("10 4:1 200:1\n", encoding="utf-8")
        out = Path(self.tmp.name) / "p.bin"
        self.assertEqual(GP.main([str(self.pack), str(trace), str(usage), "--out", str(out)]), 0)
        magic, (ver, nl, ne, slots, n) = header(out)
        self.assertEqual((magic, ver, nl, ne, slots, n), (b"STRP", 1, 79, 256, 79 * 256, 79 * 256))
        ranked = MP.read_profile(out, 256, 79)
        self.assertEqual(ranked[0], (78, 17), "the NextN block's most routed expert first")
        self.assertEqual(len(set(ranked)), 79 * 256, "every pair once")
        lead_pos = [i for i, p in enumerate(ranked) if p[0] < LEAD]
        self.assertEqual(min(lead_pos), 79 * 256 - LEAD * 256, "the dense lead's pairs last")
        self.assertNotIn((90, 1), ranked, "a layer past the layout is dropped")
        bin_ = os.environ.get("MAYA_SYNC_BIN") or os.environ.get("STRATA_MAYA_SYNC_BIN")
        if bin_:
            r = subprocess.run([bin_, str(out), "79", "256"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            r = subprocess.run([bin_, str(out), "46", "288"], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, "the engine refuses it at Flash's layout")

    def test_counts_default_and_base(self):
        (self.pack / "expert_counts.txt").write_text("5 " + " ".join(str(e) for e in range(NE)) + "\n",
                                                     encoding="utf-8")
        self.assertEqual(GP.main([str(self.pack)]), 0)
        first = self.pack / maya_sync.PACK_PROFILE
        self.assertEqual(MP.read_profile(first, NE, 79)[0], (5, 255), "the pack's expert_counts.txt by default")
        # a base of the same layout fills after the inputs
        out = Path(self.tmp.name) / "q.bin"
        trace = self.write_trace(Path(self.tmp.name) / "t.bin", [(60, [3])])
        self.assertEqual(GP.main([str(self.pack), str(trace), "--base", str(first), "--out", str(out)]), 0)
        q = MP.read_profile(out, NE, 79)
        self.assertEqual(q[:2], [(60, 3), (5, 255)])

    def test_refusals(self):
        flash = flash_pack(Path(self.tmp.name) / "flash")
        self.assertEqual(GP.main([str(flash)]), 2, "not a glm-dsa pack")
        self.assertEqual(GP.main([str(self.pack)]), 2, "no inputs, no expert_counts.txt")


class MayaHookTests(unittest.TestCase):
    """maya.py's add_profile on a glm-dsa config: never the shipped Flash profile, the pack's own when it fits."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        sys.path.insert(0, str(ROOT))
        import maya
        self.maya = maya
        self.shipped = Path(self.tmp.name) / "expert-profile-glm.bin"
        MP.write_profile(self.shipped, MP.rank_profile([], {}, 288, no_base=True, n_layer=46)[0], 288, 46)
        from unittest.mock import patch
        p = patch.object(maya, "PROFILE", self.shipped)
        p.start()
        self.addCleanup(p.stop)

    def test_add_profile(self):
        flash = flash_pack(Path(self.tmp.name) / "flash")
        cfg = {"args": ["--glm-pack", str(flash)]}
        self.assertTrue(self.maya.add_profile(cfg))
        self.assertEqual(cfg["args"][-1], str(self.shipped))
        from unittest.mock import patch
        g53 = Path(self.tmp.name) / "expert-profile-glm53.bin"   # GLM-5.3's shipped profile (S7): absent first
        with patch.object(self.maya.G53, "PROFILE", g53):
            dsa = dsa_pack(Path(self.tmp.name) / "dsa")
            cfg = {"args": ["--glm-pack", str(dsa)]}
            self.assertFalse(self.maya.add_profile(cfg), "no Flash profile on a glm-dsa config")
            MP.write_profile(g53, MP.rank_profile([], {}, NE, no_base=True, n_layer=TRUNK + 1)[0], NE, TRUNK + 1)
            self.assertTrue(self.maya.add_profile(cfg))                   # GLM-5.3's shipped one, never Flash's
            self.assertEqual(cfg["args"][-1], str(g53))
            cfg = {"args": ["--glm-pack", str(dsa)]}
            (dsa / "expert_counts.txt").write_text("4 1 2 3\n", encoding="utf-8")
            self.assertEqual(GP.main([str(dsa)]), 0)
            self.assertTrue(self.maya.add_profile(cfg))                   # the pack's own wins
            self.assertEqual(cfg["args"][-1], str(dsa / maya_sync.PACK_PROFILE))


if __name__ == "__main__":
    unittest.main()

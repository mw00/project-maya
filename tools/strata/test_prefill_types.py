#!/usr/bin/env python3
"""The batched glm-dsa prompt's FP16 types against the pack and tools/strata/type_coverage.py (W7 build review).

Every non-routed type the pack can leave native (tools/strata/glm_dsa_pack.py plan_tensor, with and without
STRATA_PACK_NATIVE32) must be one the prompt serves: F32 (SGEMM), BF16 (narrowed) or an FP16 dequant type
(src/core/glm_prefill_dsa.cu f16_covers + src/core/glm_prefill_types.hpp kExtF16Types). The lists in the C++ sources
must agree with each other and with type_coverage.py's engine table, and refusal()'s "covered" text must name exactly
the served types.
"""
from __future__ import annotations

import pathlib
import re
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gguf_audit as ga  # noqa: E402

try:
    import glm_dsa_pack as gp  # noqa: E402  (imports numpy)
    import type_coverage as tc  # noqa: E402
except ImportError as e:  # pragma: no cover
    print(f"test_prefill_types: skipped ({e})")
    sys.exit(77 if __name__ == "__main__" else 0)

TYPES_HPP = (ROOT / "src/core/glm_prefill_types.hpp").read_text()
PF_CU = (ROOT / "src/core/glm_prefill_dsa.cu").read_text()
DEQ_CU = (ROOT / "src/kernels/cuda/dequant_bf16.cu").read_text()


def int_array(text: str, name: str) -> set[int]:
    m = re.search(name + r"\[\]\s*=\s*\{([^}]*)\}", text)
    assert m, name
    return {int(x) for x in re.findall(r"\d+", m.group(1))}


def pack_native_types() -> set[str]:
    """Every type plan_tensor serves natively for some non-routed tensor (either native32 setting)."""
    names = [f"blk.0.{n}" for n in gp.GLMDSA_NATIVE_OK + gp.GLMDSA_BF16 + gp.GLMDSA_F32] + list(
        gp.GLMDSA_NATIVE_OK_GLOBAL)
    out = set()
    for n in names:
        if n == "token_embd.weight":
            continue            # gathered on the host, never a prompt GEMM
        for ty in ga.TYPE_ID:
            for n32 in (False, True):
                what, target = gp.plan_tensor(n, ty, n32)
                if what == "native":
                    out.add(ty)
                elif what in ("copy", "convert"):
                    out.add(target)
    return out


class PrefillTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tab = tc.engine_tables(ROOT, None)
        cls.ext = int_array(TYPES_HPP, "kExtF16Types")
        cls.maya_iq = int_array(TYPES_HPP, "kMayaIqF16Types")

    def test_every_pack_native_type_is_served(self):
        served = self.tab["f16"] | {0, 30}
        missing = sorted(t for t in pack_native_types() if ga.TYPE_ID[t] not in served)
        self.assertEqual(missing, [], "pack-native types the batched glm-dsa prompt cannot dequantize")

    def test_type_coverage_reads_the_extra_types(self):
        self.assertTrue(self.ext <= self.tab["f16"], "type_coverage.py must read kExtF16Types")
        self.assertEqual(self.ext, {1, 3, 7, 10, 19})

    def test_c_lists_agree(self):
        body = tc._body(PF_CU, "bool f16_covers(int type")
        self.assertEqual({int(x) for x in re.findall(r"type == (\d+)", body)}, self.maya_iq)
        iq_only = tc._body(DEQ_CU, "bool iq_only(int t)") or re.search(r"bool iq_only\(int t\) \{([^}]*)\}", DEQ_CU).group(1)
        self.assertEqual({int(x) for x in re.findall(r"t == (\d+)", iq_only)}, self.maya_iq)
        geo = set(int(x) for x in re.findall(r"case (\d+):", tc._body(TYPES_HPP, "bool ext_geometry(int t")))
        self.assertEqual(geo, self.ext)

    def test_flash_keeps_mayas_set(self):
        # Maya v1.1.3's prefill_setup deq_ok: BF16, EXL3 and dequant_bf16_supported (no i-quant). f16_covers must
        # admit nothing beyond that before its `if (!dsa) return false;` line
        body = tc._body(PF_CU, "bool f16_covers(int type")
        head, sep, _ = body.partition("if (!dsa) return false;")
        self.assertTrue(sep, "f16_covers has no Flash cut-off")
        self.assertEqual(re.findall(r"type == (\d+)", head), [], "Flash admits i-quants Maya's prompt path refuses")
        self.assertIn("dequant_bf16_supported(type)", head)
        hd = tc._body(TYPES_HPP, "bool ext_f16_type(int t)")
        self.assertEqual({int(x) for x in re.findall(r"t == (\d+)", hd)}, self.ext)
        self.assertFalse(self.ext & self.tab["f16"] - self.ext & {int(x) for x in re.findall(
            r"case (\d+):", tc._body(DEQ_CU, "bool geometry(int type, int& block_elems"))},
            "an extra type must not shadow a Maya dequant_f16 type")

    def test_refusal_text_names_the_served_types(self):
        m = re.search(r"\(covered: ([^)]*)\)", TYPES_HPP.replace('"\n           "', "").replace('" "', ""))
        self.assertIsNotNone(m)
        text = re.sub(r"\s+", " ", m.group(1))
        named = set()
        for part in [p.strip() for p in text.split(",")]:
            if ".." in part:
                lo, hi = part.split("..")
                named |= {f"Q{i}_K" for i in range(int(lo[1]), int(hi[1]) + 1)}
            else:
                named.add(part)
        served = {ga.type_name(t) for t in self.tab["f16"] | {0, 30} if t != 42}   # 42: Maya-internal Q2_0
        self.assertEqual(named, served)


if __name__ == "__main__":
    unittest.main()

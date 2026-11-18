"""tools/k31/mem_budget.py on a made-up machine (a /proc and /sys tree and a stand-in nvidia-smi): B_pool, the fixed
costs and where each goes against the K3.1 plan's figures, --ctx and --checkpoints, E_gb and its VRAM / RAM split,
the candidate verdicts, the ARC cap, the big jobs and what they started, their swap in zram and on disk, their GPU
memory, --gpus, --reserve, unit names, and a machine without nvidia-smi.

    python -m unittest tools.test_k31_mem_budget
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("k31_mem_budget", HERE / "k31" / "mem_budget.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

GiB, MiB, GB = 1 << 30, 1 << 20, 10 ** 9

SMI = r'''#!PY
import sys
a = " ".join(sys.argv[1:])
if "--query-gpu" in a:
    print("0, GPU-aaaa, Card A, 24576, 1000, 23000")
    print("1, GPU-bbbb, Card B, 12288, 6000, 6000")
elif "--query-compute-apps" in a:
    print("200, GPU-bbbb, 4000")
    print("300, GPU-aaaa, 900")
'''


def machine(tmp: pathlib.Path, smi: bool = True, arc: tuple = (6, 2), disk_swap: int = 0) -> tuple[pathlib.Path, str]:
    """arc: the ARC's size and floor, GiB; disk_swap: GiB used on a swap file next to the zram."""
    root = tmp / "root"
    proc = root / "proc"
    (proc / "spl/kstat/zfs").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemTotal: {64 * GiB >> 10} kB\nMemFree: 1024 kB\n"
                                  f"MemAvailable: {40 * GiB >> 10} kB\nSwapTotal: {16 * GiB >> 10} kB\n")
    (proc / "spl/kstat/zfs/arcstats").write_text(f"13 1 0x01\nname type data\nsize 4 {arc[0] * GiB}\n"
                                                 f"c_min 4 {arc[1] * GiB}\n")
    # 4 GiB swapped to zram, which holds it in 1 GiB
    (proc / "swaps").write_text("Filename Type Size Used Priority\n"
                                f"/dev/zram0 partition {8 * GiB >> 10} {4 * GiB >> 10} 100\n"
                                + (f"/swap.img file {8 * GiB >> 10} {disk_swap * GiB >> 10} -2\n" if disk_swap else ""))
    (root / "sys/block/zram0").mkdir(parents=True)
    (root / "sys/block/zram0/mm_stat").write_text(f"{4 * GiB} {GiB - 4096} {GiB} 0 {GiB} 0 0 0 0\n")
    work = tmp / "work"
    work.mkdir()
    # pid: (ppid, name, cmdline, cwd, cgroup, RssAnon MiB, Pss_Anon MiB or None, VmSwap MiB)
    rows = {
        100: (1, "bash", "bash /x/chain.sh", "/", "/user.slice/session-c1.scope", 5, None, 0),
        101: (100, "quantize", "llama-quantize in.gguf out.gguf", "/", "/user.slice/session-c1.scope", 3000, 2048, 1024),
        200: (1, "python", "python lab.py", str(work), "/user.slice/session-c1.scope", 1024, None, 0),
        300: (1, "server", "server -m chat.gguf", "/", "/system.slice/chat.service", 2048, None, 512),
        400: (1, "kworker", "", "/", "/", 0, None, 0),   # a kernel thread: no VmRSS
    }
    for pid, (pp, name, cmd, cwd, cg, anon, pss, swap) in rows.items():
        d = proc / str(pid)
        d.mkdir()
        st = f"Name:\t{name}\nPPid:\t{pp}\n"
        if name != "kworker":
            st += f"VmRSS:\t{(anon + 10) * 1024} kB\nRssAnon:\t{anon * 1024} kB\nVmSwap:\t{swap * 1024} kB\n"
        (d / "status").write_text(st)
        (d / "cmdline").write_text(cmd.replace(" ", "\0"))
        (d / "cgroup").write_text(f"0::{cg}\n")
        (d / "cwd").symlink_to(cwd)
        if pss is not None:
            (d / "smaps_rollup").write_text(f"Rss: {anon * 1024} kB\nPss_Anon: {pss * 1024} kB\n")
    exe = tmp / "nvidia-smi"
    exe.write_text(SMI.replace("PY", sys.executable) if smi else "#!/bin/sh\nexit 9\n")
    exe.chmod(0o755)
    return root, str(exe)


def run(tmp: pathlib.Path, *args: str, smi: bool = True, table: list | None = None, **kw) -> dict:
    root, exe = machine(tmp, smi, **kw)
    out = tmp / "b.json"
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        rc = mb.main(["--root", str(root), "--nvidia-smi", exe, "--json", str(out), *args])
    assert rc == 0
    if table is not None:
        table.extend(text.getvalue().splitlines())
    return json.loads(out.read_text())


def cost(d: dict, start: str) -> dict:
    (f,) = [f for f in d["fixed"] if f["cost"].startswith(start)]
    return f


class Budget(unittest.TestCase):
    def test_fixed_costs_match_the_plan(self):
        on = sum(b for _, b, _ in mb.ALWAYS_ON)
        self.assertTrue(47.7 * GB < on < 47.95 * GB, on)          # the plan's ~47.8 GB
        self.assertEqual(mb.TOKEN_EMBD, 1_174_405_120 * 210 // 256)
        self.assertAlmostEqual(mb.KV_PER_TOKEN, 33_700, delta=100)          # §3.3: 27.6 + 6.1 KB a token
        self.assertAlmostEqual(mb.KDA_STATE, 464e6, delta=1e6)             # §3.3: 434 MB + ~30 MB conv
        self.assertAlmostEqual(mb.KDA_CHECKPOINT, 464e6, delta=1e6)        # the plan's 464 MB a checkpoint
        fixed = mb.fixed_costs(4, 65536, 2)
        self.assertEqual([(b, w) for _, b, w in fixed],
                         [(on - mb.TOKEN_EMBD, "vram"), (mb.TOKEN_EMBD, "ram"), (4 * GB, "vram"),
                          (65536 * mb.KV_PER_TOKEN, "vram"), (mb.KDA_STATE, "vram"), (2 * mb.KDA_CHECKPOINT, "ram"),
                          (6 * GB, "ram")])
        total = sum(b for _, b, _ in fixed)
        self.assertTrue(61.4 * GB < total < 61.55 * GB, total)    # the plan's ~61 GB; 61.47 with these parts
        # docs/KIMI-K31.md §3.3: MB an expert
        self.assertEqual({k: round(mb.qbytes(mb.EXPERT_PARAMS, k) / 1e6, 2) for k in ("IQ1_S", "IQ1_M", "IQ2_XXS")},
                         {"IQ1_S": 6.45, "IQ1_M": 7.23, "IQ2_XXS": 8.52})
        self.assertEqual(mb.qbytes(256, "Q6_K"), 210)
        self.assertEqual(mb.qbytes(32, "Q8_0"), 34)

    def test_ctx_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as t:
            d = run(pathlib.Path(t), "--ctx", "131072", "--checkpoints", "5")
        self.assertEqual(d["ctx"], 131072)
        self.assertEqual(cost(d, "MLA latent")["bytes"], 131072 * mb.KV_PER_TOKEN)
        self.assertEqual(cost(d, "KDA checkpoints")["bytes"], 5 * mb.KDA_CHECKPOINT)
        self.assertEqual(d["now"]["fixed"], sum(f["bytes"] for f in d["fixed"]))

    def test_where_the_experts_sit(self):
        fixed = [("dense", 51 * GB, "vram"), ("host", 10 * GB, "ram")]
        b = mb.budget(50 * GB, {0: 60 * GB, 1: 40 * GB}, fixed)        # VRAM left after the dense part
        self.assertEqual((b["b_pool"], b["e"], b["experts_in_vram"], b["experts_in_ram"]),
                         (150 * GB, 89 * GB, 49 * GB, 40 * GB))
        b = mb.budget(50 * GB, {0: 30 * GB, 1: 11 * GB}, fixed)        # the dense part spills 10 GB into RAM
        self.assertEqual((b["b_pool"], b["e"], b["experts_in_vram"], b["experts_in_ram"]),
                         (91 * GB, 30 * GB, 0, 30 * GB))

    def test_pool_and_experts(self):
        with tempfile.TemporaryDirectory() as t:
            d = run(pathlib.Path(t))
        ram = 40 * GiB + 4 * GiB                                   # MemAvailable + ARC above its floor
        self.assertEqual(d["ram"]["ram_avail"], ram)
        self.assertEqual(d["now"]["b_pool"], ram + (23000 + 6000) * MiB)
        fixed = sum(f["bytes"] for f in d["fixed"])
        self.assertEqual(d["now"]["e"], d["now"]["b_pool"] - fixed)
        self.assertEqual([g["index"] for g in d["gpus"]], [0, 1])
        self.assertEqual(d["now"]["experts_a_layer"]["IQ1_S"], max(0, d["now"]["e"]) // (6_451_200 * 92))
        # 29000 MiB of VRAM is less than the 51.6 GB the GPUs' fixed costs take: every expert sits in RAM
        self.assertEqual((d["now"]["experts_in_vram"], d["now"]["experts_in_ram"]), (0, d["now"]["e"]))
        self.assertEqual(d["jobs"], [])
        self.assertEqual(d["services_only"]["b_pool"], d["now"]["b_pool"])

    def test_candidates(self):
        lines = []
        with tempfile.TemporaryDirectory() as t:                  # E_gb: 18.18 GB now, 25.87 services only
            d = run(pathlib.Path(t), "--job", "chain\\.sh", "--job", "/work$", "--candidates", "15,20,30", table=lines)
        self.assertEqual(d["candidates_gb"], {"15": {"now": True, "services_only": True},
                                              "20": {"now": False, "services_only": True},
                                              "30": {"now": False, "services_only": False}})
        (row,) = [x for x in lines if "candidate E_gb = 20" in x]
        self.assertEqual(row.split()[-3:], ["short", "1.8", "fits"])

    def test_arc_is_capped_at_what_the_kernel_holds_back(self):
        with tempfile.TemporaryDirectory() as t:
            root, _ = machine(pathlib.Path(t), arc=(30, 2))
            r = mb.ram(root)
        self.assertEqual(r["zfs_arc_reclaimable"], 24 * GiB)       # 28 GiB above the floor, but total - available is 24
        self.assertEqual(r["ram_avail"], 64 * GiB)

    def test_jobs_and_what_they_started(self):
        with tempfile.TemporaryDirectory() as t:
            d = run(pathlib.Path(t), "--job", "chain\\.sh", "--job", "/work$")
        self.assertEqual(sorted(j["pid"] for j in d["jobs"]), [100, 101, 200])   # 101 by its parent, 200 by its cwd
        anon = (5 + 2048 + 1024) * MiB                              # Pss_Anon where there is one
        zram = int(1024 * MiB * (GiB / (4 * GiB)))                   # 101's swap at zram's 4:1
        self.assertEqual(d["jobs_freed"]["ram_anon"], anon)
        self.assertEqual(d["jobs_freed"]["ram_zram"], zram)
        self.assertEqual(d["services_only"]["ram_avail"], d["now"]["ram_avail"] + anon + zram)
        g = {x["index"]: x for x in d["gpus"]}
        self.assertEqual(g[1]["free_services_only"], (6000 + 4000) * MiB)   # pid 200's GPU memory
        self.assertEqual(g[0]["free_services_only"], 23000 * MiB)           # pid 300 is a service
        units = {u["unit"]: u for u in d["services"]}
        self.assertEqual(set(units), {"chat.service"})
        self.assertEqual(d["services_swap"], 512 * MiB)
        self.assertEqual(d["services_swap_back"], 512 * MiB - 128 * MiB)

    def test_swap_on_disk_holds_no_ram(self):
        with tempfile.TemporaryDirectory() as t:                  # 4 GiB in zram (4:1) and 4 GiB on a swap file
            d = run(pathlib.Path(t), "--job", "chain\\.sh", disk_swap=4)
        self.assertEqual((d["ram"]["swap_used"], d["ram"]["zram_swap_used"]), (8 * GiB, 4 * GiB))
        self.assertEqual(d["jobs_freed"]["ram_zram"], 1024 * MiB // 2 // 4)          # half of it in zram, at 4:1
        self.assertEqual(d["services_swap_back"], 512 * MiB - 512 * MiB // 2 // 4)

    def test_reserve(self):
        with tempfile.TemporaryDirectory() as t:
            base = run(pathlib.Path(t))
        with tempfile.TemporaryDirectory() as t:
            d = run(pathlib.Path(t), "--reserve", "ram=2", "--reserve", "gpu1=1.5", "--reserve", "ram=0.5")
        self.assertEqual(d["reserve"], {"ram": int(2.5 * GB), "vram": {"1": int(1.5 * GB)}})
        for k in ("now", "services_only"):
            self.assertEqual(d[k]["ram_avail"], base[k]["ram_avail"] - int(2.5 * GB))
            self.assertEqual(d[k]["vram_free"], base[k]["vram_free"] - int(1.5 * GB))
            self.assertEqual(d[k]["e"], base[k]["e"] - 4 * GB)
        self.assertEqual([g["free"] for g in d["gpus"]], [23000 * MiB, 6000 * MiB])    # nvidia-smi's own figure
        with tempfile.TemporaryDirectory() as t:
            d = run(pathlib.Path(t), "--gpus", "0", "--reserve", "gpu1=1.5")       # a GPU that does not count
        self.assertEqual(d["now"]["vram_free"], 23000 * MiB)
        for bad in ("gpu5=1", "disk=3", "ram=-1", "ram"):
            with tempfile.TemporaryDirectory() as t:
                root, exe = machine(pathlib.Path(t))
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(mb.main(["--root", str(root), "--nvidia-smi", exe, "--reserve", bad]), 2, bad)

    def test_unit_names(self):
        self.assertEqual(mb.unit_of("0::/system.slice/chat.service\n"), "chat.service")
        self.assertEqual(mb.unit_of("0::/user.slice/user-1000.slice/user@1000.service/app.slice/lab.service\n"),
                         "lab.service")
        self.assertEqual(mb.unit_of("0::/system.slice/docker-" + "0123456789ab" * 5 + "cdef.scope\n"),
                         "docker-0123456789ab")
        self.assertEqual(mb.unit_of("0::/\n"), "-")

    def test_gpus_and_no_smi(self):
        with tempfile.TemporaryDirectory() as t:
            d = run(pathlib.Path(t), "--gpus", "1")
        self.assertEqual(d["gpus_counted"], [1])
        self.assertEqual(d["now"]["vram_free"], 6000 * MiB)
        self.assertIn("(1 GPUs)", " ".join(f["cost"] for f in d["fixed"]))
        with tempfile.TemporaryDirectory() as t:
            d = run(pathlib.Path(t), smi=False)
        self.assertEqual(d["gpus"], [])
        self.assertEqual(d["now"]["b_pool"], d["ram"]["ram_avail"])
        with tempfile.TemporaryDirectory() as t:
            root, exe = machine(pathlib.Path(t))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(mb.main(["--root", str(root), "--nvidia-smi", exe, "--gpus", "5"]), 2)

    def test_a_test_tree_never_holds_the_caller(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(mb.os, "getpid", return_value=101):
            d = run(pathlib.Path(t), "--job", "chain\\.sh")             # our real pid is the tree's quantize pid
        self.assertEqual(sorted(j["pid"] for j in d["jobs"]), [100, 101])

    def test_the_caller_never_matches(self):
        ps = {10: {"ppid": 1, "cmd": "bash -c mem_budget.py --job chain", "cwd": "/"},
              11: {"ppid": 10, "cmd": "python mem_budget.py --job chain", "cwd": "/"},
              12: {"ppid": 1, "cmd": "bash chain.sh", "cwd": "/"}}
        self.assertEqual(mb.jobs_of(ps, ["chain"], mb.ancestors(ps, 11)), {12})


if __name__ == "__main__":
    unittest.main()

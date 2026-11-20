"""Synthetic sysfs/affinity tests: no model, hardware probes or affinity changes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import glm_pool_cpus as tool
from tools import maya_names


class PoolCpus(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def topology(self, nodes, online=None):
        """node ID -> physical cores -> each core's logical CPU IDs."""
        all_cpus = set()
        for node, cores in nodes.items():
            node_cpus = {cpu for core in cores for cpu in core}
            all_cpus.update(node_cpus)
            path = self.root / f"devices/system/node/node{node}/cpulist"
            path.parent.mkdir(parents=True)
            path.write_text(",".join(map(str, sorted(node_cpus))))
            for core in cores:
                for cpu in core:
                    path = self.root / f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
                    path.parent.mkdir(parents=True)
                    path.write_text(",".join(map(str, core)))
        path = self.root / "devices/system/cpu/online"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(",".join(map(str, sorted(all_cpus if online is None else online))))
        return all_cpus

    def test_core_first_interleaves_noncontiguous_nodes_and_cpu_ids(self):
        allowed = self.topology({2: [[10, 50], [12, 52]], 7: [[21, 61], [23, 63]]})
        self.assertEqual(tool.pool_cpus(self.root, allowed, 7), [10, 21, 12, 23, 50, 61, 52])

    def test_unequal_three_nodes_finish_all_cores_before_smt(self):
        allowed = self.topology({0: [[0, 10], [2, 12]], 3: [[4]], 8: [[6, 16, 26, 36]]})
        self.assertEqual(tool.pool_cpus(self.root, allowed, 9), [0, 4, 6, 2, 10, 16, 12, 26, 36])

    def test_partial_cpuset_and_offline_cpus(self):
        self.topology({0: [[0, 8], [2, 10]], 1: [[1, 9], [3, 11]]}, online={0, 1, 2, 3, 8, 9, 10})
        # Only a sibling of core 0 is allowed; CPU 11 is allowed but offline.
        allowed = {8, 2, 10, 9, 11}
        self.assertEqual(tool.pool_cpus(self.root, allowed, 4), [2, 9, 8, 10])
        with self.assertRaisesRegex(ValueError, "only 4 allowed online CPUs"):
            tool.pool_cpus(self.root, allowed, 5)

    def test_single_node_without_smt_and_one_thread(self):
        allowed = self.topology({0: [[3], [17], [29]]})
        self.assertEqual(tool.pool_cpus(self.root, allowed, 1), [3])
        self.assertEqual(tool.pool_cpus(self.root, allowed, 3), [3, 17, 29])

    def test_rejects_insufficient_or_nonpositive_count(self):
        allowed = self.topology({0: [[0, 1]]})
        for count in (0, -1, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                tool.pool_cpus(self.root, allowed, count)

    def test_cpu_list_format(self):
        self.assertEqual(tool.cpu_list(" 0-2,5,9-10\n"), {0, 1, 2, 5, 9, 10})
        self.assertEqual(tool.cpu_list("\n"), set())
        for bad in ("-1", "2-0", "0,", "0-4:2", "abc", "1,,2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tool.cpu_list(bad)

    def test_missing_node_or_core_topology_is_an_error(self):
        allowed = self.topology({0: [[0, 4]]})
        node = self.root / "devices/system/node/node0/cpulist"
        node.write_text("0")
        with self.assertRaisesRegex(ValueError, "NUMA node information is missing"):
            tool.pool_cpus(self.root, allowed, 2)
        node.write_text("0,4")
        (self.root / "devices/system/cpu/cpu4/topology/thread_siblings_list").unlink()
        with self.assertRaises(OSError):
            tool.pool_cpus(self.root, allowed, 2)

    def test_inconsistent_sibling_information_is_rejected(self):
        allowed = self.topology({0: [[0, 4]]})
        (self.root / "devices/system/cpu/cpu4/topology/thread_siblings_list").write_text("4")
        with self.assertRaisesRegex(ValueError, "inconsistent SMT/NUMA topology"):
            tool.pool_cpus(self.root, allowed, 2)

    def test_unrepresentable_engine_cpu_ids_are_rejected(self):
        allowed = self.topology({0: [[0], [1024]]})
        with self.assertRaisesRegex(ValueError, "IDs >= 1024"):
            tool.pool_cpus(self.root, allowed, 1)

    def test_empty_mask_and_empty_online_intersection(self):
        self.topology({0: [[0]]}, online={0})
        for allowed in (set(), {1}):
            with self.subTest(allowed=allowed), self.assertRaisesRegex(ValueError, "no allowed online CPUs"):
                tool.pool_cpus(self.root, allowed, 1)

    def test_cli_requires_explicit_positive_count_and_linux(self):
        for argv in ([], ["--threads", "0"], ["--threads", "-1"], ["--threads", "auto"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
                tool.main(argv)
            self.assertEqual(exit.exception.code, 2)
        with mock.patch.object(tool.sys, "platform", "win32"), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as exit:
            tool.main(["--threads", "4"])
        self.assertEqual(exit.exception.code, 2)

    def test_cli_prints_only_two_settings_and_honors_machine_affinity(self):
        allowed = self.topology({0: [[0, 8]], 1: [[2, 10]]})
        real_pool_cpus = tool.pool_cpus
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(tool.sys, "platform", "linux"), \
                mock.patch.object(tool.os, "sched_getaffinity", return_value=allowed, create=True) as affinity, \
                mock.patch.object(tool, "pool_cpus", side_effect=lambda root, cpus, n: real_pool_cpus(self.root, cpus, n)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(tool.main(["--threads", "3"]), 0)
        affinity.assert_called_once_with(0)
        cfg = json.loads(out.getvalue())
        self.assertEqual(cfg, {"env": {"MAYA_GLM_CPU_LANE": "3", "MAYA_GLM_POOL_CPUS": "0,2,8"}})
        exported = {}
        maya_names.export_env(exported, cfg["env"])
        self.assertEqual(exported["STRATA_GLM_CPU_LANE"], "3")
        self.assertEqual(exported["STRATA_GLM_POOL_CPUS"], "0,2,8")
        self.assertIn("2 background workers", err.getvalue())
        self.assertIn("not a recommended thread count", err.getvalue())

    def test_cli_read_error_emits_no_config(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(tool.sys, "platform", "linux"), \
                mock.patch.object(tool.os, "sched_getaffinity", side_effect=OSError("affinity unavailable"), create=True), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as exit:
            tool.main(["--threads", "3"])
        self.assertEqual(exit.exception.code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("affinity unavailable", err.getvalue())


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from serve.response_config import limits_from_env
from serve.response_store import ResponseStore, StoreBusy
from serve.server import apply_server_env
from tools import maya_names


def item(index, content="x" * 2048):
    return {"id": f"msg_{index}", "type": "message", "role": "user", "content": content}


class SharedHistory(unittest.TestCase):
    def test_shared_blocks_survive_ancestor_deletion_and_owner_scopes_are_distinct(self):
        store = ResponseStore()
        history = [item(1)]
        first = {"id": "r1", "status": "completed", "output": [item(2)]}
        second = {"id": "r2", "status": "completed", "output": [item(4)]}
        store.put("a", first, history)
        store.put("a", second, history + first["output"] + [item(3)])
        self.assertEqual(store.stats()["blocks"], 4)
        store.delete("a", "r1")
        self.assertEqual(store.get("a", "r2")["input"], [item(1), item(2), item(3)])
        store.put("b", second, [item(1), item(2), item(3)])
        self.assertEqual(store.stats()["blocks"], 8)
        with self.assertRaises(KeyError):
            store.get("c", "r2")
        store.delete("a", "r2")
        self.assertEqual(store.stats()["blocks"], 4)
        store.delete("b", "r2")
        self.assertEqual((store.stats()["blocks"], store.stats()["bytes"]), (0, 0))

    def test_owner_quotas_evict_only_that_owners_records(self):
        store = ResponseStore(owner_entries=1)
        for owner, rid in (("b", "b"), ("a", "a1"), ("a", "a2")):
            store.put(owner, {"id": rid, "output": []}, [])
        with self.assertRaises(KeyError):
            store.get("a", "a1")
        self.assertEqual(store.get("b", "b")["response"]["id"], "b")
        self.assertEqual(store.get("a", "a2")["response"]["id"], "a2")

    def test_failed_capacity_planning_does_not_evict_good_records(self):
        store = ResponseStore(max_bytes=8192, owner_bytes=1024)
        store.put("good", {"id": "old", "output": []}, [])
        before = store.stats()["bytes"]
        with self.assertRaises(ValueError):
            store.put("large", {"id": "new", "output": [item(1)]}, [])
        self.assertEqual(store.stats()["bytes"], before)
        self.assertEqual(store.get("good", "old")["response"]["id"], "old")
        self.assertEqual(store.stats()["rejected"], 1)

    def test_access_does_not_refresh_ttl_and_all_blocks_are_reclaimed(self):
        now = [0]
        store = ResponseStore(ttl=10, clock=lambda: now[0])
        store.put("a", {"id": "r", "output": []}, [item(1)])
        now[0] = 9
        store.get("a", "r")
        now[0] = 10
        with self.assertRaises(KeyError):
            store.get("a", "r")
        self.assertEqual((store.stats()["bytes"], store.stats()["blocks"]), (0, 0))

    def test_reference_accounting_handles_replacement_and_concurrent_readers(self):
        store = ResponseStore(max_entries=16)

        def work(index):
            for revision in range(10):
                store.put(str(index), {"id": "r", "output": [item(revision)]}, [item(100)])
                data = store.get(str(index), "r")
                self.assertEqual(data["input"][0]["id"], "msg_100")
            store.delete(str(index), "r")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(8)))
        self.assertEqual((store.stats()["bytes"], store.stats()["blocks"], store.stats()["inflight_bytes"]), (0, 0, 0))
        self.assertEqual(store._owners, {})

    def test_assembly_limit_and_configuration_are_explicit(self):
        store = ResponseStore(assembly_bytes=128)
        store.put("a", {"id": "r", "output": []}, [item(1)])
        with self.assertRaises(StoreBusy):
            store.get("a", "r")
        with patch.dict(os.environ, {"STRATA_GLM_RESPONSES_OWNER_ENTRIES": "2"}):
            self.assertEqual(limits_from_env()["OWNER_ENTRIES"], 2)
        for value in ("0", "-1", "bad"):
            with patch.dict(os.environ, {"STRATA_GLM_RESPONSES_CACHE_BYTES": value}), self.assertRaises(ValueError):
                limits_from_env()

    def test_maya_environment_aliases_take_precedence_at_startup(self):
        with patch.dict(os.environ, {"MAYA_GLM_RESPONSES_CACHE_ENTRIES": "7",
                                     "STRATA_GLM_RESPONSES_CACHE_ENTRIES": "3"}, clear=True):
            maya_names.alias_env(os.environ)
            apply_server_env(maya_names.env_in({}))
            self.assertEqual(limits_from_env()["CACHE_ENTRIES"], 7)

    def test_maya_config_aliases_and_config_over_environment_precedence(self):
        for env in ({"MAYA_GLM_RESPONSES_CACHE_ENTRIES": "11", "STRATA_GLM_RESPONSES_CACHE_ENTRIES": "5"},
                    {"STRATA_GLM_RESPONSES_CACHE_ENTRIES": "11"}):
            with self.subTest(env=env), patch.dict(os.environ, {"MAYA_GLM_RESPONSES_CACHE_ENTRIES": "7"}, clear=True):
                maya_names.alias_env(os.environ)
                apply_server_env(maya_names.env_in({"env": env}))
                self.assertEqual(limits_from_env()["CACHE_ENTRIES"], 11)


if __name__ == "__main__":
    unittest.main()

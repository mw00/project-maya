import queue
import threading
import types
import unittest
from unittest import mock

from serve import state_cache
from serve.server import StrataEngine


class StateCacheDiagnostics(unittest.TestCase):
    def test_reading_and_prometheus_types(self):
        reading = state_cache.parse(
            "STATE hits=4 misses=2 units=3 ram_bytes=65536 cost_d=0.002 write_failures=1"
        )
        self.assertEqual(reading["hits"], 4)
        rows = {
            name: (kind, value) for name, kind, _, value in state_cache.metrics(reading)
        }
        self.assertEqual(rows["maya:state_cache_hits_total"], ("counter", 4))
        self.assertEqual(rows["maya:state_cache_ram_bytes"], ("gauge", 65536))
        self.assertEqual(rows["maya:state_cache_write_failures_total"], ("counter", 1))

    def test_bad_fields_do_not_reach_metrics(self):
        self.assertEqual(
            state_cache.parse("STATE hits=nan units=-1 misses=inf strange=secret"), {}
        )
        self.assertEqual(state_cache.metrics({"hits": float("inf"), "unknown": 1}), [])

    def test_cache_reading_does_not_replace_decode_or_request_lines(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.lines = queue.Queue()
        engine.multi = False
        engine._routes_lock = threading.Lock()
        engine._routes = {}
        engine.stat = {"tok_s": 25}
        process = types.SimpleNamespace(
            stdout=["STATE hits=3 units=2\n", "T 17\n", "DONE 1\n"]
        )
        with mock.patch.object(engine, "proc", process, create=True):
            engine._pump()
        self.assertEqual(engine.state_cache["hits"], 3)
        self.assertEqual(engine.stat, {"tok_s": 25})
        self.assertEqual(
            [engine.lines.get_nowait() for _ in range(3)], ["T 17\n", "DONE 1\n", None]
        )

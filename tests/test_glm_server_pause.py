import itertools
import os
import queue
import threading
import types
import unittest
from unittest.mock import patch
from serve.server import StrataEngine


class Input:
    def __init__(self, engine):
        self.engine, self.writes = engine, []

    def write(self, line):
        self.writes.append(line)
        if line.startswith('GEN '):
            if self.engine.multi:
                rid = self.engine._tls.rid
                self.engine._routes[rid].put(f'T {rid} 42')
            else:
                self.engine.lines.put('T 42')
        elif line.startswith(('PAUSE', 'STOP')):
            if self.engine.multi:
                rid = self.engine._tls.rid
                self.engine._routes[rid].put(f'DONE {rid} 0 0 0 0 cancel')
            else:
                self.engine.lines.put('DONE 0 0 0 0 cancel')

    def flush(self):
        pass


class CancelTest(unittest.TestCase):
    def test_disconnect(self):
        for multi in (False, True):
            for pause, override in ((True, False), (True, True), (False, False)):
                with self.subTest(multi=multi, pause=pause, override=override):
                    e = object.__new__(StrataEngine)
                    e.multi, e.can_pause, e.can_stop = multi, pause, True
                    e.lines = queue.Queue()
                    e._routes, e._routes_lock = {}, threading.Lock()
                    e._rids, e._tls = itertools.count(17), threading.local()
                    inp = Input(e)
                    e.proc = types.SimpleNamespace(stdin=inp)
                    e._parse_done = lambda line: None
                    with patch.dict(os.environ, {'STRATA_CANCEL_STOP': '1' if override else '0'}):
                        gen = e.generate([1, 2], 10, {}, threading.Event())
                        self.assertEqual(next(gen), 42)
                        gen.close()
                    want = ('PAUSE' if pause and not override else 'STOP')
                    want += ' 17' if multi else ''
                    want += ' keep' if pause and not override else ''
                    self.assertEqual(inp.writes[-1], want + '\n')
                    self.assertEqual(e._routes, {})

if __name__ == '__main__':
    unittest.main()

"""The web app's answers (strata_resume): they run to their end with nobody reading, and stay for a page that comes
back much later (a phone whose screen slept); only the page's Stop ends one early."""
import threading
import time
import unittest

from serve import server as S


def chunks(n, gate=None, cancel=None):
    for i in range(n):
        if gate is not None:
            gate.wait()
        if cancel is not None and cancel.is_set():
            return
        yield {"id": "x", "i": i}


class StreamJobs(unittest.TestCase):
    def setUp(self):
        S.JOBS.clear()
        S.JOBS_DIR = None

    def tearDown(self):
        S.JOBS.clear()
        S.JOBS_DIR = None

    def wait_done(self, job, secs=5):
        t0 = time.time()
        while not job.done and time.time() - t0 < secs:
            time.sleep(0.01)
        return job.done

    def test_an_unread_answer_runs_to_its_end(self):
        cancel = threading.Event()
        job = S.StreamJob("a", chunks(50), cancel)
        self.assertEqual(job.readers, 0)
        self.assertTrue(self.wait_done(job))
        self.assertFalse(cancel.is_set())
        items, done = job.wait(0, 0)
        self.assertEqual((len(items), done), (50, True))
        self.assertFalse(hasattr(S.StreamJob, "ORPHAN_S"))      # no time limit for an unread answer

    def test_a_new_answer_never_cancels_a_running_unread_one(self):
        gate, c1 = threading.Event(), threading.Event()
        first = S.StreamJob("first", chunks(5, gate, c1), c1)
        S.jobs_put(first)
        S.jobs_put(S.StreamJob("second", chunks(1), threading.Event()))
        self.assertFalse(c1.is_set())
        gate.set()
        self.assertTrue(self.wait_done(first))
        self.assertEqual(len(first.wait(0, 0)[0]), 5)
        self.assertIn("first", S.JOBS)

    def test_finished_answers_stay_a_day_and_the_oldest_go_first(self):
        jobs = []
        for i in range(S.JOB_KEEP_N + 5):
            j = S.StreamJob(f"j{i}", chunks(1), threading.Event())
            self.assertTrue(self.wait_done(j))
            j.ended = time.time() - 3600 + i                     # an hour ago: still kept
            jobs.append(j)
            S.jobs_put(j)
        self.assertLessEqual(len(S.JOBS), S.JOB_KEEP_N)
        self.assertIn(jobs[-1].id, S.JOBS)
        self.assertNotIn(jobs[0].id, S.JOBS)
        old = S.StreamJob("old", chunks(1), threading.Event())
        self.assertTrue(self.wait_done(old))
        old.ended = time.time() - S.JOB_KEEP_S - 1
        S.JOBS["old"] = old
        S.jobs_put(S.StreamJob("new", chunks(1), threading.Event()))
        self.assertNotIn("old", S.JOBS)                          # more than a day after its end: gone

    def test_running_answers_are_never_dropped_to_make_room(self):
        gate, cancel = threading.Event(), threading.Event()
        running = S.StreamJob("run", chunks(3, gate, cancel), cancel)
        S.jobs_put(running)
        for i in range(S.JOB_KEEP_N + 3):
            j = S.StreamJob(f"d{i}", chunks(1), threading.Event())
            self.assertTrue(self.wait_done(j))
            S.jobs_put(j)
        self.assertIn("run", S.JOBS)
        self.assertFalse(cancel.is_set())
        gate.set()
        self.assertTrue(self.wait_done(running))


class StreamJobFiles(StreamJobs.__bases__[0]):
    """The answers on disk (answers/ beside the config): a restart of the server keeps them for the page."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        S.JOBS.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "answers"
        self.assertEqual(S.jobs_load(self.dir), 0)

    def tearDown(self):
        S.JOBS.clear()
        S.JOBS_DIR = None
        self.tmp.cleanup()

    def restart(self):
        S.JOBS.clear()
        S.JOBS_DIR = None
        return S.jobs_load(self.dir)

    def test_a_finished_answer_survives_a_restart(self):
        job = S.StreamJob("chatcmpl-a1", chunks(40), threading.Event())
        S.jobs_put(job)
        StreamJobs.wait_done(self, job)
        time.sleep(0.05)
        self.assertEqual(self.restart(), 1)
        back = S.JOBS["chatcmpl-a1"]
        items, done = back.wait(0, 0)
        self.assertTrue(done)
        self.assertEqual(items, job.items)
        self.assertAlmostEqual(back.ended, job.ended, places=3)

    def test_an_answer_cut_by_a_stop_comes_back_with_a_note(self):
        f = self.dir / "chatcmpl-cut.jsonl"
        f.write_text('{"id": "x", "i": 0}\n{"id": "x", "i": 1}\n{"id": "x", "i"', encoding="utf-8")   # torn last line
        self.assertEqual(self.restart(), 1)
        items, done = S.JOBS["chatcmpl-cut"].wait(0, 0)
        self.assertTrue(done)
        self.assertEqual([c.get("i") for c in items[:2]], [0, 1])
        self.assertEqual(items[-1]["error"]["message"], S.RESTARTED)

    def test_old_files_and_strange_names_are_removed_and_eviction_deletes_the_file(self):
        import os
        old = self.dir / "chatcmpl-old.jsonl"
        old.write_text('{"_done": 1}\n', encoding="utf-8")
        os.utime(old, (time.time() - S.JOB_KEEP_S - 10,) * 2)
        odd = self.dir / "bad name!.jsonl"
        odd.write_text('{"_done": 1}\n', encoding="utf-8")
        self.assertEqual(self.restart(), 0)
        self.assertFalse(old.exists())
        self.assertFalse(odd.exists())
        job = S.StreamJob("chatcmpl-e", chunks(1), threading.Event())
        S.jobs_put(job)
        StreamJobs.wait_done(self, job)
        time.sleep(0.05)
        path = self.dir / "chatcmpl-e.jsonl"
        self.assertTrue(path.exists())
        job.ended = time.time() - S.JOB_KEEP_S - 1
        S.jobs_put(S.StreamJob("chatcmpl-f", chunks(1), threading.Event()))
        self.assertFalse(path.exists())

    def test_without_a_folder_nothing_is_written(self):
        S.JOBS_DIR = None
        job = S.StreamJob("chatcmpl-ram", chunks(3), threading.Event())
        StreamJobs.wait_done(self, job)
        self.assertIsNone(job.log)
        self.assertFalse((self.dir / "chatcmpl-ram.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

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

    def tearDown(self):
        S.JOBS.clear()

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


if __name__ == "__main__":
    unittest.main()

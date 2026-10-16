"""Bounded background Responses execution and credential-scoped SSE replay."""
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
import copy
import json
import threading
import time


class JobError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class BackgroundJobs:
    def __init__(self, store, limits):
        self.store, self.limits = store, limits
        self._jobs, self._queue = OrderedDict(), deque()
        self._cv = threading.Condition(threading.RLock())
        self._workers, self._closed, self._metrics = [], False, Counter()
        self._readers = Counter()

    def _sweep(self):
        now = time.monotonic()
        for key, job in list(self._jobs.items()):
            if not job["done"] and job["response"]["status"] == "queued" and now >= job["deadline"]:
                self._finish(job, error="background task exceeded its deadline")
            if job["done"] and now - job["ended"] >= self.limits["JOB_TTL_S"]:
                del self._jobs[key]

    def _append(self, job, name, payload):
        payload = copy.deepcopy(payload)
        payload["sequence_number"] = job["next"]
        size = len(json.dumps(payload, ensure_ascii=False).encode())
        if size > self.limits["EVENT_BYTES"]:
            raise JobError(413, "a background event exceeds the event-buffer byte limit")
        while job["events"] and (len(job["events"]) >= self.limits["EVENT_COUNT"] or
                                  job["bytes"] + size > self.limits["EVENT_BYTES"]):
            _, _, old_size = job["events"].popleft()
            job["bytes"] -= old_size
        job["events"].append((name, payload, size))
        job["bytes"] += size
        job["next"] += 1
        self._cv.notify_all()

    def submit(self, owner, initial, inputs, work):
        size = len(json.dumps({"response": initial, "input": inputs}, ensure_ascii=False).encode())
        if size > self.limits["JOB_INPUT_BYTES"]:
            raise JobError(413, "background request exceeds the input snapshot limit")
        key = owner, initial["id"]
        with self._cv:
            self._sweep()
            if self._closed:
                raise JobError(503, "background task manager is stopping")
            active = sum(not j["done"] for j in self._jobs.values())
            owned = sum(k[0] == owner and not j["done"] for k, j in self._jobs.items())
            if active >= self.limits["JOB_ENTRIES"] or owned >= self.limits["JOB_OWNER_ENTRIES"]:
                raise JobError(429, "background queue is at capacity")
            job = {"owner": owner, "response": copy.deepcopy(initial), "input": inputs, "work": work,
                   "events": deque(), "bytes": 0, "next": 0, "cancel": threading.Event(),
                   "cancel_requested": False, "done": False, "ended": None,
                   "deadline": time.monotonic() + self.limits["JOB_TIMEOUT_S"]}
            self._append(job, "response.queued", {"type": "response.queued", "response": initial})
            while len(self._jobs) >= self.limits["JOB_ENTRIES"]:
                victim = next(k for k, value in self._jobs.items() if value["done"])
                del self._jobs[victim]
            self._jobs[key] = job
            self._queue.append(key)
            self._metrics["submitted"] += 1
            if not self._workers:
                for index in range(self.limits["JOB_WORKERS"]):
                    worker = threading.Thread(target=self._run, name=f"maya-response-worker-{index}", daemon=True)
                    self._workers.append(worker)
                    worker.start()
            self._cv.notify_all()
            return copy.deepcopy(initial)

    def _finish(self, job, terminal=None, error=None):
        if job["done"]:
            return
        response = copy.deepcopy(terminal[1]["response"] if terminal else job["response"])
        response["created_at"] = job["response"]["created_at"]
        response["completed_at"] = int(time.time())
        if time.monotonic() >= job["deadline"]:
            error = "background task exceeded its deadline"
        elif job["cancel_requested"]:
            response.update(status="cancelled", error=None, incomplete_details=None, store=True)
            name, payload = "error", {"type": "error", "code": "response_cancelled", "param": None,
                                      "message": "background response was cancelled"}
            error = None
        else:
            name, payload = terminal if terminal else (None, None)
        if response["status"] != "cancelled":
            if error is not None:
                response.update(status="failed", output=[], store=False,
                                error={"code": "server_error", "message": str(error)})
                name = "response.failed"
            elif terminal is None:
                response.update(status="failed", output=[], store=False,
                                error={"code": "server_error", "message": "background generation ended without a terminal event"})
                name = "response.failed"
            payload = {"type": name, "response": response}
        # Check the full terminal event before committing a cache record.
        if len(json.dumps(payload, ensure_ascii=False).encode()) > self.limits["EVENT_BYTES"]:
            response.update(status="failed", output=[], store=False,
                            error={"code": "server_error", "message": "terminal response exceeds event-buffer capacity"})
            name, payload = "response.failed", {"type": "response.failed", "response": response}
        if response["status"] in ("completed", "incomplete"):
            try:
                self.store.put(job["owner"], response, job["input"])
            except ValueError as exc:
                response.update(status="failed", store=False, error={"code": "server_error", "message": str(exc)})
                name, payload = "response.failed", {"type": "response.failed", "response": response}
        if len(json.dumps(payload, ensure_ascii=False).encode()) > self.limits["EVENT_BYTES"]:
            response = {key: response[key] for key in ("id", "object", "model", "created_at")}
            response.update(status="failed", output=[], tools=[], tool_choice="auto", parallel_tool_calls=True,
                            background=True, store=False, error={"code": "server_error", "message": "terminal event exceeds capacity"})
            name, payload = "response.failed", {"type": "response.failed", "response": response}
        job["response"] = response
        self._append(job, name, payload)
        job["done"], job["ended"], job["work"] = True, time.monotonic(), None
        try:
            self._queue.remove((job["owner"], response["id"]))
        except ValueError:
            pass
        self._metrics[response["status"]] += 1
        self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._closed or self._queue)
                if self._closed and not self._queue:
                    return
                key = self._queue.popleft()
                job = self._jobs.get(key)
                if job is None or job["done"]:
                    continue
                if time.monotonic() >= job["deadline"]:
                    self._finish(job, error="background task exceeded its deadline")
                    continue
                job["response"]["status"] = "in_progress"
                work = job["work"]
            timer = threading.Timer(max(0, job["deadline"] - time.monotonic()), job["cancel"].set)
            timer.daemon = True
            timer.start()
            source, terminal, error = None, None, None
            try:
                source = work(job["cancel"], key[1])
                for event in source:
                    if event is None:
                        continue
                    name, payload = event
                    if name in ("response.completed", "response.incomplete", "response.failed"):
                        terminal = event
                        break
                    with self._cv:
                        if "response" in payload:
                            payload["response"]["created_at"] = job["response"]["created_at"]
                            job["response"] = copy.deepcopy(payload["response"])
                        self._append(job, name, payload)
            except Exception as exc:
                job["cancel"].set()
                error = str(exc)
            finally:
                timer.cancel()
                timer.join()
                if source is not None:
                    try:
                        source.close()
                    except Exception as exc:
                        error = str(exc)
                with self._cv:
                    self._finish(job, terminal, error)

    def get(self, owner, response_id):
        with self._cv:
            self._sweep()
            job = self._jobs[(owner, response_id)]
            return {"response": copy.deepcopy(job["response"]), "input": copy.deepcopy(job["input"])}

    def cancel(self, owner, response_id, wait=5):
        with self._cv:
            self._sweep()
            job = self._jobs[(owner, response_id)]
            if not job["done"]:
                job["cancel_requested"] = True
                job["cancel"].set()
                if job["response"]["status"] == "queued":
                    self._finish(job)
                else:
                    self._cv.wait_for(lambda: job["done"], timeout=wait)
            return copy.deepcopy(job["response"])

    def delete(self, owner, response_id):
        with self._cv:
            self._sweep()
            job = self._jobs[(owner, response_id)]
            if not job["done"]:
                raise JobError(409, "cancel the running response before deleting it")
            del self._jobs[(owner, response_id)]

    def check_cursor(self, owner, response_id, after):
        with self._cv:
            self._sweep()
            job = self._jobs[(owner, response_id)]
            first = job["events"][0][1]["sequence_number"]
            if after < first - 1:
                raise JobError(410, "event cursor expired; retrieve the response snapshot")
            if after >= job["next"]:
                raise JobError(400, "starting_after is beyond the latest event")

    def events(self, owner, response_id, after=-1):
        while True:
            with self._cv:
                self.check_cursor(owner, response_id, after)
                job = self._jobs[(owner, response_id)]
                ready = [(name, copy.deepcopy(payload)) for name, payload, _ in job["events"]
                         if payload["sequence_number"] > after]
                done = job["done"]
                if not ready and not done:
                    self._cv.wait(timeout=1)
            for name, payload in ready:
                after = payload["sequence_number"]
                yield name, payload
            if done:
                return
            if not ready:
                yield None

    @contextmanager
    def reader(self, owner, response_id, after=-1):
        with self._cv:
            self.check_cursor(owner, response_id, after)
            if (sum(self._readers.values()) >= self.limits["STREAM_READERS"] or
                    self._readers[owner] >= self.limits["STREAM_OWNER_READERS"]):
                raise JobError(429, "background stream reader capacity reached")
            self._readers[owner] += 1
        events = self.events(owner, response_id, after)
        try:
            yield events
        finally:
            events.close()
            with self._cv:
                self._readers[owner] -= 1
                if self._readers[owner] == 0:
                    del self._readers[owner]

    def close(self, timeout=5):
        with self._cv:
            self._closed = True
            for job in self._jobs.values():
                if not job["done"]:
                    job["cancel_requested"] = True
                    job["cancel"].set()
                    if job["response"]["status"] == "queued":
                        self._finish(job)
            self._cv.notify_all()
        deadline = time.monotonic() + timeout
        for worker in self._workers:
            worker.join(max(0, deadline - time.monotonic()))
        if any(worker.is_alive() for worker in self._workers):
            raise RuntimeError("background workers have not stopped; engine cleanup is still pending")

    def stats(self):
        with self._cv:
            self._sweep()
            return {**dict(self._metrics), "records": len(self._jobs),
                    "running": sum(j["response"]["status"] == "in_progress" for j in self._jobs.values()),
                    "queued": sum(j["response"]["status"] == "queued" for j in self._jobs.values()),
                    "readers": sum(self._readers.values()),
                    "event_bytes": sum(j["bytes"] for j in self._jobs.values())}

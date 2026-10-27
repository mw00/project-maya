import json
import contextlib
import hashlib
import struct
from pathlib import Path
import tempfile
import threading
import time
import unittest

from serve.response_jobs import BackgroundJobs
from serve.response_store import ResponseStore
from serve.test_responses import RecordingEngine, ResponseCase, ROOT, openai
from serve import ds4_vision
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer
from serve.test_server import LifecycleEngine


class BackgroundResponses(ResponseCase):
    def configure(self, **limits):
        self.svc.response_jobs.close()
        self.svc.response_limits.update(limits)
        self.svc.response_jobs = BackgroundJobs(self.svc.responses, self.svc.response_limits)

    def get(self, rid, suffix="", headers=None):
        with self.request(path="/v1/responses/" + rid + suffix, headers=headers) as response:
            return response.status, json.load(response)

    def wait(self, rid, status=None, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code, out = self.get(rid)
            self.assertEqual(code, 200, out)
            if out["status"] == status or (status is None and out["status"] not in ("queued", "in_progress")):
                return out
            time.sleep(0.01)
        self.fail(f"background response did not reach {status or 'a terminal state'}: {out}")

    def cancel(self, rid, headers=None):
        with self.request({}, path="/v1/responses/" + rid + "/cancel", headers=headers) as response:
            return response.status, json.load(response)

    def test_background_completion_replay_and_continuation(self):
        self.script("Hello", delay_s=0.002)
        status, first = self.post(background=True, store=True)
        self.assertEqual((status, first["status"]), (200, "queued"))
        out = self.wait(first["id"])
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["created_at"], first["created_at"])
        self.assertEqual(out["output"][0]["content"][0]["text"], "Hello")
        with self.request(path="/v1/responses/" + first["id"] + "?stream=true") as response:
            raw = response.read().decode()
        events = [json.loads(block.split("\ndata: ", 1)[1]) for block in raw.strip().split("\n\n") if block.startswith("event:")]
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))
        self.assertEqual((events[0]["type"], events[-1]["type"]), ("response.queued", "response.completed"))
        after = events[3]["sequence_number"]
        with self.request(path=f"/v1/responses/{first['id']}?stream=true&starting_after={after}") as response:
            tail = response.read().decode()
        resumed = [json.loads(block.split("\ndata: ", 1)[1]) for block in tail.strip().split("\n\n") if block.startswith("event:")]
        self.assertEqual(resumed, events[4:])
        self.script("Next")
        self.assertEqual(self.post(previous_response_id=first["id"])[0], 200)

    def test_background_stream_can_disconnect_without_cancelling_generation(self):
        self.script("x" * 100, delay_s=0.002)
        rid = None
        with self.request({"input": "hi", "reasoning": {"effort": "none"}, "background": True,
                           "store": True, "stream": True}) as response:
            while True:
                line = response.readline()
                self.assertTrue(line)
                if line.startswith(b"data: "):
                    event = json.loads(line[6:])
                    if "response" in event:
                        rid = event["response"]["id"]
                    if event["type"] == "response.output_text.delta":
                        break
        out = self.wait(rid)
        self.assertEqual(out["status"], "completed")
        self.assertFalse(self.engine.cancel.is_set())

    def test_foreground_and_background_responses_resume_after_idle_unload(self):
        class Vision:
            owed = False

            def active(self):
                return contextlib.nullcontext()

            def encode(self, source):
                self.running_at_encode = engine.alive()
                self.restarts_at_encode = engine.restarts
                path = self.dir / "encoded.sve"
                path.write_bytes(b"image-embedding")
                return path, 3

        with tempfile.TemporaryDirectory() as directory:
            for background in (False, True):
                for image in (False, True):
                    with self.subTest(background=background, image=image):
                        engine = self.engine = self.svc.engine = LifecycleEngine(self.tok, "Resumed", max_context=8192)
                        vision = Vision()
                        vision.dir = Path(directory)
                        self.svc.vision = vision if image else None
                        self.assertEqual(self.svc.unload(), "unloaded")
                        history = [{"role": "user", "content": [
                            {"type": "input_image", "image_url": "data:image/png;base64,fixture"}]}] if image else "hi"
                        status, out = self.post(background=background, store=True, input=history)
                        self.assertEqual(status, 200, out)
                        if background:
                            out = self.wait(out["id"])
                        self.assertEqual(out["status"], "completed")
                        self.assertEqual(out["output"][0]["content"][0]["text"], "Resumed")
                        self.assertEqual((engine.restarts, engine.running, self.svc.unloaded), (1, True, False))
                        if image:
                            self.assertEqual((vision.running_at_encode, vision.restarts_at_encode), (True, 1))
                            self.assertFalse(Path(engine.last_embeddings).exists())

    def test_cancelling_a_waiting_image_request_keeps_the_engine_unloaded(self):
        self.engine = self.svc.engine = LifecycleEngine(self.tok, "Resumed", max_context=8192)

        class Vision:
            owed = False

        self.svc.vision = Vision()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.svc.fifo.acquire()
        try:
            status, initial = self.post(background=True, store=True, input=[{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,fixture"}]}])
            self.assertEqual(status, 200, initial)
            self.wait(initial["id"], "in_progress")
            self.assertEqual(self.cancel(initial["id"])[1]["status"], "cancelled")
            self.assertEqual((self.engine.restarts, self.engine.running, self.svc.unloaded), (0, False, True))
            self.assertEqual(self.engine.last_prompt, [])
            self.assertEqual(self.svc.fifo.waiting(), 0)
        finally:
            self.svc.fifo.release()

    def test_running_cancel_drains_engine_and_is_idempotent(self):
        self.script("x" * 2000, delay_s=0.002)
        _, initial = self.post(background=True, store=True)
        self.wait(initial["id"], "in_progress")
        status, out = self.cancel(initial["id"])
        self.assertEqual((status, out["status"]), (200, "cancelled"))
        self.assertEqual(self.svc.active, 0)
        self.assertEqual(self.cancel(initial["id"])[1], out)
        self.script("Ready")
        self.assertEqual(self.post()[0], 200)
        self.assertEqual(self.post(previous_response_id=initial["id"])[0], 409)

    def test_cancel_while_waiting_for_the_engine_does_not_wait_for_its_holder(self):
        self.svc.fifo.acquire()
        try:
            _, initial = self.post(background=True, store=True)
            self.wait(initial["id"], "in_progress")
            self.assertEqual(self.cancel(initial["id"])[1]["status"], "cancelled")
            self.assertEqual(self.engine.last_prompt, [])
            self.assertEqual(self.svc.fifo.waiting(), 0)
            self.assertTrue(self.svc.fifo.locked())
        finally:
            self.svc.fifo.release()

    def test_queued_cancel_never_starts_a_second_generation(self):
        self.configure(JOB_WORKERS=1)
        self.script("x" * 2000, delay_s=0.002)
        _, first = self.post(background=True, store=True, input="first")
        self.wait(first["id"], "in_progress")
        _, queued = self.post(background=True, store=True, input="second")
        self.assertEqual(self.get(queued["id"])[1]["status"], "queued")
        self.assertEqual(self.cancel(queued["id"])[1]["status"], "cancelled")
        self.cancel(first["id"])
        self.assertLessEqual(len(self.svc.history), 1)
        self.assertEqual(len(self.svc.response_jobs._queue), 0)

    def test_expired_queued_job_never_starts_generation(self):
        self.configure(JOB_WORKERS=1)
        self.script("x" * 2000, delay_s=0.002)
        _, first = self.post(background=True, store=True)
        self.wait(first["id"], "in_progress")
        _, second = self.post(background=True, store=True)
        manager = self.svc.response_jobs
        with manager._cv:
            manager._jobs[("public", second["id"])]["deadline"] = time.monotonic() - 1
        expired = self.get(second["id"])[1]
        self.assertEqual(expired["status"], "failed")
        self.assertIn("deadline", expired["error"]["message"])
        self.cancel(first["id"])
        self.assertLessEqual(len(self.svc.history), 1)
        self.assertEqual(len(manager._queue), 0)

    def test_completion_and_cancellation_publish_only_one_terminal_state(self):
        self.script("ok", delay_s=0.001)
        for _ in range(8):
            _, initial = self.post(background=True, store=True)
            status, cancelled = self.cancel(initial["id"])
            self.assertEqual(status, 200)
            self.assertIn(cancelled["status"], ("completed", "cancelled"))
            events = list(self.svc.response_jobs.events("public", initial["id"]))
            terminal = [event for event in events if event and event[0] in
                        ("response.completed", "response.incomplete", "response.failed", "error")]
            self.assertEqual(len(terminal), 1)
            self.assertEqual(self.get(initial["id"])[1]["status"], cancelled["status"])
        self.assertEqual(self.svc.active, 0)

    def test_metrics_are_aggregate_and_do_not_expose_request_contents(self):
        self.assertEqual(self.post(input="private-fixture-content", store=True)[0], 200)
        with self.request(path="/v1/status") as response:
            out = json.load(response)
        self.assertEqual(out["responses"]["cache"]["entries"], 1)
        self.assertFalse(out["responses"]["structured_outputs"]["constrained_decoding"])
        with self.request(path="/metrics/prometheus") as response:
            text = response.read().decode()
        self.assertIn("maya:responses_cache_bytes", text)
        self.assertIn("maya:responses_jobs_readers", text)
        self.assertNotIn("private-fixture-content", text)

    def test_queue_admission_and_input_limits(self):
        self.configure(JOB_WORKERS=1, JOB_ENTRIES=2, JOB_OWNER_ENTRIES=1)
        self.script("x" * 2000, delay_s=0.002)
        _, first = self.post(background=True, store=True)
        self.assertEqual(self.post(background=True, store=True)[0], 429)
        self.cancel(first["id"])
        self.configure(JOB_INPUT_BYTES=128)
        self.assertEqual(self.post(background=True, store=True)[0], 413)
        self.assertEqual(self.svc.response_jobs.stats()["records"], 0)

    def test_deadline_stops_work_and_shutdown_joins_workers(self):
        self.configure(JOB_TIMEOUT_S=0.05)
        self.script("x" * 2000, delay_s=0.002)
        _, initial = self.post(background=True, store=True)
        out = self.wait(initial["id"])
        self.assertEqual(out["status"], "failed")
        self.assertIn("deadline", out["error"]["message"])
        self.assertEqual(self.svc.active, 0)
        self.configure(JOB_TIMEOUT_S=300)
        _, initial = self.post(background=True, store=True)
        self.wait(initial["id"], "in_progress")
        self.svc.response_jobs.close()
        self.assertTrue(all(not worker.is_alive() for worker in self.svc.response_jobs._workers))
        self.assertEqual(self.get(initial["id"])[1]["status"], "cancelled")
        self.assertEqual(self.post(background=True, store=True)[0], 503)

    def test_cursor_retention_and_event_byte_limits(self):
        self.configure(EVENT_BYTES=4096, EVENT_COUNT=4)
        self.script("x" * 200)
        _, initial = self.post(background=True, store=True)
        self.assertEqual(self.wait(initial["id"])["status"], "completed")
        with self.request(path=f"/v1/responses/{initial['id']}?stream=true&starting_after=0") as response:
            self.assertEqual(response.status, 410)
        self.assertLessEqual(self.svc.response_jobs.stats()["event_bytes"], 4096)
        self.script(" ".join(f"word{i}" for i in range(800)))   # (5000+ bytes and no loop: the loop guard ends one)
        _, initial = self.post(background=True, store=True)
        out = self.wait(initial["id"])
        self.assertEqual(out["status"], "failed")
        self.assertIn("limit", out["error"]["message"])

    def test_auth_delete_conflicts_and_cache_eviction_do_not_drop_active_jobs(self):
        self.svc.api_key = ["a", "b"]
        headers = {"Authorization": "Bearer a"}
        self.svc.responses = ResponseStore(max_entries=1)
        self.script("x" * 2000, delay_s=0.002)
        with self.request({"input": "hi", "store": True, "background": True,
                           "reasoning": {"effort": "none"}}, headers=headers) as response:
            out = json.load(response)
        rid = out["id"]
        owner = hashlib.sha256(b"a").hexdigest()
        self.svc.responses.put(owner, {"id": "other", "output": []}, [])
        self.assertEqual(self.get(rid, headers=headers)[0], 200)
        other = {"x-api-key": "b"}
        self.assertEqual(self.get(rid, headers=other)[0], 404)
        self.assertEqual(self.cancel(rid, other)[0], 404)
        with self.request(path=f"/v1/responses/{rid}", method="DELETE", headers=headers) as response:
            self.assertEqual(response.status, 409)
        self.assertEqual(self.cancel(rid, headers)[1]["status"], "cancelled")
        with self.request(path=f"/v1/responses/{rid}", method="DELETE", headers=headers) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.get(rid, headers=headers)[0], 404)

    def test_background_schema_validation_is_used(self):
        self.script('{"value":1}')
        _, initial = self.post(background=True, store=True, text={"format": {"type": "json_object"}})
        self.assertEqual(self.wait(initial["id"])["status"], "completed")
        self.script("invalid")
        _, initial = self.post(background=True, store=True, text={"format": {"type": "json_object"}})
        self.assertEqual(self.wait(initial["id"])["status"], "failed")

    def test_cancel_during_buffered_validation_never_publishes_content(self):
        self.script('{"text":"' + "x" * 2000 + '"}', delay_s=0.002)
        _, initial = self.post(background=True, store=True, text={"format": {"type": "json_object"}})
        self.wait(initial["id"], "in_progress")
        self.assertEqual(self.cancel(initial["id"])[1]["status"], "cancelled")
        events = list(self.svc.response_jobs.events("public", initial["id"]))
        self.assertNotIn("response.output_text.delta", [e[0] for e in events if e])
        self.assertEqual(self.svc.active, 0)

    def test_background_image_preparation_and_generation_share_resource_ownership(self):
        class Vision:
            owed = False

            def active(self):
                return contextlib.nullcontext()

            def encode(self, source):
                self.thread = threading.get_ident()
                path = self.dir / "encoded.sve"
                path.write_bytes(b"image-embedding")
                return path, 3

        class Engine(RecordingEngine):
            def generate(self, *args, **kwargs):
                self.thread = threading.get_ident()
                yield from super().generate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            vision = Vision()
            vision.dir = Path(directory)
            self.svc.vision = vision
            self.engine = Engine(self.tok, "Image received")
            self.svc.engine = self.engine
            _, initial = self.post(background=True, store=True, input=[{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,fixture"}]}])
            self.assertEqual(self.wait(initial["id"])["status"], "completed")
            self.assertEqual(vision.thread, self.engine.thread)
            self.assertEqual(self.engine.image_bytes, b"image-embedding")
            self.assertFalse(Path(self.engine.last_embeddings).exists())

    def test_stream_reader_admission_is_bounded_and_released(self):
        from serve.response_jobs import JobError
        self.configure(STREAM_READERS=1, STREAM_OWNER_READERS=1)
        _, initial = self.post(background=True, store=True)
        self.wait(initial["id"])
        manager = self.svc.response_jobs
        with manager.reader("public", initial["id"]):
            with self.assertRaises(JobError):
                with manager.reader("public", initial["id"]):
                    pass
            self.assertEqual(manager.stats()["readers"], 1)
        self.assertEqual(manager.stats()["readers"], 0)

    def test_ds4_images_reach_foreground_and_background_engines_and_are_cleaned(self):
        class Tokenizer(ByteTokenizer):
            SPECIALS = ByteTokenizer.SPECIALS + [ds4_vision.PLACEHOLDER]
            special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}

        class Vision(ds4_vision.Vision):
            def picture(self, source):
                self.thread = threading.get_ident()
                return ds4_vision.Picture(b"\x7f" * 12, 2, 2, 1, 1)

        class Engine(RecordingEngine):
            def generate(self, *args, **kwargs):
                self.thread = threading.get_ident()
                yield from super().generate(*args, **kwargs)

        self.tok = self.svc.tok = Tokenizer()
        self.svc.template = ChatTemplate(ROOT / "tools/ds4/chat_template.jinja")
        image_token = self.tok.special_tokens[ds4_vision.PLACEHOLDER]
        with tempfile.TemporaryDirectory() as directory:
            self.svc.vision = Vision(ds4_vision.Geometry(image_token=image_token), Path(directory), None)
            for background in (False, True):
                with self.subTest(background=background):
                    self.engine = self.svc.engine = Engine(self.tok, "Image received")
                    status, out = self.post(background=background, store=True, input=[{"role": "user", "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,fixture"}]}])
                    self.assertEqual(status, 200, out)
                    if background:
                        out = self.wait(out["id"])
                    self.assertEqual(out["status"], "completed")
                    self.assertEqual(self.svc.vision.thread, self.engine.thread)
                    magic, count, start, width, height = struct.unpack("<4siqii", self.engine.image_bytes[:24])
                    self.assertEqual((magic, count, width, height), (b"DSI1", 1, 2, 2))
                    self.assertEqual(self.engine.last_prompt[start:start + 4], [image_token] * 4)
                    self.assertEqual(self.engine.image_bytes[24:], b"\x7f" * 12)
                    self.assertFalse(Path(self.engine.last_embeddings).exists())
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_ds4_cancel_during_picture_preparation_stops_before_writing(self):
        cancel = threading.Event()

        class Vision(ds4_vision.Vision):
            def picture(self, source):
                cancel.set()
                return ds4_vision.Picture(b"\x7f" * 12, 2, 2, 1, 1)

        with tempfile.TemporaryDirectory() as directory:
            self.svc.vision = Vision(ds4_vision.Geometry(), Path(directory), None)
            with self.assertRaisesRegex(ValueError, "cancelled"):
                self.svc.prepare([{"role": "user", "content": [{"type": "image", "source": "fixture"}]}],
                                 None, {}, cancel=cancel)
            self.assertIsNone(getattr(self.svc.embeddings, "path", None))
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(openai, "optional OpenAI SDK is not installed")
    def test_sdk_background_retrieve_stream_cancel_and_delete(self):
        with openai.OpenAI(base_url=self.base + "/v1", api_key="local", max_retries=0,
                           _strict_response_validation=True) as client:
            first = client.responses.create(model="m", input="hi", store=True, background=True,
                                              reasoning={"effort": "none"})
            self.wait(first.id)
            self.assertEqual(client.responses.retrieve(first.id).status, "completed")
            events = list(client.responses.retrieve(first.id, stream=True))
            self.assertEqual((events[0].type, events[-1].type), ("response.queued", "response.completed"))
            resumed = list(client.responses.retrieve(first.id, stream=True, starting_after=events[3].sequence_number))
            self.assertEqual(len(resumed), len(events) - 4)
            client.responses.delete(first.id)
            with self.assertRaises(openai.NotFoundError):
                client.responses.retrieve(first.id)
            self.script("x" * 2000, delay_s=0.002)
            second = client.responses.create(model="m", input="hi", store=True, background=True,
                                               reasoning={"effort": "none"})
            self.wait(second.id, "in_progress")
            self.assertEqual(client.responses.cancel(second.id).status, "cancelled")


if __name__ == "__main__":
    unittest.main()

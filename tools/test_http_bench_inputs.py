"""Offline checks against a recorded input, plus calibration guard boundaries."""
import json
import unittest

from http_bench_inputs import (CalibrationError, MODEL_ALIAS, REQUEST_SEED, SEED,
                               calibrate_prompt, case_plan, request_payload, sha,
                               synthetic_prompt)


class Counter:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, payload):
        self.calls.append(payload)
        return next(self.responses)


def exact(tokens, maximum=262144):
    return {"tokens": tokens, "approximate": False, "max_context": maximum}


class InputTests(unittest.TestCase):
    def test_recorded_short_input(self):
        # Frozen original_control/short_1 result, not recomputed expectations.
        counter = Counter(exact(355), exact(256))
        prompt, marker, meta = calibrate_prompt(counter, MODEL_ALIAS, SEED, "short_1", 256, 128)
        self.assertEqual(meta["calibration_probes"], 2)
        self.assertEqual(prompt.count(marker), 1)
        self.assertEqual(sha(prompt), "982bf5c54039622e1cfb845abd642dda1f51a60638911fb339b213aa2f26cb3f")
        payload = request_payload(MODEL_ALIAS, prompt, 128)
        self.assertEqual(sha(json.dumps(payload, separators=(",", ":"))),
                         "0e23dfa08f28b61a00e8d48c0ee74d1c50f24f4d2463a614acc3ad08863ff617")
        self.assertEqual(payload["seed"], REQUEST_SEED)
        self.assertFalse(any("dspark" in key.lower() for key in payload))

    def test_cache_reuses_original_payload(self):
        self.assertEqual(case_plan("cache")[-1]["replay_of"], "cache_A")
        self.assertEqual(len(case_plan("concurrency")), 4)
        self.assertEqual(case_plan("contexts")[-1]["target_input_tokens"], 261632)

    def test_counter_response_must_be_exact_integer(self):
        for response in (exact(True), exact("256"), exact(256.0),
                         {"tokens": 256, "approximate": True}, {"tokens": 256},
                         exact(256, True), exact(-1)):
            with self.subTest(response=response), self.assertRaises(CalibrationError):
                calibrate_prompt(Counter(response), MODEL_ALIAS, SEED, "x", 256, 128)

    def test_probe_budget(self):
        counter = Counter(*(exact(1000) for _ in range(4)))
        with self.assertRaises(CalibrationError):
            calibrate_prompt(counter, MODEL_ALIAS, SEED, "x", 256, 128)
        self.assertEqual(len(counter.calls), 4)

    def test_context_boundaries(self):
        # Requested boundary fits, but actual tokenizer count can exceed it.
        calibrate_prompt(Counter(exact(256, 640)), MODEL_ALIAS, SEED, "x", 256, 128, 640)
        for counter, limit in ((Counter(exact(257)), 640), (Counter(exact(256, 639)), 262144)):
            with self.assertRaises(CalibrationError):
                calibrate_prompt(counter, MODEL_ALIAS, SEED, "x", 256, 128, limit)
        counter = Counter()
        with self.assertRaises(CalibrationError):
            calibrate_prompt(counter, MODEL_ALIAS, SEED, "x", 256, 128, 639)
        self.assertEqual(counter.calls, [])

    def test_case_specific_markers(self):
        self.assertNotEqual(synthetic_prompt(SEED, "short_1", 256)[1],
                            synthetic_prompt(SEED, "short_2", 256)[1])


if __name__ == "__main__":
    unittest.main()

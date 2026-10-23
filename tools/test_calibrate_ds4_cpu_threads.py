"""CPU-only regressions for DeepSeek calibration's engine thread metadata."""

import unittest

from tools import calibrate_ds4_tp as calibration
from tools.test_calibrate_ds4_tp import FakeEngine


def _measure_cpu_threads(
    info: dict[str, int], args: list[str] | None = None
) -> tuple[dict[str, object], list[list[str]], list[str]]:
    """Measure through the real calibration loop with controlled engine INFO."""
    starts: list[list[str]] = []
    messages: list[str] = []

    def decode_rate(tp: int, threads: int) -> float:
        return 100.0

    def prompt_rate(tp: int) -> float:
        return 900.0

    def start_engine(cfg: dict[str, object]) -> FakeEngine:
        engine = FakeEngine(cfg, decode_rate, prompt_rate, threads=12, vram=[24.0])
        engine.info = dict(info)
        starts.append(engine.args)
        return engine

    result = calibration.measure(
        {"args": args or ["--pack", "/fixture", "--tp", "2"]},
        [24.0],
        [[1, 2, 3]] * 12,
        [0] * 16,
        start_engine,
        say=messages.append,
    )
    return result, starts, messages


class CpuThreadMetadata(unittest.TestCase):
    """Check the calibration decisions and user messages, without GPU work."""

    def test_missing_thread_count_does_not_claim_all_experts_fit_in_vram(self) -> None:
        """Old engine INFO must warn about the unknown count, not residency."""
        result, starts, messages = _measure_cpu_threads({"cpu_experts": 100})
        self.assertEqual(len(starts), 1)
        self.assertEqual(result["args"], {"--tp": 2, "--cpu-threads": None})
        self.assertIn("thread count", "\n".join(messages))
        self.assertNotIn("VRAM holds them all", "\n".join(messages))

    def test_nonpositive_thread_count_does_not_claim_no_cpu_experts(self) -> None:
        """An unusable thread count must not change the reported residency."""
        for threads in (0, -1):
            with self.subTest(threads=threads):
                _, starts, messages = _measure_cpu_threads(
                    {"cpu_experts": 100, "cpu_threads": threads}
                )
                self.assertEqual(len(starts), 1)
                self.assertIn("thread count", "\n".join(messages))
                self.assertNotIn("VRAM holds them all", "\n".join(messages))

    def test_effective_thread_count_overrides_the_requested_thread_limit(self) -> None:
        """A requested ceiling of 64 must not hide an effective count of 12."""
        _, starts, _ = _measure_cpu_threads(
            {"cpu_experts": 100, "cpu_threads": 12},
            ["--pack", "/fixture", "--tp", "2", "--cpu-threads", "64"],
        )
        self.assertEqual(len(starts), 2)
        self.assertEqual(calibration.arg(starts[1], "--cpu-threads"), "9")

    def test_reported_thread_count_enables_the_cpu_sweep(self) -> None:
        """A 12-thread lane must try nine threads without a configured limit."""
        _, starts, _ = _measure_cpu_threads({"cpu_experts": 100, "cpu_threads": 12})
        self.assertEqual(len(starts), 2)
        self.assertEqual(calibration.arg(starts[1], "--cpu-threads"), "9")

    def test_explicit_thread_limit_remains_a_fallback_for_old_engines(self) -> None:
        """Old engines can still calibrate when the config supplies the count."""
        _, starts, _ = _measure_cpu_threads(
            {"cpu_experts": 100},
            ["--pack", "/fixture", "--tp", "2", "--cpu-threads", "12"],
        )
        self.assertEqual(len(starts), 2)
        self.assertEqual(calibration.arg(starts[1], "--cpu-threads"), "9")

    def test_zero_cpu_experts_skips_the_thread_sweep(self) -> None:
        """A genuinely GPU-resident model must keep the existing fast path."""
        _, starts, messages = _measure_cpu_threads(
            {"cpu_experts": 0, "cpu_threads": 12}
        )
        self.assertEqual(len(starts), 1)
        self.assertIn("VRAM holds them all", "\n".join(messages))

    def test_missing_expert_count_is_not_reported_as_zero_cpu_experts(self) -> None:
        """Missing telemetry is not evidence that all experts fit in VRAM."""
        _, starts, messages = _measure_cpu_threads({"cpu_threads": 12})
        self.assertEqual(len(starts), 1)
        self.assertIn("expert count", "\n".join(messages))
        self.assertNotIn("VRAM holds them all", "\n".join(messages))


if __name__ == "__main__":
    unittest.main()

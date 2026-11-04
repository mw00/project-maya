"""Calibration must compare effective tensor parallel groups, not requested widths."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve.server import StrataEngine
from tools import calibrate_ds4_tp as calibration
from tools.test_calibrate_ds4_tp import FakeEngine


class EffectiveTensorParallel(unittest.TestCase):
    """Inference is simulated; the real calibration loop owns the selection."""

    def measure_groups(
        self,
        groups: dict[int, list[int] | None],
        vram: list[float] | None = None,
    ) -> tuple[dict, list[FakeEngine], list[str]]:
        engines: list[FakeEngine] = []
        messages: list[str] = []

        def start_engine(cfg: dict) -> FakeEngine:
            engine = FakeEngine(
                cfg,
                lambda tp, threads: {2: 15.1, 4: 21.5, 5: 17.0}[tp],
                lambda tp: 500.0,
                cpu_experts=0,
            )
            widths = groups[engine.tp]
            engine.vram = (
                [{"dev": dev, "tp_n": width} for dev, width in enumerate(widths)]
                if widths is not None
                else []
            )
            engine.last = {}
            engines.append(engine)
            return engine

        vram = vram or [24.0] * 4
        result = calibration.measure(
            {"args": ["--tp", "2"], "gpu": list(range(len(vram)))},
            vram,
            [[1, 2, 3]] * 12,
            [0] * 16,
            start_engine,
            say=messages.append,
        )
        return result, engines, messages

    def test_fallback_duplicates_cannot_win_from_between_load_variation(self) -> None:
        result, engines, messages = self.measure_groups(
            {2: [1, 1, 1, 1], 4: [1, 1, 1, 1]}
        )
        self.assertEqual(result["args"]["--tp"], 2)
        self.assertEqual(result["report"]["effective_tp"], [1, 1, 1, 1])
        self.assertEqual(
            result["report"]["tp_groups"], {"2": [1, 1, 1, 1], "4": [1, 1, 1, 1]}
        )
        self.assertEqual(set(result["report"]["tp_sweep"]), {"2"})
        self.assertIn("4", result["report"]["tp_skipped"])
        self.assertEqual(engines[1].last, {}, "duplicate must close before inference")
        self.assertIn("effective TP per GPU: 1/1/1/1", "\n".join(messages))
        self.assertIn("same effective grouping as --tp 2", "\n".join(messages))

    def test_duplicate_of_an_alternative_is_not_measured_again(self) -> None:
        result, engines, _ = self.measure_groups(
            {2: [1, 1, 1, 1, 1], 4: [1, 2, 2, 2, 2], 5: [1, 2, 2, 2, 2]},
            vram=[24.0, 16.0, 16.0, 16.0, 16.0],
        )
        self.assertEqual(result["args"]["--tp"], 4)
        self.assertEqual(engines[2].last, {})
        self.assertIn(
            "same effective grouping as --tp 4", result["report"]["tp_skipped"]["5"]
        )

    def test_unknown_or_partial_candidate_grouping_cannot_win(self) -> None:
        for widths in (None, [1, 1, 1], [1, 1, 1, 0]):
            with self.subTest(widths=widths):
                result, engines, messages = self.measure_groups(
                    {2: [1, 1, 1, 1], 4: widths}
                )
                self.assertEqual(result["args"]["--tp"], 2)
                self.assertEqual(engines[1].last, {})
                self.assertIn("unknown", result["report"]["tp_skipped"]["4"])
                self.assertIn("unknown", "\n".join(messages))

    def test_unknown_baseline_grouping_keeps_its_requested_tp(self) -> None:
        result, engines, messages = self.measure_groups({2: None, 4: [4, 4, 4, 4]})
        self.assertEqual(result["args"]["--tp"], 2)
        self.assertIsNone(result["report"]["effective_tp"])
        self.assertEqual(engines[1].last, {})
        self.assertIn("unknown", "\n".join(messages))

    def test_distinct_mixed_groups_are_reported_and_compared(self) -> None:
        result, _, _ = self.measure_groups(
            {2: [1, 2, 2, 2, 2], 4: [1, 4, 4, 4, 4], 5: [5, 5, 5, 5, 5]},
            vram=[24.0, 16.0, 16.0, 16.0, 16.0],
        )
        self.assertEqual(result["args"]["--tp"], 4)
        self.assertEqual(result["report"]["effective_tp"], [1, 4, 4, 4, 4])
        self.assertEqual(result["report"]["tp_skipped"], {})
        self.assertEqual(set(result["report"]["tp_sweep"]), {"2", "4", "5"})


@unittest.skipIf(os.name == "nt", "a shell script stands in for the engine")
class TensorParallelProtocol(unittest.TestCase):
    """Cross the real process/VRAM parser boundary without GPU inference."""

    def test_serialized_fallback_is_recognized_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / "engine"
            exe.write_text(
                "#!/bin/sh\n"
                "echo 'INFO cpu_experts=9073 cpu_threads=12'\n"
                "echo 'VRAM dev=0 tp_n=1'\n"
                "echo 'VRAM dev=1 tp_n=1'\n"
                "echo 'VRAM dev=2 tp_n=1'\n"
                "echo 'VRAM dev=3 tp_n=1'\n"
                "echo 'READY 4096'\n"
                'while read -r command; do [ "$command" = QUIT ] && exit 0; done\n',
                encoding="utf-8",
            )
            exe.chmod(0o755)
            engines: list[StrataEngine] = []

            def start_engine(cfg: dict) -> StrataEngine:
                engine = StrataEngine(str(exe), cfg["args"])
                engines.append(engine)
                return engine

            try:
                with patch.object(
                    calibration.CAL,
                    "Session",
                    side_effect=AssertionError("duplicate topology reached inference"),
                ):
                    result = calibration.start_measure(
                        {"gpu": [0, 1, 2, 3]},
                        ["--tp", "4"],
                        [],
                        [],
                        start_engine,
                        seen_groups={(1, 1, 1, 1): 2},
                    )
                self.assertEqual(result["tp_groups"], [1, 1, 1, 1])
                self.assertEqual(result["info"]["cpu_experts"], 9073)
                self.assertEqual(result["info"]["cpu_threads"], 12)
                self.assertIn("same effective grouping as --tp 2", result["skipped"])
                self.assertEqual(engines[0].proc.returncode, 0)
            finally:
                for engine in engines:
                    if engine.proc.poll() is None:
                        engine.close()
                    if engine.proc.stdout is not None:
                        engine.proc.stdout.close()


if __name__ == "__main__":
    unittest.main()

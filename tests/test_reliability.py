"""Failure-path checks for preprocessing, local resources and diagnostics."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import matcher as vm


class ReliabilityTests(unittest.TestCase):
    def test_download_events_survive_failure_and_ignore_console_level(self):
        events = []
        previous_level = vm.MATCH_LOG.level
        vm.MATCH_LOG.setLevel(100)
        try:
            for filename in ("first.png", "second.png"):
                with patch.object(vm, "_match_shape_impl", side_effect=ValueError("test failure")):
                    with self.assertRaises(ValueError):
                        vm.match_shape("unused", source_name=filename, log_events=events)
                self.assertEqual(events[0]['filename'], filename)
                self.assertEqual([e['step'] for e in events],['01.input','error','09.finished'])
                self.assertEqual(len({e['run'] for e in events}),1)
                json.dumps(events)
                self.assertIsNone(vm._EVENTS.get())
            self.assertIsNone(vm._RUN.get())
        finally:
            vm.MATCH_LOG.setLevel(previous_level)

    def test_console_trace_records_steps_without_changing_ranking(self):
        signature = dict(endpoint_count=2, junction_count=0, loop_count=0, closed=False,
                         detected_threads=0, net_rotation=0., total_curvature=0., bends=[],
                         heading=np.zeros(10), heading_reverse=np.zeros(10))
        catalog = [dict(shape_id="synthetic", signature=signature, threads=0)]
        with patch.object(vm, "read_customer_image", return_value=np.zeros((2,2))), \
             patch.object(vm, "prepare_customer", return_value=None), \
             patch.object(vm, "build_signature", return_value=signature), \
             patch.object(vm, "load_or_build_cache", return_value=catalog), \
             self.assertLogs(vm.MATCH_LOG, level="INFO") as captured:
            _, results = vm.match_shape("unused", source_name="drawing.png")
        events = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(events[0]["filename"], "drawing.png")
        self.assertEqual(events[-1]["step"], "09.finished")
        score = next(e for e in events if e["step"] == "06.stage2.score")
        self.assertAlmostEqual(sum(score["components"].values()), results[0]["structural_dist"])
        self.assertTrue(any(e["step"] == "08.decision" for e in events))
        self.assertIsNone(vm._RUN.get())
        self.assertEqual(len({e['run'] for e in events}), 1)

    def test_failed_trace_clears_request_context(self):
        with patch.object(vm, "_match_shape_impl", side_effect=RuntimeError("test")), \
             self.assertLogs(vm.MATCH_LOG, level="ERROR") as captured:
            with self.assertRaises(RuntimeError):
                vm.match_shape("unused")
        self.assertIn('"error_type": "RuntimeError"', captured.output[0])
        self.assertIsNone(vm._RUN.get())

    def test_explicit_top_k_limits_output_after_full_reranking(self):
        signature = dict(endpoint_count=2, junction_count=0, loop_count=0,
                         detected_threads=0, net_rotation=0., total_curvature=0., bends=[])
        catalog = [dict(shape_id=str(index), signature=signature, threads=0) for index in range(4)]
        with patch.object(vm, "read_customer_image", return_value=np.zeros((1, 1))), \
             patch.object(vm, "prepare_customer", return_value=None), \
             patch.object(vm, "build_signature", return_value=signature), \
             patch.object(vm, "load_or_build_cache", return_value=catalog), \
             patch.object(vm, "compare_stage1", return_value=0.), \
             patch.object(vm, "compare_stage2", return_value=(0., 0., .98)) as rerank, \
             patch.dict(os.environ, {"BBS_MATCH_TRACE_DIR": ""}), \
             contextlib.redirect_stdout(io.StringIO()):
            stage1, final = vm.match_shape("unused", top_k=2)
        self.assertEqual(len(stage1), 4)
        self.assertEqual(len(final), 2)
        self.assertEqual(rerank.call_count, 4)

    def test_invalid_result_limit_fails_before_reading_input(self):
        with patch.object(vm, "read_customer_image") as read:
            for value in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    vm.match_shape("unused", top_k=value)
            read.assert_not_called()

    def test_database_and_cache_work_outside_project(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.object(vm, "DB_PATH", str(vm.PROJECT_ROOT / "demo_catalog.db")):
                    rows = vm.load_catalog()
                    self.assertTrue(all(Path(row[1]).is_file() for row in rows))
                    with contextlib.redirect_stdout(io.StringIO()):
                        catalog = vm.load_or_build_cache()
                self.assertEqual(len(catalog), len(rows))
                self.assertFalse(Path("catalog.db").exists())
            finally:
                os.chdir(previous)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.db"
            with patch.object(vm, "DB_PATH", str(missing)):
                with self.assertRaises(sqlite3.OperationalError):
                    vm.load_catalog()
            self.assertFalse(missing.exists())

    def test_failed_atomic_cache_publish_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cache.pkl"
            target.write_bytes(b"previous complete cache")
            with patch.object(vm, "CACHE_PATH", str(target)), \
                 patch.object(vm.os, "replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    vm.write_catalog_cache([])
            self.assertEqual(target.read_bytes(), b"previous complete cache")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_incomplete_cache_is_rebuilt_instead_of_used(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cache.pkl"
            with patch.object(vm, "CACHE_PATH", str(target)):
                vm.write_catalog_cache([])
                with patch.object(vm, "build_catalog_cache", return_value=["rebuilt"]) as rebuild, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(vm.load_or_build_cache(), ["rebuilt"])
                rebuild.assert_called_once()

    def test_preprocessing_falls_back_after_component_rejection(self):
        bad = np.ones((32, 32), dtype=np.uint8)
        good = np.eye(32, dtype=np.uint8) * 255
        def metrics(mask):
            return {"score": 9 if mask is bad else 8, "skeleton_pixels": 32}
        def decompose(mask):
            return mask, {"absolute_plausibility": mask is good}
        with patch.object(vm, "_customer_foreground_candidates", return_value={"first": bad, "second": good}), \
             patch.object(vm, "_prepare_customer_candidate", side_effect=lambda mask: mask), \
             patch.object(vm, "_customer_candidate_metrics", side_effect=metrics), \
             patch.object(vm, "_decompose_customer_foreground", side_effect=decompose), \
             patch.object(vm, "canonicalize_geometry", side_effect=lambda mask: mask):
            actual = vm.prepare_customer(np.zeros((32, 32, 3), dtype=np.uint8))
        np.testing.assert_array_equal(actual, good)

    def test_failed_match_does_not_save_input_and_preserves_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "temporary-upload.png"
            source.write_bytes(b"original uploaded bytes")
            with patch.dict(os.environ, {"BBS_MATCH_TRACE_DIR": str(root / "traces")}), \
                 patch.object(vm, "_match_shape_impl", side_effect=ValueError("original error")):
                with self.assertRaisesRegex(ValueError, "original error"):
                    vm.match_shape(str(source))
            self.assertFalse((root / "traces").exists())

    def test_trace_write_failure_does_not_break_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.png"
            source.write_bytes(b"input")
            blocked = Path(directory) / "not-a-directory"
            blocked.write_text("occupied")
            with patch.dict(os.environ, {"BBS_MATCH_TRACE_DIR": str(blocked)}), \
                 patch.object(vm, "_match_shape_impl", return_value=([], [])):
                self.assertEqual(vm.match_shape(str(source)), ([], []))


if __name__ == "__main__":
    unittest.main()

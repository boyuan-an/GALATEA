import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from galatea_recon.pipeline import hoidetr_contact as contact
from galatea_recon.pipeline.contact_cache import cached_contact, contact_inputs, manual_window


class TemporalTests(unittest.TestCase):
    def test_short_contact_and_clip_edges_are_preserved(self):
        for n in (5, 30, 35, 36, 169):
            with self.subTest(n=n):
                raw, windows, _ = contact.estimate_windows([1.] * n)
                self.assertEqual(windows, [[0, n - 1]])
                self.assertEqual(raw, windows)
        self.assertEqual(contact.estimate_windows([0.] * 16 + [1.] * 153)[1], [[16, 168]])

    def test_brief_detection_gap_and_separate_interactions(self):
        signal = [1.] * 10 + [0.] * 2 + [1.] * 10 + [0.] * 10 + [1.] * 10
        self.assertEqual(contact.estimate_windows(signal)[1], [[0, 21], [32, 41]])

    def test_explicit_trim_is_opt_in_and_invalid_options_fail(self):
        self.assertEqual(contact.estimate_windows([1.] * 60, start_trim=2, end_trim=3)[1], [[2, 56]])
        for kwargs in (dict(start_trim=-1), dict(end_trim=-1), dict(median_width=2),
                       dict(min_length=0), dict(max_gap=-1), dict(threshold=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                contact.estimate_windows([1.] * 30, **kwargs)


class ContactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="galatea-contact-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cam = self.root / "cam"
        (self.cam / "rgb").mkdir(parents=True)
        self.n = 8
        for i in range(self.n):
            cv2.imwrite(str(self.cam / "rgb" / f"{i:06d}.png"), np.full((32, 64, 3), i, np.uint8))
        self.masks = self.root / "masks.npz"
        mask = np.zeros((32, 64), bool)
        mask[10:30, 30:50] = True
        np.savez(self.masks, **{f"obj_{i}": mask for i in range(self.n)})
        self.hands = self.root / "wilor/wilor_hands.npz"
        self.hands.parent.mkdir()
        self.hand = dict(is_right=1., kp2d=np.linspace([35, 2], [50, 15], 21))
        values = np.empty(self.n, object)
        values[:] = [[self.hand] for _ in range(self.n)]
        np.savez(self.hands, hands=values, hand_observed=np.ones(self.n, bool))
        model_root = self.root / "model"
        for name in (contact.DEFAULT_CONFIG, contact.DEFAULT_CHECKPOINT, "projects/_base_/test.py"):
            path = model_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        self.args = contact.build_arg_parser().parse_args([
            "--cam-dir", str(self.cam), "--out", str(self.root / "hoidetr_contact.json"),
            "--target-masks", str(self.masks), "--target-hands", str(self.hands),
            "--max-frames", str(self.n), "--hoidetr-root", str(model_root), "--device", "cpu",
        ])
        self.detections = [
            dict(class_id=0, box=[35, 2, 50, 15], score=.9, query_idx=0),
            dict(class_id=1, box=[30, 10, 50, 30], score=.9, query_idx=1),
            dict(class_id=0, box=[1, 8, 5, 13], score=.99, query_idx=2),
            dict(class_id=1, box=[1, 1, 6, 6], score=.99, query_idx=3),
        ]
        names = ("mmcv", "torch", "mmdet", "mmdet.apis", "mmdet.datasets",
                 "mmdet.datasets.pipelines", "projects", "configs", "helpers")
        modules = {name: types.ModuleType(name) for name in names}
        ns = types.SimpleNamespace
        model = ns(cfg=ns(data=ns(test=ns(pipeline=[]))), query_head=None, eval=lambda: None)
        self.initialize = Mock(return_value=model)
        modules["mmdet.apis"].init_detector = self.initialize
        modules["mmdet.datasets.pipelines"].Compose = lambda x: x
        modules["projects"].models = ns()
        modules["configs"].CLASS_NAMES = ["hand", "firstobject", "secondobject"]
        helpers = modules["helpers"]
        helpers.compute_style = lambda *a: None
        helpers.draw_ui = lambda *a, **k: None
        helpers.find_interaction_branch = lambda *a: None
        self.inference = Mock(side_effect=lambda *a, **k: (self.detections, None))
        helpers.run_inference = self.inference
        self.enterContext(patch.dict(sys.modules, modules))
        self.addCleanup(setattr, sys, "path", sys.path[:])
        self.target_probability = .8
        self.pairs = self.enterContext(patch.object(contact, "_pair_probabilities", side_effect=self.predict_pairs))

    def predict_pairs(self, torch, branch, embeddings, hands, objects):
        return [(h, o, 1, self.target_probability if h["query_idx"] == 0 and o["query_idx"] == 1 else .999)
                for h in hands for o in objects]

    def run_contact(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return contact.estimate_contact(self.args)

    def test_only_target_pair_can_trigger_contact(self):
        report = self.run_contact()
        self.assertEqual(report["window"], "0-7")
        for frame in report["frames"]:
            self.assertEqual(frame["interaction_probability"], .8)
            self.assertEqual([(p["a"], p["b"]) for p in frame["hf"]], [(0, 1)])
        self.target_probability = .1
        self.args.force = True
        with self.assertRaisesRegex(RuntimeError, "no usable contact"):
            self.run_contact()
        self.assertFalse(Path(self.args.out).exists())

    def test_missing_target_never_falls_back_to_background(self):
        self.detections = self.detections[2:]
        with self.assertRaisesRegex(RuntimeError, "no usable contact"):
            self.run_contact()
        self.pairs.assert_not_called()

    def test_ambiguous_detection_is_not_chosen_by_contact_score(self):
        self.detections.append(dict(class_id=0, box=[35.1, 2, 50, 15], score=1., query_idx=4))
        with self.assertRaisesRegex(RuntimeError, "no usable contact"):
            self.run_contact()
        self.pairs.assert_not_called()

    def test_left_hands_and_interpolated_right_hands_are_not_targets(self):
        values = np.empty(self.n, object)
        left = dict(self.hand, is_right=0., kp2d=np.linspace([1, 8], [5, 13], 21))
        values[:] = [[left] for _ in range(self.n)]
        np.savez(self.hands, hands=values)
        targets = contact.load_targets(self.masks, self.hands, self.n)
        self.assertTrue(all(t["hand_box"] is None for t in targets))
        values[:] = [[left, self.hand] for _ in range(self.n)]
        observed = np.ones(self.n, bool)
        observed[3] = False
        np.savez(self.hands, hands=values, hand_observed=observed)
        targets = contact.load_targets(self.masks, self.hands, self.n)
        self.assertIsNone(targets[3]["hand_box"])
        self.assertEqual(targets[0]["hand_box"], [35., 2., 50., 15.])

    def test_unchanged_cache_reuses_and_force_recomputes(self):
        self.run_contact()
        self.run_contact()
        self.assertEqual(self.initialize.call_count, 1)
        self.args.force = True
        self.run_contact()
        self.assertEqual(self.initialize.call_count, 2)

    def test_input_parameter_model_and_target_changes_invalidate(self):
        self.run_contact()
        identity = contact_inputs(self.args)
        self.assertIsNotNone(cached_contact(self.args.out, identity))
        paths = [self.cam / "rgb/000000.png", self.masks, self.hands,
                 Path(self.args.hoidetr_root) / contact.DEFAULT_CONFIG,
                 Path(self.args.hoidetr_root) / contact.DEFAULT_CHECKPOINT,
                 Path(self.args.hoidetr_root) / "projects/_base_/test.py"]
        for path in paths:
            with self.subTest(path=path):
                original, stat = path.read_bytes(), path.stat()
                path.write_bytes(original + b"changed")
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertIsNone(cached_contact(self.args.out, contact_inputs(self.args)))
                path.write_bytes(original)
        for key, value in (("max_frames", 7), ("interaction_threshold", .9),
                           ("start_trim", 1), ("end_trim", 1), ("hand_match_iou", .8),
                           ("object_match_iou", .8), ("match_margin", .2), ("median_width", 3)):
            with self.subTest(parameter=key):
                original = getattr(self.args, key)
                setattr(self.args, key, value)
                self.assertIsNone(cached_contact(self.args.out, contact_inputs(self.args)))
                setattr(self.args, key, original)

    def test_legacy_incomplete_corrupt_and_edited_reports_are_not_reused(self):
        report = self.run_contact()
        identity = contact_inputs(self.args)
        path = Path(self.args.out)
        variants = [{"window": "0-7"}, dict(report, state="running"),
                    dict(report, frames=report["frames"][:-1]), dict(report, window="1-7"),
                    dict(report, error_frames=[3])]
        for value in variants:
            with self.subTest(keys=list(value)):
                path.write_text(json.dumps(value))
                self.assertIsNone(cached_contact(path, identity))
        path.write_text('{"window":')
        self.assertIsNone(cached_contact(path, identity))

    def test_failed_forced_refresh_invalidates_previous_success(self):
        self.run_contact()
        self.args.force = True
        self.initialize.side_effect = RuntimeError("interrupted model load")
        with self.assertRaisesRegex(RuntimeError, "interrupted model load"):
            self.run_contact()
        self.assertFalse(Path(self.args.out).exists())

    def shell_contact(self, force=False, threshold="0.5", window=""):
        repo = Path(__file__).resolve().parents[2]
        source = (repo / "scripts/run_pipeline_custom.sh").read_text()
        resolve = source[source.index('if [ -z "$WINDOWS" ] && [ -n "$SIDECAR" ]; then'):source.index('step "Step 0')]
        start = source.index('if [ -z "$WINDOWS" ] && [ "$CONTACT_MODE" = auto ]; then')
        auto = source[start:source.index("\nwrite_contact_labels\n", start) + 1]
        harness = '''set -eo pipefail
step() { :; }; use() { :; }; skip() { echo CACHE_HIT; }; ok() { :; }
runpy() { printf 'RECOMPUTE'; printf ' <%s>' "$@"; printf '\\n'; }
''' + resolve + auto + '\nprintf "RESULT=%s SOURCE=%s\\n" "$WINDOWS" "$WINDOW_SOURCE"\n'
        env = dict(os.environ, CAM=str(self.cam), B=str(self.root), MAXF=str(self.n),
                   W=str(self.hands.parent), ACTIVE_SAM2_MASKS=str(self.masks),
                   HOIDETR_ROOT=self.args.hoidetr_root, HOIDETR_ENV="mock", PRODUCE_QC="false",
                   FORCE=str(force).lower(), CONTACT_MODE="auto", WINDOWS=window, WINDOW_SOURCE="cli-manual" if window else "",
                   SIDECAR=str(self.cam / "contact.yaml"), HOIDETR_INTERACTION_THR=threshold,
                   PYTHONPATH=str(repo), PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
        return subprocess.run(["bash", "-c", harness], cwd=repo, env=env,
                              text=True, capture_output=True, check=True).stdout

    def test_pipeline_auto_sidecar_cannot_bypass_force_or_changed_parameters(self):
        self.run_contact()
        (self.cam / "contact.yaml").write_text('source: hoidetr-auto\nwindow: "0-7"\nauto_window: "0-7"\n')
        self.assertIn("CACHE_HIT", self.shell_contact())
        forced = self.shell_contact(force=True)
        self.assertIn("RECOMPUTE", forced)
        self.assertIn("<--force>", forced)
        self.assertIn("<--target-masks>", forced)
        self.assertIn("<--target-hands>", forced)
        self.assertIn("RECOMPUTE", self.shell_contact(threshold="0.9"))
        self.assertNotIn("RECOMPUTE", self.shell_contact(force=True, window="1-6"))


class SidecarTests(unittest.TestCase):
    def test_manual_and_edited_auto_overrides_with_legacy_migration(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "contact.yaml"
            self.assertEqual(manual_window(path), "")
            for value, expected in (
                ({"window": "1-9"}, "1-9"),
                ({"source": "manual", "window": "1-9"}, "1-9"),
                ({"source": "hoidetr-auto", "window": "1-9"}, ""),
                ({"source": "hoidetr-auto", "window": "1-9", "auto_window": "1-9"}, ""),
                ({"source": "hoidetr-auto", "window": "2-8", "auto_window": "1-9"}, "2-8"),
            ):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    self.assertEqual(manual_window(path), expected)
            audit = Path(d) / "audit.json"
            audit.write_text('{"window": "2-8"}')
            path.write_text(json.dumps(dict(source="hoidetr-auto", window="1-9", audit=str(audit))))
            self.assertEqual(manual_window(path), "")


if __name__ == "__main__":
    unittest.main()

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from galatea_recon.pipeline import hand_obj_optimizer as joint
from galatea_recon.pipeline import wilor_runner as wilor
from galatea_recon.pipeline.frame0_cache import load_predictions, save_predictions
from galatea_recon.pipeline.object_quality import atomic_json
from galatea_recon.pipeline.perception_cache import atomic_numpy
from galatea_recon.pipeline.trajectory_cache import (
    hand_cache_complete, optimization_cache_complete, optimization_outputs,
)


class TrajectoryCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="galatea-trajectory-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cam, self.hands, self.depth = [self.root / name for name in ("cam", "hands", "depth")]
        for directory in (self.cam / "rgb", self.hands, self.depth, self.root / "joint"):
            directory.mkdir(parents=True)
        for f in range(3):
            cv2.imwrite(str(self.cam / "rgb" / f"{f:06d}.png"), np.full((6, 8, 3), 40 + f, np.uint8))
            np.save(self.depth / f"{f:06d}.npy", np.ones((6, 8), np.float32))
        (self.cam / "cam_intrinsics.txt").write_text("10 10 4 3 8 6\n")
        (self.root / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        np.save(self.root / "poses.npy", np.tile(np.eye(4), (3, 1, 1)))
        np.savez(self.root / "masks.npz", obj_0=np.ones((6, 8), bool))
        (self.root / "labels.json").write_text('{"0": []}')
        self.hand_args = wilor.build_arg_parser().parse_args([
            "--cam_dir", str(self.cam), "--out", str(self.hands), "--max_frames", "3",
            "--hand-side", "right", "--fill-missing", "--clean-jumps"])
        self.vertices = np.tile([0, 0, 1], (778, 1)).astype(np.float32)
        self.joints = np.tile([0, 0, 1], (21, 1)).astype(np.float32)
        wp = {"pred_keypoints_3d": self.joints[None],
              "pred_keypoints_2d": np.full((1, 21, 2), 3, np.float32),
              "pred_vertices": self.vertices[None], "global_orient": np.zeros((1, 1, 3)),
              "hand_pose": np.zeros((1, 15, 3)), "betas": np.zeros((1, 10))}
        self.prediction = {"is_right": 1.0, "wilor_preds": wp}
        self.infer = self.enterContext(patch.object(wilor, "predict_requested_hands",
                                                    return_value=([self.prediction], 0)))
        self.enterContext(patch.object(wilor, "place_hand_h2o", return_value=(
            self.vertices, self.joints, np.eye(3), np.array([0, 0, 1]))))
        self.enterContext(patch.object(wilor, "_overlay"))
        self.optimize = self.enterContext(patch.object(joint, "_run", side_effect=self.fake_optimization))

    def run_hands(self):
        wilor.run(self.hand_args, pipe=object())

    def joint_args(self):
        return joint.build_arg_parser().parse_args([
            "--cam_dir", str(self.cam), "--mesh", str(self.root / "mesh.obj"),
            "--fp_poses", str(self.root / "poses.npy"), "--wilor_hands", str(self.hands / "wilor_hands.npz"),
            "--depth_dir", str(self.depth), "--sam2_masks", str(self.root / "masks.npz"),
            "--out", str(self.root / "joint/gr_hsz.npy"), "--max_frames", "3", "--iters", "150",
            "--full_objective", "--use_mano", "--hand_silhouette_z", "--fill-missing", "--early-stop",
            "--seg_labels", str(self.root / "labels.json"), "--manual_json", str(self.root / "labels.json")])

    def fake_optimization(self, args, report):
        n = min(args.max_frames, 3)
        paths = optimization_outputs(args)
        atomic_numpy(paths[".npy"], np.tile(np.eye(4), (n, 1, 1)))
        atomic_numpy(paths["_hand.npy"], np.ones((n, 21, 3), np.float32))
        atomic_numpy(paths["_handverts.npy"], np.ones((n, 778, 3), np.float32))
        if "observations" in paths:
            atomic_numpy(paths["observations"], {name: np.ones(n, bool) for name in (
                "object_observed", "hand_observed", "object_filled", "hand_filled")})
        if "faces" in paths:
            atomic_numpy(paths["faces"], np.array([[0, 1, 2]], np.int64))
        if "early_stop" in paths:
            atomic_json(paths["early_stop"], {"stages": []})

    def test_hand_reuse_and_parameter_changes(self):
        self.run_hands()
        self.assertTrue(hand_cache_complete(self.hand_args))
        for key, value in {"max_frames": 2, "hand_side": "all", "hand_conf": .5,
                           "clean_jumps": False, "fill_missing": False, "max_repair_span": 4,
                           "jump_position_mm": 100., "jump_rotation_deg": 90.}.items():
            with self.subTest(key=key):
                args = copy.deepcopy(self.hand_args)
                setattr(args, key, value)
                self.assertFalse(hand_cache_complete(args))

    def test_extended_or_changed_rgb_and_camera_invalidate_hands(self):
        self.hand_args.max_frames = 2
        self.run_hands()
        self.hand_args.max_frames = 3
        self.assertFalse(hand_cache_complete(self.hand_args))
        self.run_hands()
        path = self.cam / "rgb/000001.png"
        stat = path.stat()
        cv2.imwrite(str(path), np.full((6, 8, 3), 180, np.uint8))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(hand_cache_complete(self.hand_args))
        self.run_hands()
        (self.cam / "cam_intrinsics.txt").write_text("11 11 4 3 8 6\n")
        self.assertFalse(hand_cache_complete(self.hand_args))

    def test_legacy_missing_and_corrupt_hand_outputs(self):
        self.run_hands()
        (self.hands / "wilor_complete.json").unlink()
        (self.hands / "gap_completion_v1").touch()
        self.assertFalse(hand_cache_complete(self.hand_args))
        for name in ("wilor_hands.npz", "wilor_hands_raw.npz", "wilor_cleanup.json"):
            with self.subTest(name=name):
                self.run_hands()
                (self.hands / name).write_bytes(b"corrupt")
                self.assertFalse(hand_cache_complete(self.hand_args))

    def test_interrupted_hand_refresh_invalidates_completion(self):
        self.run_hands()
        self.infer.side_effect = RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_hands()
        self.assertFalse(hand_cache_complete(self.hand_args))
        self.assertFalse((self.hands / "wilor_complete.json").exists())

    def test_frame0_predictions_require_matching_image_and_confidence(self):
        path = self.root / "frame0.npz"
        rgb = np.zeros((6, 8, 3), np.uint8)
        save_predictions(path, [self.prediction], rgb)
        self.assertEqual(len(load_predictions(path, rgb)), 1)
        self.assertIsNone(load_predictions(path, rgb + 1))
        self.assertIsNone(load_predictions(path, rgb, .5))
        np.savez(path, predictions=np.array([self.prediction], dtype=object))
        self.assertIsNone(load_predictions(path, rgb))

    def test_stale_frame0_is_not_reused_by_runner(self):
        path = self.root / "frame0.npz"
        save_predictions(path, [], np.zeros((6, 8, 3), np.uint8))
        self.hand_args.frame0_predictions = str(path)
        self.run_hands()
        self.assertEqual(self.infer.call_count, 3)
        self.assertTrue(hand_cache_complete(self.hand_args))
        save_predictions(path, [], np.ones((6, 8, 3), np.uint8))
        self.assertFalse(hand_cache_complete(self.hand_args))

    def test_optimizer_all_numeric_and_early_stop_parameters_invalidate(self):
        self.run_hands()
        args = self.joint_args()
        joint.run(args)
        self.assertTrue(optimization_cache_complete(args))
        keys = [key for key, value in vars(args).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)]
        keys += ["early_stop", "fill_missing", "unfreeze", "no_contact_boundary_guard"]
        for key in keys:
            with self.subTest(key=key):
                changed = copy.deepcopy(args)
                value = getattr(changed, key)
                setattr(changed, key, not value if isinstance(value, bool) else value + 1)
                self.assertFalse(optimization_cache_complete(changed))
        checking = copy.deepcopy(args)
        checking.check_cache = True
        self.assertTrue(optimization_cache_complete(checking))

    def test_contact_iters_environment_participates_in_identity(self):
        self.run_hands()
        with patch.dict(os.environ, CONTACT_ITERS="200"):
            joint.run(self.joint_args())
        with patch.dict(os.environ, CONTACT_ITERS="201"):
            self.assertFalse(optimization_cache_complete(self.joint_args()))

    def test_optimizer_inputs_invalidate_even_with_preserved_mtime(self):
        self.run_hands()
        args = self.joint_args()
        joint.run(args)
        for path in (self.root / "mesh.obj", self.root / "poses.npy", self.root / "labels.json",
                     self.hands / "wilor_hands.npz", self.depth / "000001.npy"):
            with self.subTest(path=path.name):
                old, stat = path.read_bytes(), path.stat()
                path.write_bytes(old + b"\n")
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertFalse(optimization_cache_complete(args))
                path.write_bytes(old)
        self.assertTrue(optimization_cache_complete(args))

    def test_optimizer_output_damage_and_interrupted_run(self):
        self.run_hands()
        args = self.joint_args()
        joint.run(args)
        (self.root / "joint/gr_hsz_hand.npy").write_bytes(b"partial")
        self.assertFalse(optimization_cache_complete(args))
        joint.run(args)
        self.optimize.side_effect = RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            joint.run(args)
        self.assertFalse(optimization_cache_complete(args))

    def test_optimizer_does_not_publish_partial_outputs(self):
        self.run_hands()
        args = self.joint_args()
        self.optimize.side_effect = lambda args, report: None
        with self.assertRaises(FileNotFoundError):
            joint.run(args)
        self.assertFalse(optimization_cache_complete(args))

    def test_hand_cli_cache_check(self):
        self.run_hands()
        base = [sys.executable, "-m", "galatea_recon.pipeline.wilor_runner", "--cam_dir", str(self.cam),
                "--out", str(self.hands), "--hand-side", "right", "--fill-missing", "--clean-jumps",
                "--check-cache", "--max_frames"]
        self.assertEqual(subprocess.run(base + ["3"], capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(base + ["10"], capture_output=True).returncode, 1)


if __name__ == "__main__":
    unittest.main()

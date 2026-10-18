import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from galatea_recon.pipeline import export_depth, sam_masks
from galatea_recon.pipeline.h2o_loader import H2OSequence
from galatea_recon.pipeline.perception_cache import (
    atomic_numpy, depth_cache_complete, depth_identity, load_mask_cache, mask_identity,
)
from galatea_recon.pipeline.step34_calibrate import get_sam2_masks


class PerceptionCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="galatea-cache-test-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.cam, self.objects, self.out = root / "cam", root / "objects", root / "out"
        for path in (self.cam / "rgb", self.cam / "depth", self.cam / "obj_pose_rt",
                     self.objects / "object", self.out):
            path.mkdir(parents=True)
        self.shape = (6, 8)
        for i in range(3):
            cv2.imwrite(str(self.cam / "rgb" / f"{i:06d}.png"),
                        np.full((*self.shape, 3), 20 + i, np.uint8))
        cv2.imwrite(str(self.cam / "depth/000000.png"), np.full(self.shape, 1000, np.uint16))
        (self.cam / "cam_intrinsics.txt").write_text("10 10 4 3 8 6\n")
        (self.cam / "obj_name.txt").write_text("object\n")
        (self.cam / "obj_noun.txt").write_text("bottle\n")
        np.savetxt(self.cam / "obj_pose_rt/000000.txt", np.r_[0, np.eye(4).ravel()])
        (self.objects / "object/object.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        self.mask_model = self.enterContext(patch(
            "galatea_recon.pipeline.sam_masks.compute_sam2_masks", side_effect=self.compute_masks))
        self.depth_model = self.enterContext(patch(
            "galatea_recon.adapters.depth.est_depth", side_effect=lambda paths, **kw: [
                torch.ones(self.shape, dtype=torch.float32) for _ in paths]))

    def seq(self):
        return H2OSequence(str(self.cam), str(self.objects))

    def compute_masks(self, *args, n_frames, **kwargs):
        return self.seq(), {f: {0: np.ones(self.shape, bool)} for f in range(n_frames)}

    def masks(self, n=3, **kwargs):
        return get_sam2_masks(str(self.cam), str(self.objects), str(self.out), n - 1,
                              1000.0, "cpu", **kwargs)

    def export(self, **kwargs):
        export_depth.run(str(self.cam), str(self.objects), str(self.out), device="cpu", **kwargs)

    def complete(self, n=3):
        return depth_cache_complete(self.out, depth_identity(self.seq(), n))

    def shell_stage(self, force=False):
        repo = Path(__file__).resolve().parents[2]
        source = (repo / "scripts/run_pipeline_custom.sh").read_text()
        block = source[source.index('step "Step 1/10'):source.index('if [ "$SAM_MASK_VIZ" = true ]; then')]
        harness = '''set -eo pipefail
step() { :; }
use() { :; }
skip() { echo SKIPPED; }
ok() { :; }
runpy() { printf 'RECOMPUTE'; printf ' <%s>' "$@"; printf '\\n'; }
''' + block
        env = dict(os.environ, CAM=str(self.cam), OBJROOT=str(self.objects),
                   DEPTH=str(self.out), MAXF="3", FORCE=str(force).lower(),
                   PYTHONPATH=str(repo), PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
        return subprocess.run(["bash", "-c", harness], env=env, cwd=repo,
                              capture_output=True, text=True, check=True).stdout

    def test_mask_hit_and_force_refresh(self):
        self.masks()
        self.masks()
        self.assertEqual(self.mask_model.call_count, 1)
        self.masks(force=True)
        self.assertEqual(self.mask_model.call_count, 2)

    def test_legacy_short_and_corrupt_masks_recompute(self):
        np.savez(self.out / "sam2_masks.npz", fg_0=np.ones(self.shape, bool))
        self.masks(n=1)
        fg, obj = self.masks(n=3)
        self.assertEqual(set(fg), {0, 1, 2})
        self.assertEqual(set(obj), {0, 1, 2})
        (self.out / "sam2_masks.npz").write_bytes(b"truncated")
        self.masks()
        self.assertEqual(self.mask_model.call_count, 3)

    def test_changed_input_invalidates_masks(self):
        self.masks()
        for path in ("rgb/000001.png", "depth/000000.png", "obj_noun.txt", "cam_intrinsics.txt"):
            with self.subTest(path=path):
                p = self.cam / path
                stat = p.stat()
                if path.startswith("rgb/"):
                    cv2.imwrite(str(p), np.full((*self.shape, 3), 180, np.uint8))
                elif path.startswith("depth/"):
                    cv2.imwrite(str(p), np.full(self.shape, 1200, np.uint16))
                elif path == "obj_noun.txt":
                    p.write_text("cup\n")
                else:
                    p.write_text("11 11 4 3 8 6\n")
                os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                calls = self.mask_model.call_count
                self.masks()
                self.assertEqual(self.mask_model.call_count, calls + 1)

    def test_failed_mask_refresh_invalidates_old_marker(self):
        self.masks()
        self.mask_model.side_effect = RuntimeError("interrupted SAM")
        with self.assertRaisesRegex(RuntimeError, "interrupted SAM"):
            self.masks(force=True)
        self.assertIsNone(load_mask_cache(self.out, mask_identity(self.seq(), 3, 1000.0)))
        self.assertFalse((self.out / "sam2_masks.json").exists())

    def test_incomplete_model_output_is_not_published(self):
        self.mask_model.side_effect = lambda *args, **kwargs: (
            self.seq(), {0: {0: np.ones(self.shape, bool)}})
        with self.assertRaisesRegex(ValueError, "complete mask timeline"):
            self.masks()
        self.assertFalse((self.out / "sam2_masks.json").exists())

    def test_export_force_reaches_sam_and_cache_is_reusable(self):
        self.export()
        self.assertTrue(self.complete())
        self.assertIn("SKIPPED", self.shell_stage())
        self.export(force=True)
        self.assertEqual(self.mask_model.call_count, 2)
        self.assertEqual(self.depth_model.call_count, 2)
        self.assertTrue(self.complete())
        forced = self.shell_stage(force=True)
        self.assertIn("RECOMPUTE", forced)
        self.assertIn("<--force>", forced)

    def test_one_depth_file_does_not_skip_export(self):
        atomic_numpy(self.out / "moge_cal/000000.npy", np.ones(self.shape, np.float32))
        self.assertFalse(self.complete())
        self.assertIn("RECOMPUTE", self.shell_stage())
        self.export()
        self.assertTrue(self.complete())

    def test_missing_truncated_nonfinite_and_wrong_shape_depth_are_repaired(self):
        self.export()
        for kind in ("missing", "truncated", "nan", "shape", "raw_missing", "wrong_format"):
            with self.subTest(kind=kind):
                path = self.out / "moge_cal/000001.npy"
                if kind == "missing":
                    path.unlink()
                elif kind == "raw_missing":
                    (self.out / "moge_raw/000001.npy").unlink()
                elif kind == "truncated":
                    path.write_bytes(b"\x93NUMPY")
                elif kind == "nan":
                    np.save(path, np.full(self.shape, np.nan))
                elif kind == "wrong_format":
                    with path.open("wb") as stream:
                        np.savez(stream, array=np.ones(self.shape))
                else:
                    np.save(path, np.ones((1, 1)))
                self.assertFalse(self.complete())
                self.export()
                self.assertTrue(self.complete())
        self.assertEqual(self.mask_model.call_count, 1)

    def test_interrupted_refresh_cannot_reuse_previous_completion(self):
        self.export()
        original = export_depth.atomic_numpy
        writes = 0

        def interrupted(path, value):
            nonlocal writes
            writes += 1
            if writes == 3:
                raise OSError("simulated disk failure")
            original(path, value)

        with patch.object(export_depth, "atomic_numpy", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.export(force=True)
        self.assertFalse((self.out / "depth_complete.json").exists())
        self.assertFalse(self.complete())
        self.assertIn("RECOMPUTE", self.shell_stage())
        self.export()
        self.assertTrue(self.complete())

    def test_changed_parameters_or_missing_manifest_invalidate_depth(self):
        self.export()
        for inputs in (depth_identity(self.seq(), 2), depth_identity(self.seq(), 3, trim_pct=0.1),
                       depth_identity(self.seq(), 3, depth_scale=500.0)):
            self.assertFalse(depth_cache_complete(self.out, inputs))
        for content in ("{", "null", "[]"):
            (self.out / "depth_complete.json").write_text(content)
            self.assertFalse(self.complete())
        (self.out / "depth_complete.json").unlink()
        self.assertFalse(self.complete())

    def test_nonfinite_model_depth_does_not_publish_completion(self):
        self.depth_model.side_effect = lambda paths, **kw: [torch.full(self.shape, float("nan")) for _ in paths]
        with self.assertRaisesRegex(ValueError, "depth export incomplete"):
            self.export()
        self.assertFalse(self.complete())
        self.assertFalse((self.out / "depth_complete.json").exists())

    def test_jpeg_inputs_refresh_and_old_tail_is_removed(self):
        jpg = self.out / "jpg"
        sam_masks._write_jpg_frames(self.seq(), str(jpg))
        cv2.imwrite(str(self.cam / "rgb/000000.png"), np.full((*self.shape, 3), 200, np.uint8))
        sam_masks._write_jpg_frames(self.seq(), str(jpg), n_frames=1)
        self.assertEqual([p.name for p in jpg.glob("*.jpg")], ["00000.jpg"])
        self.assertGreater(cv2.imread(str(jpg / "00000.jpg")).mean(), 190)

    def test_atomic_write_failure_keeps_previous_array_and_cleans_temp(self):
        path = self.out / "array.npy"
        atomic_numpy(path, np.ones(self.shape))
        with patch("galatea_recon.pipeline.perception_cache.np.save", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                atomic_numpy(path, np.zeros(self.shape))
        np.testing.assert_array_equal(np.load(path), np.ones(self.shape))
        self.assertEqual(list(self.out.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()

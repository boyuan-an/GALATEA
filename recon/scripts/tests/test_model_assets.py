import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from galatea_recon.adapters.wilor_assets import ASSETS, mano_right_path, wilor_asset_dir


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "wilor"
        self.enterContext(patch.dict(os.environ, {"WILOR_CACHE_DIR": str(self.assets)}))

    def place(self, names=ASSETS):
        directory = self.assets / "pretrained_models"
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / name).write_bytes(b"local test asset")
        return directory

    def test_missing_assets_explain_official_acquisition_without_writes(self):
        with self.assertRaises(FileNotFoundError) as raised:
            wilor_asset_dir()
        message = str(raised.exception)
        for name in ASSETS:
            self.assertIn(name, message)
        self.assertIn("https://mano.is.tue.mpg.de/", message)
        self.assertIn("https://github.com/rolpotamias/WiLoR", message)
        self.assertIn("docs/model-assets.md", message)
        self.assertFalse(self.assets.exists())

    def test_local_files_are_reused_without_modification(self):
        directory = self.place()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        self.assertEqual(wilor_asset_dir(), str(self.assets))
        self.assertEqual(mano_right_path(), str(directory / "MANO_RIGHT.pkl"))
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        self.assertEqual(before, after)

    def test_empty_asset_is_rejected_and_preserved(self):
        directory = self.place()
        model = directory / "MANO_RIGHT.pkl"
        model.write_bytes(b"")
        with self.assertRaisesRegex(FileNotFoundError, "MANO_RIGHT.pkl"):
            wilor_asset_dir()
        self.assertTrue(model.is_file())
        self.assertEqual(model.stat().st_size, 0)

    def test_missing_mano_is_not_taken_from_installed_package(self):
        self.place(tuple(name for name in ASSETS if name != "MANO_RIGHT.pkl"))
        legacy = self.root / "site-packages/wilor_mini/pretrained_models"
        legacy.mkdir(parents=True)
        (legacy / "MANO_RIGHT.pkl").write_bytes(b"legacy asset")
        spec = SimpleNamespace(origin=str(legacy.parent / "__init__.py"))
        with patch("importlib.util.find_spec", return_value=spec), self.assertRaises(FileNotFoundError):
            wilor_asset_dir()
        self.assertFalse((self.assets / "pretrained_models/MANO_RIGHT.pkl").exists())

    def test_user_selected_symlinks_are_supported(self):
        directory = self.place(tuple(name for name in ASSETS if name != "MANO_RIGHT.pkl"))
        model = self.root / "licensed-mano.pkl"
        model.write_bytes(b"local test asset")
        (directory / "MANO_RIGHT.pkl").symlink_to(model)
        self.assertEqual(wilor_asset_dir(), str(self.assets))
        model.unlink()
        with self.assertRaises(FileNotFoundError):
            wilor_asset_dir()

    def test_optimizer_requires_mano_without_unrelated_inference_weights(self):
        directory = self.place(("MANO_RIGHT.pkl",))
        self.assertEqual(mano_right_path(), str(directory / "MANO_RIGHT.pkl"))
        with self.assertRaises(FileNotFoundError):
            wilor_asset_dir()

    def test_shared_cache_and_explicit_override(self):
        self.place()
        env = {"CACHE_ROOT": str(self.root), "XDG_CACHE_HOME": str(self.root / "other")}
        with patch.dict(os.environ, env):
            os.environ.pop("WILOR_CACHE_DIR")
            self.assertEqual(wilor_asset_dir(), str(self.assets))
            os.environ.pop("CACHE_ROOT")
            os.environ["XDG_CACHE_HOME"] = str(self.root)
            self.assertEqual(wilor_asset_dir(), str(self.assets))
            os.environ["WILOR_CACHE_DIR"] = str(self.root / "explicit-missing")
            with self.assertRaisesRegex(FileNotFoundError, "explicit-missing"):
                wilor_asset_dir()

    def test_every_pipeline_entry_point_stops_before_upstream_constructor(self):
        from galatea_recon.pipeline.sam_masks import load_wilor_pipeline as mask_loader
        from galatea_recon.pipeline.wilor_runner import load_wilor_pipeline as hand_loader

        constructor = Mock(side_effect=AssertionError("Upstream downloader must not be reached"))
        module = SimpleNamespace(WiLorHandPose3dEstimationPipeline=constructor)
        name = "wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline"
        with patch.dict("sys.modules", {name: module}):
            for loader, argument in ((mask_loader, "cpu"), (hand_loader, "cpu")):
                with self.subTest(loader=loader.__module__), self.assertRaises(FileNotFoundError):
                    loader(argument)
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()

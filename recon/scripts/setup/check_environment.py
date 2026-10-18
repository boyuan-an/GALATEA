import argparse
import ctypes
import importlib
import os
import sys
from pathlib import Path


def check(component, root, skip_weights=False, gpu=False):
    import torch
    import torchvision

    cuda128_stack = torch.__version__ == "2.7.1+cu128"
    factory_python = (3, 10) if cuda128_stack else (3, 8)
    expected_python = (3, 11) if component == "main" else factory_python
    if sys.version_info[:2] != expected_python:
        raise RuntimeError("Expected Python %s, got %s" % (expected_python, sys.version))
    expected_torch, expected_tv = (
        ("2.5.1+cu121", "0.20.1+cu121") if component == "main" else ("2.4.1+cu121", "0.19.1+cu121")
    )
    if cuda128_stack:
        expected_torch, expected_tv = "2.7.1+cu128", "0.22.1+cu128"
    if (torch.__version__, torchvision.__version__) != (expected_torch, expected_tv):
        raise RuntimeError(
            "Unexpected torch/torchvision: %s / %s" % (torch.__version__, torchvision.__version__)
        )
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    modules = ["pytorch3d._C", "cv2", "trimesh", "open3d", "scipy"]
    sys.path.insert(0, str(root))
    if component == "main":
        try:
            ctypes.CDLL("libXrender.so.1")
        except OSError as error:
            raise RuntimeError(
                "Missing libXrender: install libxrender1 on Ubuntu, including headless servers. "
                "See docs/installation.md for OS prerequisites."
            ) from error
        modules += [
            "pyrender",
            "sam2",
            "smplx",
            "chumpy",
            "dill",
            "moge.model.v2",
            "transformers.models.grounding_dino.modeling_grounding_dino",
            "wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline",
            "galatea_recon.pipeline.hand_obj_optimizer",
        ]
        if not skip_weights:
            from galatea_recon.adapters.wilor_assets import wilor_asset_dir

            wilor_asset_dir()
    else:
        fp = root / "imports/FoundationPose"
        sys.path[:0] = [str(fp), str(fp / "mycpp/build")]
        modules += [
            "bundlesdf.mycuda.common",
            "gridencoder",
            "mycpp",
            "nvdiffrast.torch",
            "estimater",
        ]
        if not skip_weights:
            for stamp in ("2024-01-11-20-02-45", "2023-10-28-18-33-37"):
                for name in ("model_best.pth", "config.yml"):
                    path = fp / "weights" / stamp / name
                    if not path.is_file() or path.stat().st_size < (
                        1000000 if name.endswith(".pth") else 1
                    ):
                        raise RuntimeError(
                            "Missing/incomplete FoundationPose asset: %s. "
                            "Follow https://github.com/NVlabs/FoundationPose#data-prepare "
                            "and docs/model-assets.md for the original download and placement." % path
                        )
    for name in modules:
        importlib.import_module(name)
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("No allocated/visible CUDA GPU; allocate one before --gpu checks.")
        from pytorch3d.ops import knn_points

        points = torch.rand(1, 8, 3, device="cuda")
        result = knn_points(points, points).dists
        if not torch.isfinite(result).all():
            raise RuntimeError("PyTorch3D GPU operation returned invalid values")
        if component == "main":
            import pyrender

            renderer = pyrender.OffscreenRenderer(16, 16)
            renderer.delete()
        if component == "factory":
            import nvdiffrast.torch as dr

            pos = torch.tensor(
                [[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]]],
                dtype=torch.float32,
                device="cuda",
            )
            tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
            raster, _ = dr.rasterize(dr.RasterizeCudaContext(), pos, tri, resolution=[16, 16])
            if not (raster[..., 3] > 0).any():
                raise RuntimeError("nvdiffrast GPU rasterizer produced no triangle")
        torch.cuda.synchronize()
        print("GPU:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    print(
        "PASS %s: Python %s; torch %s; imports/native extensions%s"
        % (component, sys.version.split()[0], torch.__version__, "; GPU smoke" if gpu else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check one reconstruction environment without downloading or modifying it.")
    parser.add_argument("component", choices=["main", "factory"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    try:
        check(args.component, args.root, args.skip_weights, args.gpu)
    except Exception as error:
        print("FAIL %s: %s: %s" % (args.component, type(error).__name__, error), file=sys.stderr)
        sys.exit(1)

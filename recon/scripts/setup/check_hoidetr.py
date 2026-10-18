import argparse
import sys
from pathlib import Path


def check(root, skip_weights=False, gpu=False):
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Contact detection requires the new Python 3.10 environment codetr-cu121"
        )
    sys.path[:0] = [str(root), str(root / "demo")]
    import torch
    import torchvision
    import mmcv
    from mmcv.ops import nms
    from mmdet.apis import init_detector
    from projects import models
    from helpers import find_interaction_branch, run_inference

    expected = ("2.4.1+cu121", "0.19.1+cu121", "1.7.2")
    if torch.__version__ == "2.7.1+cu128":
        expected = ("2.7.1+cu128", "0.22.1+cu128", "1.7.2")
    versions = (torch.__version__, torchvision.__version__, mmcv.__version__)
    if versions != expected:
        raise RuntimeError("Expected torch/torchvision/mmcv %s, got %s" % (expected, versions))
    config = (
        root
        / "projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py"
    )
    if not config.is_file():
        raise FileNotFoundError(config)
    checkpoint = root / "checkpoints/epoch_5.pth"
    if not skip_weights and (not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000):
        raise FileNotFoundError(
            "Missing/incomplete HOI-DETR checkpoint: %s. "
            "Follow https://github.com/AhmadDarKhalil/HOI-DETR#1-download-checkpoint "
            "and docs/model-assets.md for the author's download and placement." % checkpoint
        )
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("Allocate a GPU before --gpu checks")
        capability = torch.cuda.get_device_capability()
        supported = {(7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0)}
        if torch.__version__ == "2.7.1+cu128":
            supported.update({(10, 0), (12, 0)})
        if capability not in supported:
            raise RuntimeError(
                "GPU %s needs a separately supported CUDA/PyTorch stack" % (capability,)
            )
        boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 5.0, 5.0]], device="cuda")
        scores = torch.tensor([0.9, 0.8], device="cuda")
        _, keep = nms(boxes, scores, 0.5)
        if keep.numel() != 1:
            raise RuntimeError("CUDA NMS returned an unexpected result")
        from mmcv.ops.multi_scale_deform_attn import (
            MultiScaleDeformableAttnFunction,
            multi_scale_deformable_attn_pytorch,
        )

        value = torch.arange(32, dtype=torch.float32, device="cuda").reshape(1, 4, 2, 4) / 32
        value.requires_grad_()
        shapes = torch.tensor([[2, 2]], device="cuda", dtype=torch.long)
        starts = torch.tensor([0], device="cuda", dtype=torch.long)
        locations = torch.full((1, 2, 2, 1, 2, 2), 0.4, device="cuda")
        weights = torch.full((1, 2, 2, 1, 2), 0.5, device="cuda")
        output = MultiScaleDeformableAttnFunction.apply(
            value, shapes, starts, locations, weights, 64
        )
        reference = multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
        output.sum().backward()
        if not torch.isfinite(value.grad).all():
            raise RuntimeError("Deformable attention CUDA backward returned non-finite gradients")
        from mmcv.parallel import DataContainer, scatter

        for target in (0, "cuda:0", torch.device("cuda:0")):
            batch = {"image": DataContainer([torch.ones(1, 3, 4, 4)])}
            copied = scatter(batch, [target])[0]["image"]
            if not copied.is_cuda or copied.sum().item() != 48:
                raise RuntimeError("MMCV input transfer to GPU failed")
        torch.cuda.synchronize()
        print(
            "PASS CUDA NMS + deformable attention + input transfer:",
            torch.cuda.get_device_name(),
            capability,
        )
    print(
        "PASS HOI-DETR imports/config%s: %s" % ("/checkpoint" if not skip_weights else "", versions)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check either supported contact environment and its required native operators.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    try:
        check(args.root.resolve(), args.skip_weights, args.gpu)
    except Exception as error:
        parser.exit(1, "HOI-DETR check failed: %s: %s\n" % (type(error).__name__, error))

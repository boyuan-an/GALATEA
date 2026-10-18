import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_moge_model(device="cuda"):
    moge_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "imports", "MoGe")
    moge_dir = os.path.abspath(moge_dir)
    if moge_dir not in sys.path:
        sys.path.insert(0, moge_dir)

    from moge.model.v2 import MoGeModel

    print("Loading MoGe-2 model (Ruicheng/moge-2-vitl-normal)...")
    return MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()


def est_depth(
    image_list,
    device="cuda",
    intrinsics=None,
    gt_depth_first_frame=None,
    model=None,
    rectify_intrinsics=False,
):
    owns_model = model is None
    if owns_model:
        model = load_moge_model(device)

    original_image_size = cv2.imread(image_list[0]).shape[:2]
    camera = None

    fov_x = None
    if intrinsics is not None:
        if isinstance(intrinsics, str):
            K = np.loadtxt(intrinsics)
        else:
            K = np.array(intrinsics)
        if rectify_intrinsics:
            from galatea_recon.adapters.depth_camera import CenteredDepthCamera

            camera = CenteredDepthCamera(K, original_image_size)
            fov_x = camera.fov_x
            print(
                f"Using calibrated virtual camera: fov_x={fov_x:.2f} degrees; restoring original camera depth"
            )
        else:
            fx = K[0, 0]
            W = original_image_size[1]
            fov_x = 2 * math.atan(W / (2 * fx)) * 180.0 / math.pi
            print(
                f"Using known intrinsics: fx={fx:.2f}, image_width={W}, fov_x={fov_x:.2f} degrees"
            )
    else:
        print("No intrinsics provided, MoGe will recover focal length internally")

    depth_list = []
    for image_path in tqdm(image_list, desc="Estimating depth (MoGe)"):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if camera is not None:
            img = camera.image(img)
        image_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(
            2, 0, 1
        )

        with torch.no_grad():
            output = model.infer(image_tensor, fov_x=fov_x, resolution_level=9, use_fp16=True)

        depth = output["depth"]

        depth = torch.where(torch.isinf(depth), torch.zeros_like(depth), depth)

        if depth.shape != tuple(original_image_size):
            depth = (
                F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=original_image_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        if camera is not None:
            depth = torch.as_tensor(
                camera.restore(depth.detach().float().cpu().numpy()), device=device
            )

        depth_list.append(depth)

    if owns_model:
        del model
        torch.cuda.empty_cache()
    print(f"MoGe: produced {len(depth_list)} metric depth frames")

    return depth_list

import argparse
from pathlib import Path

import numpy as np
import torch

from armhand.data.rotations import aa_to_rotmat, rotmat_to_aa
from hand import HAND_BODY_KEYPOINTS
from reference import ReferenceLoader, palm_rotation

PALM_BODIES = ("right_index_MCP_VL", "right_middle_MCP_VL", "right_ring_MCP_VL", "right_pinky_MCP_VL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--table-z", type=float, default=0.415)
    args = parser.parse_args()
    device = torch.device(args.device)
    loader = ReferenceLoader(
        str((args.bank / "data").resolve(strict=True)), device, torch.eye(4, device=device), reject_jumps=False
    )
    rotation = aa_to_rotmat(torch.tensor([0.0, 0.0, -np.pi / 2], device=device)) @ aa_to_rotmat(
        torch.tensor([np.pi / 2, 0.0, 0.0], device=device)
    )
    bodies = [name for name, keypoint in HAND_BODY_KEYPOINTS.items() if keypoint != "wrist"]
    output_dir = args.bank / "wrist_targets"
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip in loader.data_indices:
        data = loader[clip]
        wrist_pos = data["wrist_pos"].to(device=device, dtype=torch.float32)
        packed = torch.concat([data["keypoints"][HAND_BODY_KEYPOINTS[name]] for name in bodies], dim=-1)
        packed = packed.reshape(packed.shape[0], -1, 3).to(device=device, dtype=torch.float32)
        wrist_pos = (rotation @ wrist_pos.T).T
        wrist_pos[:, 2] += float(args.table_z)
        packed = (rotation @ packed.reshape(-1, 3).T).T.reshape_as(packed)
        packed[..., 2] += float(args.table_z)
        palm = torch.stack([packed[:, bodies.index(name)] for name in PALM_BODIES], dim=1)
        np.savez(
            output_dir / f"{clip}.npz",
            position=wrist_pos.cpu().numpy(),
            rotation=rotmat_to_aa(palm_rotation(wrist_pos, palm)).cpu().numpy(),
        )
        print(f"[wrist] {clip} frames={wrist_pos.shape[0]}", flush=True)


if __name__ == "__main__":
    main()

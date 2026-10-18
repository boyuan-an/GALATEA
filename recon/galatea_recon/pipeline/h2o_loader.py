import os
from glob import glob

import cv2
import numpy as np

H2O_OBJECTS = {
    1: "book",
    2: "espresso",
    3: "lotion",
    4: "spray",
    5: "milk",
    6: "cocoa",
    7: "chips",
    8: "cappuccino",
    99: "mug",
    100: "wineglass",
}

DEFAULT_DEPTH_SCALE = 1000.0


def _read_mat4(path):
    return np.loadtxt(path).reshape(4, 4).astype(np.float64)


class H2OSequence:
    def __init__(self, cam_dir, object_models_root, depth_scale=DEFAULT_DEPTH_SCALE):
        self.cam_dir = os.path.abspath(cam_dir)
        self.object_models_root = os.path.abspath(object_models_root)
        self.depth_scale = float(depth_scale)

        self.rgb_files = sorted(glob(os.path.join(self.cam_dir, "rgb", "*.png")))
        if not self.rgb_files:
            raise FileNotFoundError(f"No rgb/*.png under {self.cam_dir}")
        self.depth_files = sorted(glob(os.path.join(self.cam_dir, "depth", "*.png")))
        self.num_frames = len(self.rgb_files)

        fx, fy, cx, cy, W, H = np.loadtxt(
            os.path.join(self.cam_dir, "cam_intrinsics.txt")
        ).tolist()
        self.width, self.height = int(W), int(H)
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        obj_id0, _ = self.load_obj_pose(0)
        self.obj_id = int(obj_id0)
        self.obj_name = None
        side = os.path.join(self.cam_dir, "obj_name.txt")
        if os.path.exists(side):
            with open(side) as f:
                self.obj_name = f.read().strip() or None
        if self.obj_name is None:
            self.obj_name = H2O_OBJECTS.get(self.obj_id, None)
        if self.obj_name is None:
            raise KeyError(
                f"Unknown object id {self.obj_id}: add it to H2O_OBJECTS, or write the "
                f"object-model folder name to {os.path.join(self.cam_dir, 'obj_name.txt')}"
            )
        self.obj_noun = self.obj_name
        noun_side = os.path.join(self.cam_dir, "obj_noun.txt")
        if os.path.exists(noun_side):
            with open(noun_side) as f:
                self.obj_noun = f.read().strip() or self.obj_name
        self.mesh_path = os.path.join(
            self.object_models_root, self.obj_name, f"{self.obj_name}.obj"
        )
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"Object mesh not found: {self.mesh_path}")

    def load_rgb(self, i):
        bgr = cv2.imread(self.rgb_files[i], cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_depth(self, i):
        raw = cv2.imread(self.depth_files[i], cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot read depth: {self.depth_files[i]}")
        return raw.astype(np.float32) / self.depth_scale

    def _frame_txt(self, sub, i):
        return os.path.join(self.cam_dir, sub, f"{i:06d}.txt")

    def load_cam_pose(self, i):
        return _read_mat4(self._frame_txt("cam_pose", i))

    def load_obj_pose(self, i):
        vals = np.loadtxt(self._frame_txt("obj_pose_rt", i)).astype(np.float64)
        obj_id = int(round(vals[0]))
        pose = vals[1:17].reshape(4, 4)
        return obj_id, pose

    def load_hand_joints(self, i):
        v = np.loadtxt(self._frame_txt("hand_pose", i)).astype(np.float64)
        hands = []
        if v.shape[0] >= 64 and v[0] > 0.5:
            hands.append(v[1:64].reshape(21, 3))
        if v.shape[0] >= 128 and v[64] > 0.5:
            hands.append(v[65:128].reshape(21, 3))
        return hands

    def obj_pose_seq(self):
        return np.stack([self.load_obj_pose(i)[1] for i in range(self.num_frames)], 0)

    def summary(self):
        d0 = self.load_depth(0)
        valid = d0[d0 > 0]
        cam0, camN = self.load_cam_pose(0), self.load_cam_pose(self.num_frames - 1)
        cam_drift = float(np.abs(cam0 - camN).max())
        return {
            "cam_dir": self.cam_dir,
            "num_frames": self.num_frames,
            "image_size": (self.width, self.height),
            "K": self.K.tolist(),
            "obj_id": self.obj_id,
            "obj_name": self.obj_name,
            "mesh_path": self.mesh_path,
            "depth_scale": self.depth_scale,
            "depth_m_min": float(valid.min()) if valid.size else None,
            "depth_m_median": float(np.median(valid)) if valid.size else None,
            "depth_m_max": float(valid.max()) if valid.size else None,
            "cam_pose_drift_max": cam_drift,
        }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Inspect one H2O camera clip (Step 1)")
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--object_models_root", required=True)
    ap.add_argument("--depth_scale", type=float, default=DEFAULT_DEPTH_SCALE)
    args = ap.parse_args()

    seq = H2OSequence(args.cam_dir, args.object_models_root, args.depth_scale)
    print(json.dumps(seq.summary(), indent=2))

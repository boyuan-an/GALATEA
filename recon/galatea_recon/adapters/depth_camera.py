import cv2
import numpy as np


class CenteredDepthCamera:
    def __init__(self, intrinsics, shape):
        height, width = shape
        k = np.asarray(intrinsics, dtype=np.float64)
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or k[0, 0] <= 0
            or k[1, 1] <= 0
            or min(shape) < 4
            or not np.allclose(k[2], [0, 0, 1])
            or abs(k[0, 1]) > 1e-8
        ):
            raise ValueError(
                "depth rectification needs finite pinhole intrinsics and a nonempty image"
            )
        inv = np.linalg.inv(k)
        z = inv @ [width / 2, height / 2, 1]
        z /= np.linalg.norm(z)
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        rotation = np.stack([x, np.cross(z, x), z])
        corners = np.array(
            [[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]]
        ).T
        rays = rotation @ inv @ corners
        if np.any(rays[2] <= 0):
            raise ValueError("camera field of view crosses the virtual camera plane")
        extent = np.abs(rays[:2] / rays[2]).max(axis=1)
        focal = min((width / 2 - 2) / extent[0], (height / 2 - 2) / extent[1])
        virtual = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        self.homography = virtual @ rotation @ inv
        self.shape = (height, width)
        self.fov_x = float(np.rad2deg(2 * np.arctan(width / (2 * focal))))
        yy, xx = np.mgrid[:height, :width]
        pixels = np.stack([xx, yy, np.ones_like(xx)]).reshape(3, -1)
        projection = self.homography @ pixels
        self.map_x = (projection[0] / projection[2]).reshape(shape).astype(np.float32)
        self.map_y = (projection[1] / projection[2]).reshape(shape).astype(np.float32)
        self.z_ratio = (rotation @ inv @ pixels)[2].reshape(shape).astype(np.float32)
        self.rotation, self.virtual_intrinsics = rotation, virtual

    def image(self, rgb):
        if rgb.shape[:2] != self.shape:
            raise ValueError("RGB shape changed within a calibrated sequence")
        h, w = self.shape
        return cv2.warpPerspective(
            rgb, self.homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    def restore(self, virtual_depth):
        if virtual_depth.shape != self.shape:
            raise ValueError("model depth and virtual camera shape differ")
        valid = np.isfinite(virtual_depth) & (virtual_depth > 0)
        numerator = cv2.remap(
            np.where(valid, virtual_depth, 0).astype(np.float32),
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
        )
        weight = cv2.remap(valid.astype(np.float32), self.map_x, self.map_y, cv2.INTER_LINEAR)
        depth = np.zeros(self.shape, np.float32)
        keep = (weight > 0.999) & (self.z_ratio > 0)
        depth[keep] = numerator[keep] / weight[keep] / self.z_ratio[keep]
        depth[~np.isfinite(depth)] = 0
        return depth

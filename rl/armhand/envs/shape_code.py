from bps_torch.bps import bps_torch


class ShapeEncoder:
    def __init__(self, n_points, radius, device):
        self.encoder = bps_torch(
            bps_type="grid_sphere", n_bps_points=n_points, radius=radius, randomize=False, device=device
        )

    def __call__(self, points):
        return self.encoder.encode(points, feature_type="dists")["dists"]

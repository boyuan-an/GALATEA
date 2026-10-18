import os
from pathlib import Path


ASSETS = ("mano_mean_params.npz", "MANO_RIGHT.pkl", "wilor_final.ckpt", "detector.pt")


def _require_assets(names):
    repo = Path(__file__).resolve().parents[2]
    cache = Path(os.environ.get("CACHE_ROOT") or os.environ.get("XDG_CACHE_HOME") or repo / ".cache")
    root = Path(os.environ.get("WILOR_CACHE_DIR") or cache / "wilor").expanduser().resolve()
    target = root / "pretrained_models"
    missing = [
        name for name in names
        if not (target / name).is_file() or (target / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Missing/empty WiLoR assets in %s: %s. "
            "GALATEA does not download these files or retrieve MANO from a mirror. "
            "Follow https://github.com/rolpotamias/WiLoR#original-installation for WiLoR; "
            "register and obtain MANO under its terms at https://mano.is.tue.mpg.de/. "
            "See %s for file placement and official links."
            % (target, ", ".join(missing), repo / "docs/model-assets.md")
        )
    return str(root)


def wilor_asset_dir():
    return _require_assets(ASSETS)


def mano_right_path():
    path = Path(_require_assets(("MANO_RIGHT.pkl",))) / "pretrained_models/MANO_RIGHT.pkl"
    return str(path)

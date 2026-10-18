import hashlib
import os
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def image_identity(rgb):
    rgb = np.ascontiguousarray(rgb)
    return hashlib.sha256(str((rgb.shape, rgb.dtype.str)).encode() + rgb.tobytes()).hexdigest()


def save_predictions(path, predictions, rgb=None, hand_conf=0.3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.empty(len(predictions), dtype=object)
    values[:] = predictions
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, predictions=values,
                                image_sha256="" if rgb is None else image_identity(rgb),
                                hand_conf=float(hand_conf))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_predictions(path, rgb, hand_conf=0.3):
    if rgb is None:
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if (str(data["image_sha256"].item()) != image_identity(rgb)
                    or float(data["hand_conf"].item()) != hand_conf):
                return None
            return list(data["predictions"])
    except (OSError, ValueError, EOFError, BadZipFile, KeyError):
        return None

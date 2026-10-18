import importlib.util
from pathlib import Path


def replace_once(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError("Unexpected MMCV source layout: %s" % path)
    path.write_text(text.replace(old, new))


def patch(root):
    replace_once(
        root / "parallel/_functions.py",
        "streams = [_get_stream(device) for device in target_gpus]",
        "streams = [_get_stream(torch.device('cuda', device) if isinstance(device, int) "
        "else torch.device(device)) for device in target_gpus]",
    )
    replace_once(
        root / "parallel/distributed.py",
        "self._use_replicated_tensor_module else self.module",
        "getattr(self, '_use_replicated_tensor_module', False) else self.module",
    )


if __name__ == "__main__":
    spec = importlib.util.find_spec("mmcv")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install MMCV 1.7.2 in the contact environment first")
    root = Path(spec.origin).parent
    patch(root)
    print("Applied MMCV/PyTorch compatibility fixes:", root)

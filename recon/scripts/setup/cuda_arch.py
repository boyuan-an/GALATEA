import os
import re


def normalize_arches(value, cuda_version=None):
    if cuda_version is None:
        import torch

        cuda_version = torch.version.cuda or "0.0"
    supported = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}
    if tuple(int(v) for v in cuda_version.split(".")[:2]) >= (12, 8):
        supported.update({"10.0", "12.0"})
    arches = re.split(r"[;\s]+", value.strip())
    if not value.strip() or any(a.removesuffix("+PTX") not in supported for a in arches):
        raise ValueError(
            "CUDA %s setup expects numeric targets: %s (optional +PTX). "
            "For RTX 50 / Blackwell, install with --cuda128 in fresh environments; "
            "see docs/installation.md." % (cuda_version, ", ".join(sorted(supported)))
        )
    return ";".join(sorted(set(arches)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Select explicit numeric CUDA targets; never guess Ada on a GPU-less host.")
    parser.add_argument("--arch", default=os.environ.get("TORCH_CUDA_ARCH_LIST", ""))
    args = parser.parse_args()
    try:
        requested = args.arch
        if not requested:
            import torch

            if not torch.cuda.is_available():
                raise ValueError(
                    "No allocated/visible CUDA GPU. Allocate a GPU, or set "
                    "TORCH_CUDA_ARCH_LIST for the target machine (A100/A800: 8.0; "
                    "RTX 3090: 8.6; RTX 4090/5880/6000 Ada: 8.9)."
                )
            requested = ";".join(
                "%d.%d" % torch.cuda.get_device_capability(i)
                for i in range(torch.cuda.device_count())
            )
        print(normalize_arches(requested))
    except ValueError as error:
        parser.exit(1, str(error) + "\n")

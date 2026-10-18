from pathlib import Path

import numpy as np

from galatea_recon.pipeline.h2o_loader import H2OSequence
from galatea_recon.pipeline.object_quality import atomic_json, file_hash
from galatea_recon.pipeline.perception_cache import atomic_numpy, load_mask_cache, mask_identity


def trimmed_affine_fit(x, y, trim_pct=0.2):
    A = np.stack([x, np.ones_like(x)], 1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if trim_pct and trim_pct > 0 and len(x) > 50:
        res = np.abs(a * x + b - y)
        keep = res <= np.quantile(res, 1.0 - trim_pct)
        a, b = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
    return float(a), float(b)


def get_sam2_masks(cam_dir, object_models_root, work_dir, max_frame,
                   depth_scale, device, prompt_source="auto", wilor_pipeline=None,
                   object_detector=None, sam2_predictor=None,
                   frame0_wilor_cache=None, frame0_prompts=None,
                   frame0_wilor_predictions=None, force=False):
    seq = H2OSequence(cam_dir, object_models_root, depth_scale)
    n = min(max_frame + 1, seq.num_frames)
    inputs = mask_identity(seq, n, depth_scale, prompt_source)
    cache = Path(work_dir) / "sam2_masks.npz"
    report_path = cache.with_suffix(".json")
    cached = (load_mask_cache(work_dir, inputs)
              if not force and frame0_prompts is None and frame0_wilor_predictions is None
              else None)
    if cached is not None:
        print(f"[sam2] loaded cached masks: {cache}")
        return cached
    report_path.unlink(missing_ok=True)
    from galatea_recon.pipeline.sam_masks import compute_sam2_masks

    seq, masks = compute_sam2_masks(cam_dir, object_models_root, work_dir,
                                    n_frames=n, depth_scale=depth_scale,
                                    device=device, whole_person=False,
                                    prompt_source=prompt_source,
                                    wilor_pipeline=wilor_pipeline,
                                    object_detector=object_detector,
                                    sam2_predictor=sam2_predictor,
                                    frame0_wilor_cache=frame0_wilor_cache,
                                    frame0_prompts=frame0_prompts,
                                    frame0_wilor_predictions=frame0_wilor_predictions)
    H, W = seq.height, seq.width
    fg, obj = {}, {}
    for f, md in masks.items():
        u = np.zeros((H, W), bool)
        for m in md.values():
            u |= np.asarray(m, bool)
        fg[f] = u
        obj[f] = np.asarray(md.get(0, np.zeros((H, W), bool)), bool)
    if set(fg) != set(range(n)) or any(
            a.shape != (H, W) for values in (fg, obj) for a in values.values()):
        raise ValueError("SAM2 did not produce a complete mask timeline")
    store = {f"fg_{f}": fg[f] for f in fg}
    store.update({f"obj_{f}": obj[f] for f in obj})
    atomic_numpy(cache, store)
    atomic_json(report_path, {"inputs": inputs, "sha256": file_hash(cache)})
    print(f"[sam2] cached masks -> {cache}")
    return fg, obj

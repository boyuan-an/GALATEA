import json
import os


def _empty(n):
    return {i: set() for i in range(n)}


def from_manual(label_json, seq_id, num_frames):
    contact = _empty(num_frames)
    with open(label_json) as f:
        spec = json.load(f)
    intervals = spec.get(str(seq_id), spec.get(seq_id, []))
    for iv in intervals:
        hand = iv["hand"].lower()
        for fr in range(max(0, iv["start"]), min(num_frames - 1, iv["end"]) + 1):
            contact[fr].add(hand)
    return contact


def always(num_frames, hands=("left", "right")):
    return {i: set(hands) for i in range(num_frames)}


def from_manual_fingers(label_json, seq_id, num_frames):
    contact = {i: {} for i in range(num_frames)}
    with open(label_json) as f:
        spec = json.load(f)
    for iv in spec.get(str(seq_id), spec.get(seq_id, [])):
        hand, finger = iv["hand"].lower(), iv["finger"].lower()
        for fr in range(max(0, iv["start"]), min(num_frames - 1, iv["end"]) + 1):
            contact[fr].setdefault(hand, set()).add(finger)
    return contact


def load_contact(mode, num_frames, *, seq_id=None, manual_json=None):
    if mode == "always":
        return always(num_frames)
    if mode == "manual":
        if not manual_json:
            raise ValueError("manual mode needs --manual_json")
        return from_manual(manual_json, seq_id, num_frames)
    raise ValueError(f"unknown contact mode: {mode}")


if __name__ == "__main__":
    import argparse
    from glob import glob

    ap = argparse.ArgumentParser(description="Inspect contact labels")
    ap.add_argument("--mode", choices=["manual", "always"], required=True)
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--seq_id", default=None)
    ap.add_argument("--manual_json", default=None)
    args = ap.parse_args()

    rgb = sorted(glob(os.path.join(args.cam_dir, "rgb", "*.png")))
    c = load_contact(args.mode, len(rgb), seq_id=args.seq_id, manual_json=args.manual_json)
    nfr = sum(1 for v in c.values() if v)
    print(f"[contact:{args.mode}] {nfr}/{len(rgb)} frames in contact; "
          f"e.g. frame0={c[0]}, frame{len(rgb)//2}={c[len(rgb)//2]}")

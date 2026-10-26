#!/usr/bin/env python3
"""Render a PlateLock synthetic registration demo, no footage or GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from platelock import geometry  # noqa: E402

H, W, INNER = 132, 236, 24
SHIFT_X, SHIFT_Y = 16, -10


def make_synthetic_plate() -> np.ndarray:
    img = np.full((H, W, 3), (35, 48, 55), np.uint8)
    for x in range(0, W, 24):
        cv2.line(img, (x, 0), (x, H), (69, 92, 97), 1)
    for y in range(0, H, 24):
        cv2.line(img, (0, y), (W, y), (69, 92, 97), 1)
    cv2.rectangle(img, (35, 30), (112, 107), (124, 177, 216), -1)
    cv2.rectangle(img, (48, 43), (99, 92), (25, 60, 88), 3)
    cv2.circle(img, (163, 66), 35, (179, 212, 230), -1)
    cv2.circle(img, (163, 66), 24, (39, 88, 104), -1)
    cv2.line(img, (163, 66), (182, 48), (130, 190, 231), 4)
    cv2.putText(img, "SYNTH", (46, 70), cv2.FONT_HERSHEY_SIMPLEX,
                .46, (226, 237, 244), 1, cv2.LINE_AA)
    return img


def register_synthetic() -> dict:
    original = make_synthetic_plate()
    transform = np.array([[1, 0, SHIFT_X], [0, 1, SHIFT_Y]],
                         dtype=np.float32)
    clean = cv2.warpAffine(original, transform, (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mx, my = geometry.affine_to_backward_map(transform, H, W)
    registered = geometry.apply_backward_map(clean, mx, my)
    crop = np.s_[INNER:-INNER, INNER:-INNER]
    unaligned = float(np.abs(original[crop].astype(np.int16)
                             - clean[crop].astype(np.int16)).mean())
    aligned = float(np.abs(original[crop].astype(np.int16)
                           - registered[crop].astype(np.int16)).mean())
    if aligned > 1e-6 or aligned >= unaligned:
        raise AssertionError("Known integer translation did not register")
    return {"original": original, "unaligned_clean": clean,
            "registered_clean": registered,
            "unaligned_mae": unaligned, "registered_mae": aligned}


def make_contact_sheet(frames: dict) -> np.ndarray:
    gap, header = 12, 43
    sheet = np.full((H + header + 22, 3 * W + 2 * gap, 3),
                    (242, 244, 243), np.uint8)
    labels = ("SYNTHETIC ORIGINAL", "MISALIGNED COPY", "PLATELOCK REGISTERED")
    for i, key in enumerate(("original", "unaligned_clean", "registered_clean")):
        x = i * (W + gap)
        sheet[header:header + H, x:x + W] = frames[key]
        cv2.putText(sheet, labels[i], (x + 3, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (31, 51, 64), 1, cv2.LINE_AA)
    cv2.putText(sheet, "KNOWN TRANSLATION / CPU / NO TRACKER OR MODEL",
                (8, H + header + 15), cv2.FONT_HERSHEY_SIMPLEX,
                .35, (45, 62, 70), 1, cv2.LINE_AA)
    return sheet


def render_png(path: Path) -> dict:
    """Explicit new output file; preserve existing files."""
    if not path.parent.is_dir():
        raise ValueError("Output parent must already exist")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Output already exists: " + str(path))
    result = register_synthetic()
    ok, data = cv2.imencode(".png", make_contact_sheet(result),
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("Could not encode synthetic image")
    with path.open("xb") as out:
        out.write(data.tobytes())
    return {"synthetic_only": True,
            "method": "platelock.geometry backward map",
            "known_translation_px": [SHIFT_X, SHIFT_Y],
            "unaligned_mae": result["unaligned_mae"],
            "registered_mae": result["registered_mae"],
            "measurement": "BGR pixel MAE on 24-px interior crop",
            "output": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output new PNG; never overwrite")
    args = parser.parse_args()
    try:
        print(json.dumps(render_png(args.out), sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, AssertionError) as exc:
        parser.exit(1, "DEMO ERROR: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""
platelock.roi_loader — minimal manual trusted-ROI / exclusion loader.

ROIs are stored in full-resolution pixel coords (e.g. configs/rois/*.json at
5120x2700). Downstream stages run at a working scale; pass scale = working/full
to map ROIs into working coords.

A trusted ROI marks background we DO trust for correspondence. An exclusion box
marks actor/flame/generated regions we DO NOT trust. For the first RoMa
falsification, trusted-ROI inclusion alone suffices (the ROIs sit on background,
so restricting matches to them inherently avoids actor/flame).
"""

import json
import numpy as np


def load_rois(path):
    with open(path) as f:
        return json.load(f)["rois"]


def _scaled(roi, scale):
    return (roi["x"] * scale, roi["y"] * scale,
            roi["w"] * scale, roi["h"] * scale,
            float(roi.get("weight", 1.0)))


def inclusion_mask(rois, H, W, scale=1.0):
    """Bool HxW mask: True inside any trusted ROI (working coords)."""
    m = np.zeros((H, W), bool)
    for r in rois:
        x, y, w, h, _ = _scaled(r, scale)
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def points_in_rois(pts, rois, scale=1.0):
    """Bool (N,) — point (working coords) inside any trusted ROI."""
    pts = np.asarray(pts, np.float32)
    keep = np.zeros(len(pts), bool)
    for r in rois:
        x, y, w, h, _ = _scaled(r, scale)
        inside = ((pts[:, 0] >= x) & (pts[:, 0] < x + w) &
                  (pts[:, 1] >= y) & (pts[:, 1] < y + h))
        keep |= inside
    return keep


def point_weights(pts, rois, scale=1.0):
    """Per-point weight = max weight of containing ROI, else 0."""
    pts = np.asarray(pts, np.float32)
    wts = np.zeros(len(pts), np.float32)
    for r in rois:
        x, y, w, h, wt = _scaled(r, scale)
        inside = ((pts[:, 0] >= x) & (pts[:, 0] < x + w) &
                  (pts[:, 1] >= y) & (pts[:, 1] < y + h))
        wts[inside] = np.maximum(wts[inside], wt)
    return wts

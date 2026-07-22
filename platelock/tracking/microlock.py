"""
platelock.tracking.microlock — Stage C part 1: per-frame anisotropic pixel-lock.

After the smooth tracked field is applied, the residual offset between the
warped clean plate and the TRUE original is measured directly (Sobel-gradient-
magnitude NCC template match, sub-pixel refined) on a coarse-to-fine tile
pyramid and folded back in as a correction. Anisotropic by construction: a
window's vertical bars constrain only x, floor lines only y (classic aperture
problem) -- an isotropic 2D fit would inject the unconstrained axis's garbage
into the field. Each axis is measured and RBF'd independently, gated by the
NCC surface's per-axis curvature (a flat axis = unconstrained, or a lighting-
gradient false match -- both rejected).
"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


def gradient_magnitude(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3) ** 2)


def _subpixel_peak(res, ml):
    x, y = ml
    out = [float(x), float(y)]
    if 0 < x < res.shape[1] - 1:
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
        d = a - 2 * b + c
        if abs(d) > 1e-9:
            out[0] = x + 0.5 * (a - c) / d
    if 0 < y < res.shape[0] - 1:
        a, b, c = res[y - 1, x], res[y, x], res[y + 1, x]
        d = a - 2 * b + c
        if abs(d) > 1e-9:
            out[1] = y + 0.5 * (a - c) / d
    return out


def measure_tiles(GO, GW, mask, W, H, TX, TY, R, min_score=0.35, min_std=3.0, curv_gate=0.02):
    """One tile-grid scale of the coarse-to-fine pyramid. GO/GW are Sobel
    gradient-magnitude images of the original and the (progressively corrected)
    warped clean. Returns (pts, offs, wts); wts is zeroed per-axis where that
    axis's NCC curvature is below curv_gate (unconstrained-axis / false-match guard)."""
    pts, offs, wts = [], [], []
    for ty in range(TY):
        for tx in range(TX):
            x0, y0 = tx * W // TX, ty * H // TY
            x1, y1 = x0 + W // TX, y0 + H // TY
            if x1 - x0 <= 2 * R + 8 or y1 - y0 <= 2 * R + 8:
                continue
            if mask[y0:y1, x0:x1].mean() > 0.25:
                continue
            tpl = GW[y0 + R:y1 - R, x0 + R:x1 - R]
            if tpl.std() < min_std:
                continue
            res = cv2.matchTemplate(GO[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv < min_score:
                continue
            x, y = ml
            if not (0 < x < res.shape[1] - 1 and 0 < y < res.shape[0] - 1):
                continue
            cx = abs(res[y, x - 1] - 2 * res[y, x] + res[y, x + 1])
            cy = abs(res[y - 1, x] - 2 * res[y, x] + res[y + 1, x])
            if max(cx, cy) < curv_gate:
                continue
            sx, sy = _subpixel_peak(res, ml)
            pts.append([(x0 + x1) / 2, (y0 + y1) / 2])
            offs.append([sx - R, sy - R])
            wts.append([mv if cx >= curv_gate else 0.0, mv if cy >= curv_gate else 0.0])
    return np.array(pts), np.array(offs), np.array(wts)


def correction_field(pts, offs, wts, W, H, clamp=14.0, hole=None, coarse=(48, 26),
                      zero_anchor_min_dist=140):
    """Two independent per-axis RBFs (never one 2D vector field -- see module
    docstring). `hole`, if given, suppresses virtual zero-anchors inside it so
    the field BRIDGES across an unmeasurable region (e.g. an actor silhouette)
    from its surroundings instead of fading the correction to zero there."""
    cw, ch = coarse
    out = np.zeros((ch, cw, 2), np.float32)
    if len(pts) < 4:
        return out
    gy, gx = np.meshgrid(np.linspace(0, H - 1, ch), np.linspace(0, W - 1, cw), indexing="ij")
    ev = np.stack([gx.ravel(), gy.ravel()], 1)
    gy0, gx0 = np.meshgrid(np.arange(100, H, 200), np.arange(100, W, 200), indexing="ij")
    virt0 = np.stack([gx0.ravel(), gy0.ravel()], 1).astype(float)
    for ax in (0, 1):
        sel = wts[:, ax] > 0
        if sel.sum() < 3:
            continue
        P0 = pts[sel]
        V0 = np.clip(offs[sel, ax], -clamp, clamp)
        W0 = wts[sel, ax]
        d, _ = cKDTree(P0).query(virt0)
        keep = d > zero_anchor_min_dist
        if hole is not None:
            hx = np.clip(virt0[:, 0].astype(int), 0, W - 1)
            hy = np.clip(virt0[:, 1].astype(int), 0, H - 1)
            keep &= ~hole[hy, hx]
        virt = virt0[keep]
        P = np.vstack([P0, virt])
        V = np.concatenate([V0, np.zeros(len(virt))])
        sm = np.concatenate([0.5 / W0 ** 2, np.full(len(virt), 0.3)])
        rb = RBFInterpolator(P, V[:, None], kernel="linear", smoothing=sm)
        out[..., ax] = rb(ev).reshape(ch, cw)
    return np.clip(out, -clamp, clamp)


def microlock_frame(orig_img, clean_img, base_mx, base_my, mask0, cmask, W, H, pyramid,
                     clamp=14.0, hole=None, coarse=(48, 26), damping=0.8, dilate_px=25,
                     min_score=0.35, min_std=3.0, curv_gate=0.02):
    """Run the full coarse-to-fine pyramid for one frame, against the smooth
    tracked field (base_mx, base_my) BEFORE this frame's correction. Returns
    the coarse (ch,cw,2) correction total -- upsample and SUBTRACT from the
    base map at composition time (platelock.dense_field.assemble_map convention:
    content shifts by +s => map -= corr, verified empirically)."""
    cw, ch = coarse
    GO = gradient_magnitude(orig_img)
    total = np.zeros((ch, cw, 2), np.float32)
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    for TX, TY, R in pyramid:
        full = cv2.resize(np.clip(total, -clamp, clamp), (W, H), interpolation=cv2.INTER_CUBIC)
        wa = cv2.remap(clean_img, base_mx - full[..., 0], base_my - full[..., 1], cv2.INTER_LINEAR)
        wcm = cv2.remap(cmask.astype(np.float32), base_mx - full[..., 0], base_my - full[..., 1],
                        cv2.INTER_LINEAR) > 127
        mask = mask0 | cv2.dilate(wcm.astype(np.uint8), kernel).astype(bool)
        pts, offs, wts = measure_tiles(GO, gradient_magnitude(wa), mask, W, H, TX, TY, R,
                                       min_score=min_score, min_std=min_std, curv_gate=curv_gate)
        if len(pts) >= 4:
            total += damping * cv2.resize(
                correction_field(pts, offs, wts, W, H, clamp=clamp, hole=hole, coarse=coarse),
                (cw, ch), interpolation=cv2.INTER_AREA)
    return total

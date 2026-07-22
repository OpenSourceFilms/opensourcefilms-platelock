"""
platelock.diagnostics — crop-based truth. Full-frame PSNR/SSIM have repeatedly
lied on this project; the ONLY sign-off is whether trusted-background crop edges
align. Green = original edges, red = candidate (clean/warped) edges; yellow where
they coincide (good lock). The chamfer score is the mean distance (px) from
original edge pixels to the nearest candidate edge pixel — lower is better.
"""

import numpy as np
import cv2


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _norm(gray):
    """Local contrast normalisation so faint structure in dim crops becomes
    visible to Canny / to the eye."""
    return _CLAHE.apply(gray)


def edges(img, lo=50, hi=150, normalize=False):
    g_ = _gray(img)
    if normalize:
        g_ = _norm(g_)
    return cv2.Canny(g_, lo, hi)


def edge_chamfer(orig_crop, cand_crop):
    """Mean distance (px) from ORIGINAL edge pixels to nearest CANDIDATE edge.
    Lock-quality proxy on a trusted crop. Lower = tighter lock."""
    oe, ce = edges(orig_crop), edges(cand_crop)
    if oe.sum() == 0 or ce.sum() == 0:
        return float("inf")
    dt = cv2.distanceTransform(255 - ce, cv2.DIST_L2, 3)
    ys, xs = np.where(oe > 0)
    return float(dt[ys, xs].mean())


def edge_overlay(orig_crop, cand_crop, dim=0.5, normalize=True):
    """BGR viz: original edges green, candidate edges red, overlap yellow.
    normalize=True CLAHE-boosts contrast so dim crops are legible."""
    base = _norm(_gray(orig_crop)) if normalize else _gray(orig_crop)
    viz = cv2.cvtColor((base * dim).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    oe, ce = edges(orig_crop, normalize=normalize), edges(cand_crop, normalize=normalize)
    viz[oe > 0] = (0, 255, 0)
    viz[ce > 0] = (0, 0, 255)
    viz[(oe > 0) & (ce > 0)] = (0, 255, 255)
    return viz


def grad_ncc(a, b):
    """Brightness-robust crop alignment score: NCC of Sobel gradient magnitude.
    Range ~[-1,1], higher = better aligned. Replaces edge_chamfer, which returns
    inf on dim/low-contrast crops where Canny finds no edges."""
    def gm(x):
        x = _gray(x).astype(np.float32)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    A, B = gm(a), gm(b)
    A = (A - A.mean()) / (A.std() + 1e-6)
    B = (B - B.mean()) / (B.std() + 1e-6)
    return float((A * B).mean())


def grad_ncc_gated(a, b, pct=70, exclude=None):
    """Full-frame, position-INDEPENDENT lock score. Scores gradient-magnitude NCC
    only where the ORIGINAL has real structure (top (100-pct)% of gradient), so it
    is meaningful at any camera position across a pan — unlike a fixed ROI box that
    pans onto featureless wall. Range ~[-1,1], higher = better lock.

    exclude: optional bool (H,W) mask, True = do NOT score here (actor/fire/paintout
    regions with no valid correspondence). The top-gradient threshold is computed
    over the SCORABLE region only, so actor edges don't inflate it and then starve
    the real-structure pixels."""
    def gm(x):
        x = _gray(x).astype(np.float32)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    A, B = gm(a), gm(b)
    if exclude is not None:
        valid = ~np.asarray(exclude, bool)
        if int(valid.sum()) < 50:
            return 0.0
        m = (A >= np.percentile(A[valid], pct)) & valid
    else:
        m = A >= np.percentile(A, pct)
    if int(m.sum()) < 50:
        return 0.0
    Av, Bv = A[m], B[m]
    Av = (Av - Av.mean()) / (Av.std() + 1e-6)
    Bv = (Bv - Bv.mean()) / (Bv.std() + 1e-6)
    return float((Av * Bv).mean())


def crop(img, box):
    """box = (x,y,w,h) in img coords; clipped to bounds."""
    H, W = img.shape[:2]
    x, y, w, h = [int(v) for v in box]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return img[y0:y1, x0:x1]


def label(img, text):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack_pad(imgs, pad=6, bg=40):
    imgs = [i if i.ndim == 3 else cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs]
    h = max(i.shape[0] for i in imgs)
    out = []
    for im in imgs:
        if im.shape[0] < h:
            im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(bg, bg, bg))
        out.append(im)
        out.append(np.full((h, pad, 3), bg, np.uint8))
    return np.hstack(out[:-1])


def vstack_pad(imgs, pad=6, bg=40):
    imgs = [i if i.ndim == 3 else cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs]
    w = max(i.shape[1] for i in imgs)
    out = []
    for im in imgs:
        if im.shape[1] < w:
            im = cv2.copyMakeBorder(im, 0, 0, 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=(bg, bg, bg))
        out.append(im)
        out.append(np.full((pad, w, 3), bg, np.uint8))
    return np.vstack(out[:-1])


def draw_matches(orig_bgr, clean_bgr, kA, kB, keep=None, max_draw=300):
    """Side-by-side orig|clean with match lines. keep=bool mask colours kept
    (green) vs dropped (dim). Dropped drawn first so kept sit on top."""
    canvas = hstack_pad([orig_bgr, clean_bgr], pad=8)
    ox = orig_bgr.shape[1] + 8
    n = len(kA)
    if n == 0:
        return canvas
    idx = np.random.default_rng(0).choice(n, min(max_draw, n), replace=False)
    for i in idx:
        if keep is not None and not keep[i]:
            a = (int(kA[i, 0]), int(kA[i, 1])); b = (int(kB[i, 0]) + ox, int(kB[i, 1]))
            cv2.line(canvas, a, b, (60, 60, 120), 1, cv2.LINE_AA)
    for i in idx:
        if keep is None or keep[i]:
            a = (int(kA[i, 0]), int(kA[i, 1])); b = (int(kB[i, 0]) + ox, int(kB[i, 1]))
            cv2.line(canvas, a, b, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, a, 2, (0, 255, 255), -1)
    return canvas

"""
platelock.tracking.hole_fill — Stage C part 2: hole fill.

Two methods:

`harmonic` (default since 2026-07-11, fixes the compositor-reported occlusion
wobble): solve the Laplace equation inside each hole component with Dirichlet
boundary = the real surrounding field (as a residual over a per-frame ring-
plane prior), on a x4-downsampled grid, bicubic-upsampled back with a narrow
seam blend. Real content outside the hole is untouched TO THE PIXEL, each
component fills from ITS own local surroundings, and the fill follows the
boundary's real frame-to-frame motion by construction.

`ring_plane` (legacy, shipped in the original multikey13 deliverable): fit an
affine plane over a wide ring around the (unioned) mask and blend it in with
a 180px feather, savgol-smoothing the 6 coefficients over time. Diagnosed
failure (2026-07-11): the feathered blend yanks real ring content a median
~4px off its tracked position with ~0.5px/frame fluctuation (a wobbling halo
traveling with the actor), the temporally-oversmoothed plane slides relative
to its own boundary, and disconnected mask components all share one global
actor-dominated plane. Kept for comparison/fallback.

(An anchor-constrained harmonic variant — pinning the interior to PCHIP-
bridged in-hole tracks — was tested and REJECTED: bridged positions are
interpolated guesses whose temporal jitter (~8px in-hole jerk vs 1.4 pure-
boundary) injects straight into the fill.)
"""
import cv2
import numpy as np
import scipy.sparse as sp
from scipy.signal import savgol_filter
from scipy.sparse.linalg import splu


def ring_regions(mask, inner_px=61, outer_px=241):
    m8 = mask.astype(np.uint8)
    inner = cv2.dilate(m8, np.ones((inner_px, inner_px), np.uint8)).astype(bool)
    outer = cv2.dilate(m8, np.ones((outer_px, outer_px), np.uint8)).astype(bool)
    return inner, outer & ~inner, outer


def fit_ring_plane(mx, my, ring, n_samples=4000, seed=0):
    """lstsq-fit map(x,y) ~= [x,y,1]@coef over ring-sampled points. Returns a
    (2,3) array [coef_x; coef_y], or None if the ring has too few pixels."""
    ys, xs = np.where(ring)
    if len(ys) < 500:
        return None
    sub = np.random.default_rng(seed).choice(len(ys), min(n_samples, len(ys)), replace=False)
    A = np.stack([xs[sub], ys[sub], np.ones(len(sub))], 1).astype(np.float64)
    cx, _, _, _ = np.linalg.lstsq(A, mx[ys[sub], xs[sub]].astype(np.float64), rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, my[ys[sub], xs[sub]].astype(np.float64), rcond=None)
    return np.stack([cx, cy])


def smooth_coefficients(coef_seq, window=9, polyorder=2):
    """coef_seq: (T,2,3). Smooth temporally -- the plane should evolve as
    smoothly as the camera pan it represents."""
    return savgol_filter(coef_seq, window, polyorder, axis=0)


def apply_plane(mx, my, inner, outer, coef, feather_px=180):
    """Blend the affine plane into (mx,my) across `outer`, weighted by
    distance-transform feathering from the inner boundary outward -- full
    plane at the hole core, fading to the original field at the ring's edge."""
    yy, xx = np.where(outer)
    px = (xx * coef[0, 0] + yy * coef[0, 1] + coef[0, 2]).astype(np.float32)
    py = (xx * coef[1, 0] + yy * coef[1, 1] + coef[1, 2]).astype(np.float32)
    dist = cv2.distanceTransform((~inner).astype(np.uint8), cv2.DIST_L2, 3)
    w = np.clip(1.0 - dist / feather_px, 0, 1)[yy, xx]
    mx2, my2 = mx.copy(), my.copy()
    mx2[yy, xx] = w * px + (1 - w) * mx[yy, xx]
    my2[yy, xx] = w * py + (1 - w) * my[yy, xx]
    return mx2, my2


def fit_ring_homography(mx, my, ring, n_samples=4000, seed=0):
    """RANSAC homography fit over ring-sampled points: src=(x,y), dst=
    (map_x,map_y). Same ring-sampling as fit_ring_plane (identical rng
    seeding) so the two are directly comparable. Returns a 3x3 H normalized
    so H[2,2]==1, or None if the ring has too few pixels or the fit fails.
    ring_model="homography" opt-in: the wall is planar and the camera pans,
    so the true map is projective, not affine -- fit_ring_plane's affine fit
    is geometrically wrong for this scene, this is the principled fix."""
    ys, xs = np.where(ring)
    if len(ys) < 500:
        return None
    sub = np.random.default_rng(seed).choice(len(ys), min(n_samples, len(ys)), replace=False)
    src = np.stack([xs[sub], ys[sub]], axis=1).astype(np.float64)
    dst = np.stack([mx[ys[sub], xs[sub]], my[ys[sub], xs[sub]]], axis=1).astype(np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None
    H = H / H[2, 2]
    return H


# This linear smoothing of homography matrix entries is NOT geometrically
# principled (not equivalent to smoothing in the projective group / SL(3)) but
# is intentionally consistent with how smooth_coefficients smooths affine
# coefficients elsewhere in this module -- a documented simplification, not an
# oversight. A geometrically correct approach would smooth in the Lie algebra
# (matrix log) of the projective transform group; this was not implemented.
def smooth_homographies(H_seq, window=9, polyorder=2):
    """H_seq: (T,3,3). Savgol-smooth the 8 free params (H[2,2] pinned to 1)."""
    T = H_seq.shape[0]
    H_flat = H_seq.reshape(T, 9)
    smoothed = savgol_filter(H_flat[:, :8], window, polyorder, axis=0)
    result = np.zeros((T, 3, 3), dtype=H_seq.dtype)
    result[:, :2, :] = smoothed[:, :6].reshape(T, 2, 3)
    result[:, 2, :2] = smoothed[:, 6:8]
    result[:, 2, 2] = 1.0
    return result


def apply_homography(mx, my, inner, outer, H, feather_px=180):
    """Same blend structure as apply_plane (distance-transform feather from
    the inner boundary outward) but projecting with a homography instead of
    an affine coefficient multiply."""
    yy, xx = np.where(outer)
    pts = np.stack([xx, yy], axis=1).astype(np.float64).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    px = proj[:, 0].astype(np.float32)
    py = proj[:, 1].astype(np.float32)
    dist = cv2.distanceTransform((~inner).astype(np.uint8), cv2.DIST_L2, 3)
    w = np.clip(1.0 - dist / feather_px, 0, 1)[yy, xx]
    mx2, my2 = mx.copy(), my.copy()
    mx2[yy, xx] = w * px + (1 - w) * mx[yy, xx]
    my2[yy, xx] = w * py + (1 - w) * my[yy, xx]
    return mx2, my2


def component_regions(mask, inner_px=61, outer_px=241):
    """Like ring_regions but per DISCONNECTED COMPONENT of `mask`, each
    dilated independently from only its own pixels (not the unioned mask).
    fit_ring_plane currently unions all components into one fit, so a small
    sconce hole inherits a plane dominated by the big actor blob --
    per_component=True opt-in fixes that. Returns (labels, n_components,
    regions) where labels is the (H,W) int label image from
    cv2.connectedComponents (0=background), n_components is the count of
    foreground components, and regions = {label_id: (inner, ring, outer)}."""
    m8 = mask.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(m8, connectivity=8)
    regions = {}
    for lbl in range(1, n_labels):
        comp = (labels == lbl).astype(np.uint8)
        inner = cv2.dilate(comp, np.ones((inner_px, inner_px), np.uint8)).astype(bool)
        outer = cv2.dilate(comp, np.ones((outer_px, outer_px), np.uint8)).astype(bool)
        regions[lbl] = (inner, outer & ~inner, outer)
    return labels, n_labels - 1, regions


def build_component_tracks(labels_seq, iou_thresh=0.1):
    """Build component IDENTITY TRACKS across time by greedily matching each
    frame's components to the next frame's by mask IoU (mutual best match,
    descending IoU, above iou_thresh). NO gap bridging: component identity is
    NOT stable across frames in this sequence -- components appear/disappear
    (e.g. a new one entering near frame 105) -- so an unmatched component
    simply ends its track, and an unmatched next-frame component starts a new
    one. This is deliberate: per-component coefficients can only be
    savgol-smoothed over time along a track's own stable lifetime, never
    across an identity change (that was one of the originally diagnosed
    ring_plane failure causes -- a mask-composition transition popping the
    global fit). labels_seq: list of T (H,W) int label images (0=background),
    one per frame, as produced by component_regions. Returns a list of
    tracks: [{"frames": [f0,f1,...], "labels": [l0,l1,...]}, ...]."""
    tracks = []
    active = {}  # label id in previous frame -> index into `tracks`
    prev_img = None
    for fr, cur in enumerate(labels_seq):
        cur_ids = [int(l) for l in np.unique(cur) if l != 0]
        if prev_img is None:
            new_active = {}
            for lid in cur_ids:
                tracks.append({"frames": [fr], "labels": [lid]})
                new_active[lid] = len(tracks) - 1
            active = new_active
            prev_img = cur
            continue
        prev_ids = list(active.keys())
        cand = []
        for pid in prev_ids:
            pmask = (prev_img == pid)
            parea = int(pmask.sum())
            if parea == 0:
                continue
            for cid in cur_ids:
                cmask = (cur == cid)
                inter = int(np.logical_and(pmask, cmask).sum())
                if inter == 0:
                    continue
                union = parea + int(cmask.sum()) - inter
                iou = inter / union
                if iou > iou_thresh:
                    cand.append((iou, pid, cid))
        cand.sort(key=lambda t: t[0], reverse=True)
        matched_prev, matched_cur, new_active = set(), set(), {}
        for iou, pid, cid in cand:
            if pid in matched_prev or cid in matched_cur:
                continue
            matched_prev.add(pid); matched_cur.add(cid)
            tidx = active[pid]
            tracks[tidx]["frames"].append(fr)
            tracks[tidx]["labels"].append(cid)
            new_active[cid] = tidx
        for cid in cur_ids:
            if cid not in matched_cur:
                tracks.append({"frames": [fr], "labels": [cid]})
                new_active[cid] = len(tracks) - 1
        active = new_active
        prev_img = cur
    return tracks


def fit_and_smooth_tracks(track, per_frame_fit_fn, window=9, polyorder=2):
    """track: one {"frames":[...], "labels":[...]} dict from
    build_component_tracks. per_frame_fit_fn(frame_idx, label_id) -> raw
    coefficient array for that component in that frame (affine (2,3) or
    homography (3,3), caller-supplied, this function is agnostic to which).
    Savgol-smooths along the track's OWN lifetime only -- never across an
    identity change. If the track is shorter than `window`, falls back to the
    largest odd window <= track length that is still > polyorder; if even
    that is impossible (track too short), returns the raw per-frame fits
    unsmoothed. Prints a one-line report per track (frame range + length)."""
    frames, labels = track["frames"], track["labels"]
    n = len(frames)
    raw = np.stack([per_frame_fit_fn(fr, lid) for fr, lid in zip(frames, labels)])
    print(f"[hole fill] component track: frames {frames[0]}-{frames[-1]} (len {n})", flush=True)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w > polyorder and w >= 1:
        return savgol_filter(raw, w, polyorder, axis=0)
    return raw


def apply_components(mx, my, regions_this_frame, coefs_this_frame, model="affine", feather_px=180):
    """Blend ALL active components into (mx,my) in one pass. Where two
    components' `outer` regions overlap, do NOT let one overwrite the other:
    accumulate each component's (weight, weight*predicted_value) into
    per-pixel running sums (weight = the usual distance-transform feather in
    [0,1] from that component's own inner boundary), then for touched pixels:
    predicted = accumulated_weighted_value / accumulated_weight; authority =
    clip(accumulated_weight, 0, 1); output = authority*predicted +
    (1-authority)*original. Pixels untouched by any component's outer region
    are returned bit-identical to the input. regions_this_frame: list of
    (inner,ring,outer) tuples. coefs_this_frame: parallel list of coefficient
    arrays matching `model` ("affine" (2,3) or "homography" (3,3))."""
    H, W = mx.shape
    acc_w = np.zeros((H, W), np.float64)
    acc_x = np.zeros((H, W), np.float64)
    acc_y = np.zeros((H, W), np.float64)
    touched = np.zeros((H, W), bool)
    for (inner, ring, outer), coef in zip(regions_this_frame, coefs_this_frame):
        yy, xx = np.where(outer)
        if len(yy) == 0:
            continue
        if model == "affine":
            px = (xx * coef[0, 0] + yy * coef[0, 1] + coef[0, 2]).astype(np.float64)
            py = (xx * coef[1, 0] + yy * coef[1, 1] + coef[1, 2]).astype(np.float64)
        else:
            pts = np.stack([xx, yy], axis=1).astype(np.float64).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(pts, coef).reshape(-1, 2)
            px, py = proj[:, 0], proj[:, 1]
        dist = cv2.distanceTransform((~inner).astype(np.uint8), cv2.DIST_L2, 3)
        w = np.clip(1.0 - dist / feather_px, 0, 1)[yy, xx]
        acc_w[yy, xx] += w
        acc_x[yy, xx] += w * px
        acc_y[yy, xx] += w * py
        touched[yy, xx] = True
    mx2, my2 = mx.copy(), my.copy()
    valid = touched & (acc_w > 1e-9)
    authority = np.clip(acc_w, 0, 1)
    pred_x = np.zeros((H, W)); pred_y = np.zeros((H, W))
    pred_x[valid] = acc_x[valid] / acc_w[valid]
    pred_y[valid] = acc_y[valid] / acc_w[valid]
    mx2[valid] = (authority[valid] * pred_x[valid] + (1 - authority[valid]) * mx[valid]).astype(np.float32)
    my2[valid] = (authority[valid] * pred_y[valid] + (1 - authority[valid]) * my[valid]).astype(np.float32)
    return mx2, my2


def harmonic_extend(fields, region):
    """Solve the Laplace equation inside `region` (bool HxW) for each array in
    `fields`, Dirichlet boundary from adjacent outside pixels, Neumann at the
    image border. One sparse factorization shared across fields."""
    h, w = region.shape
    ys, xs = np.where(region)
    n = len(ys)
    if n == 0:
        return [f.copy() for f in fields]
    idx = -np.ones((h, w), np.int64)
    idx[ys, xs] = np.arange(n)
    diag = np.zeros(n)
    rc, cc, vv = [], [], []
    bs = [np.zeros(n) for _ in fields]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nyc, nxc = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
        nidx = np.where(ok, idx[nyc, nxc], -1)
        diag += ok
        inreg = ok & (nidx >= 0)
        rc.append(np.arange(n)[inreg]); cc.append(nidx[inreg]); vv.append(-np.ones(inreg.sum()))
        bnd = ok & (nidx < 0)
        for f, b in zip(fields, bs):
            b[bnd] += f[nyc[bnd], nxc[bnd]]
    A = sp.csc_matrix((np.concatenate([diag] + vv),
                       (np.concatenate([np.arange(n)] + rc),
                        np.concatenate([np.arange(n)] + cc))), shape=(n, n))
    lu = splu(A)
    outs = []
    for f, b in zip(fields, bs):
        g = f.copy()
        g[ys, xs] = lu.solve(b)
        outs.append(g)
    return outs


def apply_harmonic(mx, my, mask, inner_px=61, downsample=4, seam_px=16,
                   ring_coef=None):
    """Replace (mx,my) inside dilate(mask, inner_px) with the harmonic
    extension of the surrounding field, solved on a x`downsample` grid as a
    residual over `ring_coef` (a fit_ring_plane prior; fitted here if None),
    with a `seam_px` distance-transform blend at the hole rim. Pixels outside
    the dilated mask are returned bit-identical."""
    H, W = mx.shape
    inner = cv2.dilate(mask.astype(np.uint8),
                       np.ones((inner_px, inner_px), np.uint8)).astype(bool)
    if ring_coef is None:
        _, ring, _ = ring_regions(mask, inner_px)
        ring_coef = fit_ring_plane(mx, my, ring)
    if ring_coef is None:
        return mx, my
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    px = xx * ring_coef[0, 0] + yy * ring_coef[0, 1] + ring_coef[0, 2]
    py = xx * ring_coef[1, 0] + yy * ring_coef[1, 1] + ring_coef[1, 2]
    h4, w4 = H // downsample, W // downsample
    r4 = [cv2.resize(r, (w4, h4), interpolation=cv2.INTER_AREA)
          for r in (mx - px, my - py)]
    reg4 = cv2.resize(inner.astype(np.uint8), (w4, h4),
                      interpolation=cv2.INTER_NEAREST) > 0
    s4 = harmonic_extend(r4, reg4)
    su = [cv2.resize(s, (W, H), interpolation=cv2.INTER_CUBIC) for s in s4]
    dt = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
    wgt = np.clip(dt / seam_px, 0, 1)
    return (wgt * (px + su[0]) + (1 - wgt) * mx).astype(np.float32), \
           (wgt * (py + su[1]) + (1 - wgt) * my).astype(np.float32)

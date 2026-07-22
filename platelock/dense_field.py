"""
platelock.dense_field — reconstruct a DENSE backward warp from RoMa anchors.

Per the crystallized spec: RoMa is an ANCHOR generator, not a directly-usable
dense warp (on this shot only ~3-4% of pixels clear cert>0.5). We therefore:

  RoMa dense corr + certainty
    -> take the A-side half (grid over ORIGINAL; proven ~identity grid)
    -> reject low-confidence / unsupported pixels (threshold OR top-k)
    -> fit a global affine prior from the surviving anchors
    -> add a LOCAL residual (RBF/TPS or confidence-diffused) so each region
       is nudged onto its true position, generated regions riding their support
    -> dense backward map (for each ORIGINAL pixel -> sample coord in CLEAN)

Direction (PROVEN on f115): for original pixel (x,y), map gives clean (u,v);
render = geometry.apply_backward_map(clean, map_x, map_y). This is the correct
backward-map / STMap convention from geometry.py.

numpy / scipy / cv2 only. RoMa tensors are decoded via model.to_pixel_coordinates.
"""

import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator, griddata
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from platelock import geometry as g


# ---------------------------------------------------------------- anchors ----
def extract_anchors(warp, cert, H, W, model):
    """Decode RoMa output -> dense anchors on the ORIGINAL grid.

    Returns orig_xy (N,2) pixel coords in ORIGINAL, clean_xy (N,2) pixel coords
    in CLEAN, c (N,) certainty. N = gh*gw over RoMa's A-side grid.
    """
    kA, kB = model.to_pixel_coordinates(warp, H, W, H, W)   # (1,gh,2gw,2)
    kA = kA[0].detach().cpu().numpy()
    kB = kB[0].detach().cpu().numpy()
    ce = cert[0].detach().cpu().numpy()
    gw = kA.shape[1] // 2                                     # A-side = first half
    orig = kA[:, :gw, :].reshape(-1, 2).astype(np.float32)
    clean = kB[:, :gw, :].reshape(-1, 2).astype(np.float32)
    c = ce[:, :gw].reshape(-1).astype(np.float32)
    return orig, clean, c


def filter_anchors(orig, clean, c, thresh=None, topk_frac=None):
    """Keep anchors by certainty threshold, OR the top `topk_frac` by certainty.
    Exactly one of thresh/topk_frac should be given."""
    if topk_frac is not None:
        k = max(16, int(len(c) * topk_frac))
        idx = np.argpartition(-c, k - 1)[:k]
        keep = np.zeros(len(c), bool); keep[idx] = True
    else:
        keep = c >= (thresh if thresh is not None else 0.5)
    return orig[keep], clean[keep], c[keep], keep


# ------------------------------------------------------------ reconstruct ----
def fit_affine(orig, clean, thresh=3.0):
    """Robust affine mapping ORIGINAL -> CLEAN (backward-map convention).
    Returns (M 2x3 or None, inlier_bool)."""
    if len(orig) < 3:
        return None, np.zeros(len(orig), bool)
    M, inl = cv2.estimateAffine2D(orig, clean, method=cv2.RANSAC,
                                  ransacReprojThreshold=thresh,
                                  maxIters=5000, confidence=0.999)
    inl = np.zeros(len(orig), bool) if inl is None else inl.ravel().astype(bool)
    return M, inl


def affine_field(M, H, W):
    """Dense backward map from a global affine (whole-frame — the corner-pin
    baseline we are trying to BEAT, kept for comparison)."""
    if M is None:
        return g.identity_map(H, W)
    return g.affine_to_backward_map(M, H, W)


def rbf_residual_field(orig, clean, H, W, M, max_anchors=1600, smoothing=1.0):
    """affine PRIOR + thin-plate-spline residual on the leftover displacement.
    Each region is nudged locally off the global fit. TPS is O(N^3) so anchors
    are subsampled to max_anchors."""
    base_x, base_y = affine_field(M, H, W)
    if M is None or len(orig) < 8:
        return base_x, base_y
    # residual at anchors = true clean coord - affine-predicted clean coord
    pred = (orig @ M[:, :2].T) + M[:, 2]
    resid = clean - pred                                     # (N,2)
    if len(orig) > max_anchors:
        sel = np.random.default_rng(0).choice(len(orig), max_anchors, replace=False)
        orig, resid = orig[sel], resid[sel]
    rbf = RBFInterpolator(orig, resid, kernel="thin_plate_spline", smoothing=smoothing)
    ys, xs = np.mgrid[0:H, 0:W]
    grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    rf = rbf(grid).reshape(H, W, 2).astype(np.float32)
    return (base_x + rf[..., 0]).astype(np.float32), (base_y + rf[..., 1]).astype(np.float32)


def diffused_field(orig, clean, H, W, smooth_px=25, subsample=6000):
    """True dense fill: scatter DISPLACEMENT (clean-orig) at anchors, interpolate
    across the frame (linear + nearest fallback), Gaussian-smooth. Unsupported
    regions (bare wall, inserted content) inherit the surrounding surface motion.
    v1 is isotropic (not edge-aware) — see notes for v2 guided filtering."""
    if len(orig) < 8:
        return g.identity_map(H, W)
    disp = clean - orig                                      # (N,2)
    if len(orig) > subsample:
        sel = np.random.default_rng(0).choice(len(orig), subsample, replace=False)
        orig, disp = orig[sel], disp[sel]
    ys, xs = np.mgrid[0:H, 0:W]
    gpts = np.stack([xs.ravel(), ys.ravel()], 1)
    lin = griddata(orig, disp, gpts, method="linear")
    nn = griddata(orig, disp, gpts, method="nearest")
    lin[np.isnan(lin[:, 0])] = nn[np.isnan(lin[:, 0])]       # fill convex-hull exterior
    field = lin.reshape(H, W, 2).astype(np.float32)
    if smooth_px > 0:
        k = smooth_px | 1
        field = cv2.GaussianBlur(field, (k, k), 0)
    idx, idy = g.identity_map(H, W)
    return (idx + field[..., 0]).astype(np.float32), (idy + field[..., 1]).astype(np.float32)


def roi_balanced_topk(orig, clean, c, H, W, tiles=(8, 5), frac=0.05, max_total=5000):
    """Spatially-balanced anchors: split the frame into a tile grid and keep the
    top `frac` by certainty WITHIN each tile, so one strong window can't own the
    warp. Caps total at max_total. Returns (orig, clean, c) subset."""
    tx, ty = tiles
    col = np.clip((orig[:, 0] / W * tx).astype(int), 0, tx - 1)
    row = np.clip((orig[:, 1] / H * ty).astype(int), 0, ty - 1)
    tile = row * tx + col
    per_tile = max(8, int(max_total / (tx * ty)))
    keep = []
    for t in range(tx * ty):
        idx = np.where(tile == t)[0]
        if len(idx) == 0:
            continue
        k = max(1, min(len(idx), int(len(idx) * frac), per_tile))
        top = idx[np.argpartition(-c[idx], k - 1)[:k]]
        keep.append(top)
    if not keep:
        return orig[:0], clean[:0], c[:0]
    keep = np.concatenate(keep)
    return orig[keep], clean[keep], c[keep]


def add_boundary_anchors(orig, clean, M, H, W, grid=(7, 4)):
    """Append border anchors whose clean coord = affine(orig): pins the residual
    field to ~0 at the frame edge so extrapolated regions don't flap."""
    if M is None:
        return orig, clean
    gx, gy = grid
    xs = np.linspace(0, W - 1, gx); ys = np.linspace(0, H - 1, gy)
    bx, by = np.meshgrid(xs, ys)
    border = (bx == 0) | (bx == W - 1) | (by == 0) | (by == H - 1)
    bpts = np.stack([bx[border], by[border]], 1).astype(np.float32)
    bclean = (bpts @ M[:, :2].T) + M[:, 2]                    # residual 0
    return np.vstack([orig, bpts]), np.vstack([clean, bclean])


def assemble_map(M, cres, H, W):
    """Rebuild a dense backward map from an affine prior + control-grid residual.
    `cres` is the coarse residual lattice (gy,gx,2) from reconstruct_decomposed
    (or None -> affine only). This is the single assembly path shared by the
    per-frame mainline and the temporal smoother, so smoothed (M, cres) rebuild
    identically to unsmoothed ones."""
    base_x, base_y = affine_field(M, H, W)
    if cres is None:
        return base_x, base_y
    res_full = cv2.resize(cres.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return (base_x + res_full[..., 0]).astype(np.float32), (base_y + res_full[..., 1]).astype(np.float32)


def coverage_x_right(trO, vi, T, percentile=99.5, temporal_window=11):
    """Per-frame right-hand track-coverage boundary (working-res x), temporally
    smoothed so a clamp anchored to it cannot itself flicker. Robust percentile
    rather than max: a single straggler track must not extend 'coverage'."""
    x = np.array([float(np.percentile(trO[fr][vi[fr]][:, 0], percentile))
                  if vi[fr].any() else 0.0 for fr in range(T)])
    if temporal_window and T >= temporal_window:
        x = savgol_filter(x, temporal_window, 2)
    return x


def damp_cres_beyond_coverage(creses, xcov, W, band_px):
    """Edge clamp: zero the control-grid residual beyond the per-frame track
    coverage boundary, so the uncovered frame margin falls back to the affine
    prior instead of TPS extrapolation (the end-of-sequence right-edge smear —
    between the last real anchors and the sparse affine boundary anchors the
    TPS ramp is unconstrained). Weight is 1 up to xcov-band_px, smoothsteps to
    0 at xcov; beyond data the field is exactly affine."""
    T, gy, gx, _ = creses.shape
    node_x = np.linspace(0, W - 1, gx)
    out = creses.copy()
    for fr in range(T):
        t = np.clip((xcov[fr] - node_x) / float(band_px), 0.0, 1.0)
        w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        out[fr] *= w[None, :, None]
    return out


def reconstruct_decomposed(orig, clean, H, W, M, grid=(160, 84), neighbors=64,
                           smoothing=2.0, max_anchors=5000, boundary=(7, 4)):
    """As reconstruct(), but return the DECOMPOSED pieces (M, cres, grid) instead
    of the assembled map. `cres` is the coarse control-grid residual (gy,gx,2), or
    None when the frame is degenerate (affine-only fallback). Temporal smoothing
    operates on M and cres across frames, then assemble_map() rebuilds each map."""
    gx, gy = grid
    if M is None or len(orig) < 8:
        return M, None, grid
    if len(orig) > max_anchors:
        sel = np.random.default_rng(0).choice(len(orig), max_anchors, replace=False)
        orig, clean = orig[sel], clean[sel]
    if boundary:
        orig, clean = add_boundary_anchors(orig, clean, M, H, W, boundary)
    # de-dup near-identical anchors (harmless, keeps the field pristine)
    _, uniq = np.unique(np.round(orig, 1), axis=0, return_index=True)
    orig, clean = orig[uniq], clean[uniq]
    pred = (orig @ M[:, :2].T) + M[:, 2]
    resid = clean - pred
    nb = min(neighbors, len(orig))
    cxs = np.linspace(0, W - 1, gx); cys = np.linspace(0, H - 1, gy)
    mgx, mgy = np.meshgrid(cxs, cys)
    cpts = np.stack([mgx.ravel(), mgy.ravel()], 1).astype(np.float32)
    # RoMa anchors bunch on vertical window edges (locally collinear) -> the
    # degree-1 TPS monomial matrix can go singular (neighbors= defers the solve
    # to eval time, so it raises there). Keep good frames jitter-FREE; only the
    # frames that actually fail get escalating sub-pixel jitter to restore rank.
    rng = np.random.default_rng(0)
    for jit in (0.0, 0.05, 0.3):
        o2 = orig if jit == 0 else orig + rng.normal(0, jit, orig.shape).astype(np.float32)
        try:
            rbf = RBFInterpolator(o2, resid, kernel="thin_plate_spline",
                                  neighbors=nb, smoothing=smoothing, degree=1)
            cres = rbf(cpts).reshape(gy, gx, 2).astype(np.float32)
            return M, cres, grid
        except np.linalg.LinAlgError:
            continue
    return M, None, grid                     # still degenerate -> affine fallback


def reconstruct(orig, clean, H, W, M, grid=(160, 84), neighbors=64,
                smoothing=2.0, max_anchors=5000, boundary=(7, 4)):
    """FAST control-grid reconstruction (Chat's speed fix): affine prior + local
    TPS residual evaluated on a coarse lattice, then upsampled to full frame.
    Returns dense backward map (map_x, map_y) float32 HxW."""
    M, cres, _ = reconstruct_decomposed(orig, clean, H, W, M, grid=grid,
                                        neighbors=neighbors, smoothing=smoothing,
                                        max_anchors=max_anchors, boundary=boundary)
    return assemble_map(M, cres, H, W)


def field_health(map_x, map_y):
    """Jacobian-based sanity of a backward map. fold_frac = fraction of pixels
    with negative Jacobian determinant (tearing/folding); smoothness = mean
    magnitude of the map's 2nd derivative (lower = smoother)."""
    dxdy, dxdx = np.gradient(map_x)
    dydy, dydx = np.gradient(map_y)
    det = dxdx * dydy - dxdy * dydx
    fold_frac = float((det <= 0).mean())
    lap = np.abs(cv2.Laplacian(map_x, cv2.CV_32F)) + np.abs(cv2.Laplacian(map_y, cv2.CV_32F))
    return {"fold_frac": fold_frac, "smoothness": float(lap.mean()),
            "det_min": float(det.min()), "det_median": float(np.median(det))}


# --------------------------------------------------------------- previews ----
def displacement_preview(map_x, map_y, H, W, step=32, scale=1.0):
    """Quiver-style viz of the backward displacement (map - identity)."""
    idx, idy = g.identity_map(H, W)
    dx, dy = map_x - idx, map_y - idy
    viz = np.full((H, W, 3), 30, np.uint8)
    for y in range(0, H, step):
        for x in range(0, W, step):
            ex, ey = int(x + dx[y, x] * scale), int(y + dy[y, x] * scale)
            cv2.arrowedLine(viz, (x, y), (ex, ey), (0, 220, 0), 1, cv2.LINE_AA, tipLength=0.3)
    return viz


def coverage_weights_perimeter(trO, vi, T, grids, W, H, track_res=None,
                               dist_px=90.0, band_px=120.0, margin_px=240.0,
                               temporal_window=11, cov_k=1):
    """Perimeter coverage clamp weights for a set of control lattices.
    Beyond track coverage both the TPS residual and the microlock correction
    are extrapolation noise -> weight them to 0 so the field falls back to the
    affine prior / zero correction. Border-restricted (margin_px): interior
    nodes always weigh exactly 1.0, so hole-fill territory is untouched. The
    per-node nearest-track distance is savgol-smoothed over time so the clamp
    cannot itself flicker. Returns one (T,gy,gx) float32 weight array per
    (gx,gy) in `grids` (the cres and microlock lattices differ in resolution)."""
    sx = sy = 1.0
    if track_res is not None:
        sx, sy = W / float(track_res[0]), H / float(track_res[1])
    nodes = []
    for gx, gy in grids:
        nxx, nyy = np.meshgrid(np.linspace(0, W - 1, gx), np.linspace(0, H - 1, gy))
        nodes.append((nxx, nyy, np.stack([nxx.ravel(), nyy.ravel()], 1)))
    dists = [np.full((T, gy, gx), 1e6, np.float64) for gx, gy in grids]
    for fr in range(T):
        pts = trO[fr][vi[fr]].astype(np.float64)
        if len(pts) < 8:
            continue                     # degenerate frame -> fully uncovered
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        tree = cKDTree(pts)              # one tree per frame, queried per grid
        k = min(cov_k, len(pts))         # cov_k>1: distance to k-th nearest ->
        for gi, (gx, gy) in enumerate(grids):
            d, _ = tree.query(nodes[gi][2], k=k)   # "enough local data", not
            if k > 1:                              # just one straggler track
                d = d[:, -1]
            dists[gi][fr] = d.reshape(gy, gx)

    def _ss(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    out = []
    for gi, (gx, gy) in enumerate(grids):
        d = dists[gi]
        if temporal_window and T >= temporal_window:
            d = savgol_filter(d, temporal_window, 2, axis=0)
        w_data = 1.0 - _ss((d - dist_px) / float(band_px))
        nxx, nyy = nodes[gi][0], nodes[gi][1]
        db = np.minimum(np.minimum(nxx, W - 1 - nxx), np.minimum(nyy, H - 1 - nyy))
        border = 1.0 - _ss(db / float(margin_px))   # 1 at the frame edge, 0 interior
        out.append((1.0 - border * (1.0 - w_data)).astype(np.float32))
    return out


def damp_lattice(fields, weights):
    """Damp a (T,gy,gx,C) lattice by per-node weights (T,gy,gx) -> float32 copy."""
    return (fields * weights[..., None]).astype(np.float32)

"""Shared robust correspondence fitter for video VFX plate lock."""

import logging
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


def ransac_affine(src, dst, thresh=2.0, max_iters=2000, confidence=0.99):
    """Full affine RANSAC (6 DoF). Returns (M_2x3 float32, bool_inlier_mask)."""
    if len(src) < 3:
        return np.eye(2, 3, dtype=np.float32), np.zeros(len(src), dtype=bool)
    M, inliers = cv2.estimateAffine2D(
        np.float32(src), np.float32(dst),
        method=cv2.RANSAC,
        ransacReprojThreshold=thresh,
        maxIters=max_iters,
        confidence=confidence,
    )
    if M is None:
        return np.eye(2, 3, dtype=np.float32), np.zeros(len(src), dtype=bool)
    return M.astype(np.float32), inliers.ravel().astype(bool)


def affine_to_flow(M, H, W):
    """
    Backward flow from affine M that maps src→dst.

    For each output pixel (x, y) in dst space: src = M^{-1} @ [x, y, 1].
    flow[y, x] = (x_src − x, y_src − y).
    """
    try:
        M_inv = cv2.invertAffineTransform(M)
    except cv2.error:
        return np.zeros((H, W, 2), dtype=np.float32)
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy, np.ones_like(xx)], axis=-1)  # (H, W, 3)
    src = np.einsum("ij,hwj->hwi", M_inv, grid)            # (H, W, 2)
    return (src - grid[..., :2]).astype(np.float32)


def piecewise_affine_flow(src_pts, dst_pts, H, W, global_fallback=None):
    """
    (H, W, 2) backward flow via Delaunay triangulation in dst space.

    Per triangle: dst→src affine applied to all pixels in that triangle.
    Pixels outside the convex hull take global_fallback (or zero).
    Returns None if triangulation fails.

    Expects inlier-only points — do not pass raw unfiltered correspondences.
    """
    if len(dst_pts) < 3:
        return None
    try:
        tri = Delaunay(np.float32(dst_pts))
    except Exception as e:
        logger.warning("Delaunay failed: %s", e)
        return None

    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))  # (H*W, 2)

    simplex_ids = tri.find_simplex(points)
    flow = (global_fallback.reshape(-1, 2).copy()
            if global_fallback is not None
            else np.zeros((H * W, 2), dtype=np.float32))

    for sid in np.unique(simplex_ids[simplex_ids >= 0]):
        pix_mask = simplex_ids == sid
        dst_tri  = np.float32(dst_pts)[tri.simplices[sid]]
        src_tri  = np.float32(src_pts)[tri.simplices[sid]]
        try:
            M_tri = cv2.getAffineTransform(dst_tri, src_tri)
        except cv2.error:
            continue
        tri_pts = points[pix_mask]                                            # (K, 2)
        tri_hom = np.column_stack([tri_pts, np.ones(len(tri_pts), np.float32)])
        flow[pix_mask] = tri_hom @ M_tri.T - tri_pts

    return flow.reshape(H, W, 2).astype(np.float32)


def fit_robust(src_pts, dst_pts, H, W,
               ransac_thresh=2.0, min_piecewise_inliers=15,
               min_piecewise_ratio=0.4, max_iters=2000):
    """
    Best-fit backward flow from src (clean) → dst (original) correspondences.

    src_pts : (N, 2)  positions in clean frame (x, y)
    dst_pts : (N, 2)  positions in original frame (x, y)

    Returns
    -------
    flow  : (H, W, 2)  flow[y, x] = (x_clean − x, y_clean − y)
    debug : dict       model_used, n_inliers, inlier_ratio, median_residual_px
    """
    N = len(src_pts)
    debug = dict(model_used="identity", n_inliers=0, inlier_ratio=0.0,
                 median_residual_px=float("inf"))
    if N < 3:
        return np.zeros((H, W, 2), np.float32), debug

    M, inlier_mask = ransac_affine(src_pts, dst_pts, thresh=ransac_thresh, max_iters=max_iters)
    n_in  = int(inlier_mask.sum())
    ratio = n_in / N

    # Median residual on inliers only
    if n_in >= 3:
        src_in  = np.float32(src_pts[inlier_mask])
        dst_in  = np.float32(dst_pts[inlier_mask])
        src_hom = np.column_stack([src_in, np.ones(n_in, np.float32)])
        dst_hat = src_hom @ M.T
        med_res = float(np.median(np.linalg.norm(dst_hat - dst_in, axis=1)))
    else:
        med_res = float("inf")

    debug.update(n_inliers=n_in, inlier_ratio=ratio, median_residual_px=med_res)
    if n_in < 3:
        return np.zeros((H, W, 2), np.float32), debug

    # Global affine always computed — used as outside-hull fallback in piecewise mode
    global_flow = affine_to_flow(M, H, W)

    if n_in >= min_piecewise_inliers and ratio >= min_piecewise_ratio:
        # Pass inlier-only points so hallucinated-region outliers don't pollute the mesh
        flow = piecewise_affine_flow(
            src_pts[inlier_mask], dst_pts[inlier_mask], H, W,
            global_fallback=global_flow,
        )
        if flow is None:
            flow = global_flow
            debug["model_used"] = "affine"
        else:
            debug["model_used"] = "piecewise_affine"
    else:
        flow = global_flow
        debug["model_used"] = "affine"

    return flow, debug


def smooth_flows_temporal(flows, sigma=1.5):
    """Gaussian temporal smoothing over a list of (H, W, 2) flow fields."""
    if len(flows) < 3 or sigma <= 0:
        return flows
    stacked  = np.stack(flows, axis=0).astype(np.float32)  # (T, H, W, 2)
    smoothed = gaussian_filter1d(stacked, sigma=sigma, axis=0)
    return [smoothed[i] for i in range(len(flows))]

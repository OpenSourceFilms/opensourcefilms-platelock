"""
platelock.warp_fit — fit a BACKWARD warp from correspondences.

Correspondences are (orig_pts, clean_pts): matched points in original-plate and
clean-plate working coords. We fit M mapping ORIGINAL -> CLEAN (i.e. the backward
map convention from geometry.py: for an output/original pixel, where to sample in
clean). So cv2 estimators take from=orig_pts, to=clean_pts.

Falsification-stage models: robust similarity and full affine. (Smoothed TPS/RBF
residual + boundary anchors are added once RoMa earns the backbone.)
"""

import numpy as np
import cv2


def fit_similarity(orig_pts, clean_pts, thresh=3.0):
    """4-DoF (rot+uniform scale+translation). Returns (M 2x3, inlier_bool)."""
    M, inl = cv2.estimateAffinePartial2D(
        np.asarray(orig_pts, np.float32), np.asarray(clean_pts, np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=5000, confidence=0.999)
    inl = np.zeros(len(orig_pts), bool) if inl is None else inl.ravel().astype(bool)
    return M, inl


def fit_affine(orig_pts, clean_pts, thresh=3.0):
    """6-DoF affine. Returns (M 2x3, inlier_bool)."""
    M, inl = cv2.estimateAffine2D(
        np.asarray(orig_pts, np.float32), np.asarray(clean_pts, np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=5000, confidence=0.999)
    inl = np.zeros(len(orig_pts), bool) if inl is None else inl.ravel().astype(bool)
    return M, inl


def apply_affine(M, pts):
    pts = np.asarray(pts, np.float32)
    return (pts @ M[:, :2].T) + M[:, 2]


def reproj_error(M, orig_pts, clean_pts, inliers=None):
    """Median reprojection error (px) of M(orig)->clean, over inliers if given."""
    if M is None:
        return float("inf")
    pred = apply_affine(M, orig_pts)
    err = np.linalg.norm(pred - np.asarray(clean_pts, np.float32), axis=1)
    if inliers is not None and inliers.any():
        err = err[inliers]
    return float(np.median(err)) if len(err) else float("inf")


def hull_coverage(pts, H, W):
    """Convex-hull area of pts as a fraction of the frame — measures how much of
    the image the constraints actually span (low = warp is under-constrained)."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull) / (H * W))

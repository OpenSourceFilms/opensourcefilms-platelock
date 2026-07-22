"""
platelock.masks — tool-agnostic mask algebra.
Mask exclusion: orig_mask excludes anchors/metric-pixels in ORIGINAL coordinates (real objects);
clean_mask excludes in CLEAN coordinates (hallucinated objects). They are independent.
numpy / cv2 only.
"""

import cv2
import numpy as np
import os


def resize_mask(mask: np.ndarray, H: int, W: int):
    """Return bool mask resized to (H, W) using nearest-neighbour."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def dilate(mask: np.ndarray, px: int):
    """Dilate binary mask with elliptical kernel of size px (odd). px <= 0 returns unchanged."""
    if not np.issubdtype(mask.dtype, np.bool_):
        mask = mask.astype(bool)
    if px <= 0:
        return mask.copy()
    if px % 2 == 0:
        px += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def feather(mask: np.ndarray, px: int):
    """Return float32 alpha [0,1] with soft falloff outside mask over px pixels."""
    if not np.issubdtype(mask.dtype, np.bool_):
        mask = mask.astype(bool)
    if px <= 0:
        return mask.astype(np.float32)
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1.0 - dist / px, 0.0, 1.0).astype(np.float32)
    return alpha


def save_mask(path: str, mask: np.ndarray):
    """Save bool mask as single-channel PNG (0/255) atomically."""
    mask_u8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    cv2.imwrite(tmp, mask_u8)
    os.replace(tmp, path)


def load_mask(path: str):
    """Load bool mask from single-channel PNG (0/255)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return img > 127


def combine(masks: list):
    """Logical OR of a list of same-shape bool masks. Raises ValueError if empty."""
    if not masks:
        raise ValueError("combine(): masks list is empty")
    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return out


def exclude_anchors(orig, clean, c, orig_mask=None, clean_mask=None):
    """
    Filter correspondences where anchor falls inside a mask.
    orig / clean : (N,2) float32 (x,y) coordinates.
    orig_mask / clean_mask : (H,W) bool arrays (row,col indexing).
    Returns (orig[keep], clean[keep], c[keep], keep).
    """
    keep = np.ones(len(orig), dtype=bool)
    if orig_mask is not None:
        H, W = orig_mask.shape
        xs = np.clip(orig[:, 0].astype(int), 0, W - 1)
        ys = np.clip(orig[:, 1].astype(int), 0, H - 1)
        keep &= ~orig_mask[ys, xs]
    if clean_mask is not None:
        H, W = clean_mask.shape
        xs = np.clip(clean[:, 0].astype(int), 0, W - 1)
        ys = np.clip(clean[:, 1].astype(int), 0, H - 1)
        keep &= ~clean_mask[ys, xs]
    return orig[keep], clean[keep], c[keep], keep

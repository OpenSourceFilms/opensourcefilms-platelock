"""
Per-frame colour normalisation: match clean plate colour statistics to original.

Uses Reinhard LAB transfer (match mean+std per LAB channel).  Result is used
only for feature-finding; the final warp is applied to the un-normalised plate.

Robust to large intentionally-different regions (flames, actors, paintouts):
the per-channel stats are computed with sigma-clipping so outlier pixels don't
skew the transfer.
"""
import cv2
import numpy as np


def _sigma_clip_stats(arr, n_sigma=2.5, max_iter=5):
    """Mean and std after iterative sigma-clipping."""
    vals = arr.ravel().astype(np.float64)
    for _ in range(max_iter):
        mu, sd = vals.mean(), vals.std()
        if sd < 1e-6:
            break
        vals = vals[np.abs(vals - mu) < n_sigma * sd]
    return float(vals.mean()), float(vals.std()) if len(vals) > 1 else (float(mu), 1.0)


def colour_normalise(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """
    Return src_bgr colour-matched to ref_bgr using Reinhard LAB transfer.

    Both inputs are uint8 BGR.  Output is uint8 BGR.
    Stats are estimated with sigma-clipping to ignore large outlier regions.
    """
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    out_lab = src_lab.copy()
    for c in range(3):
        s_mu, s_sd = _sigma_clip_stats(src_lab[:, :, c])
        r_mu, r_sd = _sigma_clip_stats(ref_lab[:, :, c])
        if s_sd < 1e-6:
            continue
        out_lab[:, :, c] = (src_lab[:, :, c] - s_mu) * (r_sd / s_sd) + r_mu

    out_lab[:, :, 0] = np.clip(out_lab[:, :, 0], 0, 255)
    out_lab[:, :, 1] = np.clip(out_lab[:, :, 1] + 128, 0, 255) - 128 + 128  # keep AB centred
    out_lab[:, :, 1:] = np.clip(out_lab[:, :, 1:], -128 + 128, 127 + 128)
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def normalise_sequence(cleans, originals):
    """Return colour-normalised copies of cleans matched per-frame to originals."""
    return [colour_normalise(c, o) for c, o in zip(cleans, originals)]

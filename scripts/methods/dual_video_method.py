"""
Dual-video plate lock — mirrors CoTracker's architecture using classical optical flow.

CoTracker works because it never compares orig to clean directly. It tracks
within each video independently and computes the correction from the trajectory
difference. This method does the same thing with LK+RANSAC:

  1. ECC on orig[0] vs clean[0]        -> initial spatial offset A (near-identical)
  2. LK+RANSAC on orig[i-1]->orig[i]   -> per-frame orig camera delta
     (actor is a RANSAC outlier — it moves differently from the background)
  3. LK+RANSAC on clean[i-1]->clean[i] -> per-frame clean camera delta
     (no actor in clean, all pixels are reliable background)
  4. correction[i] = A + cumsum(orig_delta[0..i]) - cumsum(clean_delta[0..i])
  5. Temporal smooth the cumulative warp signal
  6. Apply: warpAffine(clean[i], M, flags=INTER_LINEAR|WARP_INVERSE_MAP)
"""
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from utils.flow_utils import compute_bg_mask

logger = logging.getLogger(__name__)


def _phase_translation(frame_a, frame_b):
    """Phase-correlation camera translation between consecutive same-video frames.

    Uses FFT cross-power spectrum to find the dominant global shift.
    Robust to large moving objects (actors) because the background pixels
    outnumber actor pixels and dominate the correlation peak.
    Returns (tx, ty).
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    shift, _ = cv2.phaseCorrelate(gray_a, gray_b)
    return float(shift[0]), float(shift[1])


def _ecc_frame0(orig0, clean0, num_iters=500, eps=1e-6):
    """ECC translation on orig[0] vs clean[0] (near-identical WAN conditioning frames)."""
    gray_o = cv2.cvtColor(orig0, cv2.COLOR_RGB2GRAY)
    gray_c = cv2.cvtColor(clean0, cv2.COLOR_RGB2GRAY)
    tmask = compute_bg_mask(orig0, clean0, threshold=20, dilation_px=15)
    imask = np.full_like(tmask, 255)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters, eps)
    try:
        _, warp = cv2.findTransformECCWithMask(
            gray_o, gray_c, tmask, imask, warp, cv2.MOTION_TRANSLATION, criteria, 5)
        tx, ty = float(warp[0, 2]), float(warp[1, 2])
    except cv2.error as e:
        logger.warning("ECC frame-0 failed (%s) — using identity", e)
        tx, ty = 0.0, 0.0
    return tx, ty


class DualVideoMethod:
    def __init__(self, temporal_smooth_kernel=7, ecc_iters=500):
        self.temporal_smooth_kernel = temporal_smooth_kernel
        self.ecc_iters = ecc_iters

    def process_sequence(self, originals, cleans, show_progress=True):
        H, W = originals[0].shape[:2]
        n = len(originals)

        # Initial offset: ECC on nearly-identical frame-0 pair
        a_tx, a_ty = _ecc_frame0(originals[0], cleans[0], self.ecc_iters)
        logger.info("Frame-0 ECC offset: tx=%.3f  ty=%.3f", a_tx, a_ty)

        # Per-frame within-video camera deltas
        orig_tx = np.zeros(n, dtype=np.float64)
        orig_ty = np.zeros(n, dtype=np.float64)
        clean_tx = np.zeros(n, dtype=np.float64)
        clean_ty = np.zeros(n, dtype=np.float64)

        it = range(1, n)
        if show_progress:
            it = tqdm(it, total=n - 1, desc="DualVideo")

        for i in it:
            # Phase correlation on within-orig consecutive frames.
            # The background (camera motion) is the dominant signal; the actor
            # is a minority region and doesn't corrupt the FFT correlation peak.
            otx, oty = _phase_translation(originals[i - 1], originals[i])
            orig_tx[i], orig_ty[i] = otx, oty

            # Phase correlation on within-clean consecutive frames.
            # No actor present — all background, very clean signal.
            ctx, cty = _phase_translation(cleans[i - 1], cleans[i])
            clean_tx[i], clean_ty[i] = ctx, cty

            logger.debug("Frame %d: orig(%.3f,%.3f) clean(%.3f,%.3f) diff(%.3f,%.3f)",
                         i, otx, oty, ctx, cty, otx - ctx, oty - cty)

        # Cumulative correction = A + cumsum(orig) - cumsum(clean)
        corr_tx = a_tx + np.cumsum(orig_tx) - np.cumsum(clean_tx)
        corr_ty = a_ty + np.cumsum(orig_ty) - np.cumsum(clean_ty)

        # Temporal smooth
        k = min(self.temporal_smooth_kernel, n if n % 2 == 1 else n - 1)
        if k >= 3:
            corr_tx = medfilt(corr_tx, kernel_size=k)
            corr_ty = medfilt(corr_ty, kernel_size=k)

        logger.info("Correction tx [%.2f, %.2f]  ty [%.2f, %.2f]",
                    corr_tx.min(), corr_tx.max(), corr_ty.min(), corr_ty.max())

        # Apply warps
        warped_frames, smoothed_warps = [], []
        for i in range(n):
            M = np.eye(2, 3, dtype=np.float32)
            M[0, 2] = float(corr_tx[i])
            M[1, 2] = float(corr_ty[i])
            smoothed_warps.append(M)
            aligned = cv2.warpAffine(cleans[i], M, (W, H),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)
            warped_frames.append(aligned)

        return {"warped_frames": warped_frames, "flows": None,
                "warps": smoothed_warps, "method_name": "dual_video"}

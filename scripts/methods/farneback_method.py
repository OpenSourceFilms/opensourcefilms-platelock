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


class FarnebackMethod:
    """Plate lock via Lucas-Kanade sparse tracking + RANSAC (Euclidean/partial-affine).

    Using Farneback dense flow to compare orig-to-clean directly fails: the 22% of
    pixels changed by AI inpainting overwhelm the sub-pixel camera motion signal and
    Farneback returns near-zero median flow. Accumulating orig-to-orig camera motion
    is wrong for timestamp-aligned clean frames where each clean[i] is already the
    AI clean version of the same timestamp as orig[i].

    Correct approach for timestamp-aligned pairs:
      1. Detect background keypoints in orig (masked to exclude AI-changed region).
      2. Track them into the corresponding clean frame via LK pyramid optical flow.
      3. Use RANSAC to reject outlier matches (AI texture differences cause bad tracks).
      4. Fit a partial-affine (translation + rotation + uniform scale) from inliers.
      5. Temporally smooth the translation components to suppress per-frame jitter.
    """

    def __init__(self,
                 max_corners: int = 2000,
                 quality_level: float = 0.01,
                 min_distance: int = 12,
                 lk_win_size: int = 21,
                 lk_max_level: int = 4,
                 ransac_threshold: float = 2.0,
                 temporal_smooth_kernel: int = 5,
                 max_jump_px: float = 0.5):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.ransac_threshold = ransac_threshold
        self.temporal_smooth_kernel = temporal_smooth_kernel
        self.max_jump_px = max_jump_px

    def _align_frame(self, original: np.ndarray, clean: np.ndarray
                     ) -> tuple[np.ndarray, dict]:
        """LK+RANSAC translation lock for one frame pair. Returns (warp_2x3, info)."""
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        gray_clean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)

        # Detect keypoints only in reliable background of original
        bg_mask = compute_bg_mask(original, clean, threshold=30, dilation_px=20)

        pts0 = cv2.goodFeaturesToTrack(
            gray_orig,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=bg_mask,
        )
        if pts0 is None or len(pts0) < 6:
            logger.warning("Too few keypoints (%d); using identity", 0 if pts0 is None else len(pts0))
            return np.eye(2, 3, dtype=np.float32), {"converged": False, "n_inliers": 0}

        lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        pts1, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_orig, gray_clean, pts0, None,
            winSize=(self.lk_win_size, self.lk_win_size),
            maxLevel=self.lk_max_level,
            criteria=lk_criteria,
        )

        good0 = pts0[status[:, 0] == 1].reshape(-1, 2)
        good1 = pts1[status[:, 0] == 1].reshape(-1, 2)

        if len(good0) < 6:
            logger.warning("Too few tracked points (%d); using identity", len(good0))
            return np.eye(2, 3, dtype=np.float32), {"converged": False, "n_inliers": len(good0)}

        # Partial affine (translation + rotation + uniform scale) with RANSAC
        # to robustly reject AI-texture outlier tracks
        M, inliers = cv2.estimateAffinePartial2D(
            good0, good1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=5000,
            confidence=0.995,
            refineIters=20,
        )

        if M is None:
            logger.warning("RANSAC failed; using identity")
            return np.eye(2, 3, dtype=np.float32), {"converged": False, "n_inliers": 0}

        n_inliers = int(inliers.sum()) if inliers is not None else 0
        logger.debug("LK+RANSAC: %d/%d inliers  tx=%.3f ty=%.3f",
                     n_inliers, len(good0), M[0, 2], M[1, 2])

        return M.astype(np.float32), {
            "converged": True, "n_inliers": n_inliers,
            "n_tracked": len(good0), "tx": float(M[0, 2]), "ty": float(M[1, 2]),
        }

    def process_sequence(self, originals: list, cleans: list,
                         show_progress: bool = True) -> dict:
        H, W = originals[0].shape[:2]
        raw_warps, infos = [], []

        it = range(len(originals))
        if show_progress:
            it = tqdm(it, total=len(originals), desc="LK+RANSAC")

        for i in it:
            M, info = self._align_frame(originals[i], cleans[i])
            raw_warps.append(M)
            infos.append(info)

        # Temporal smoothing on translation components to suppress per-frame jitter
        n = len(raw_warps)
        tx = np.array([w[0, 2] for w in raw_warps], dtype=np.float32)
        ty = np.array([w[1, 2] for w in raw_warps], dtype=np.float32)
        k = min(self.temporal_smooth_kernel, n if n % 2 == 1 else max(n - 1, 1))
        if k >= 3:
            tx = medfilt(tx, kernel_size=k).astype(np.float32)
            ty = medfilt(ty, kernel_size=k).astype(np.float32)
        for i in range(1, n):
            tx[i] = float(np.clip(tx[i], tx[i-1] - self.max_jump_px, tx[i-1] + self.max_jump_px))
            ty[i] = float(np.clip(ty[i], ty[i-1] - self.max_jump_px, ty[i-1] + self.max_jump_px))

        warped_frames = []
        smoothed_warps = []
        for i, w in enumerate(raw_warps):
            sw = w.copy()
            sw[0, 2] = tx[i]; sw[1, 2] = ty[i]
            smoothed_warps.append(sw)
            aligned = cv2.warpAffine(cleans[i], sw, (W, H),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)
            warped_frames.append(aligned)
            logger.debug("Frame %d: tx=%.3f ty=%.3f inliers=%d",
                         i, tx[i], ty[i], infos[i].get("n_inliers", 0))

        return {"warped_frames": warped_frames, "flows": None,
                "warps": smoothed_warps, "method_name": "farneback"}

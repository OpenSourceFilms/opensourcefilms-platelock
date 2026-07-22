import sys
from pathlib import Path
import logging
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from utils.colour_normalise import colour_normalise as _colour_normalise
from utils.flow_utils import flow_to_stmap

logger = logging.getLogger(__name__)

_SIFT_MAX_W = 1920


class SIFTHomographyMethod:
    """
    SIFT feature matching + RANSAC homography plate-lock.

    Why this works where dense flow / ECC fail:
    - SIFT uses local gradient structure, not pixel values → robust to global
      colour grade differences (WAN AI model output vs. camera original).
    - RANSAC homography votes out regions that don't fit a consistent transform →
      flames, actors, paintouts, AI-added elements become outliers, naturally
      discarded.  No masking required.
    - Temporal smoothing of the homography time-series prevents per-frame jitter.
    """

    def __init__(self, n_features=3000, ratio_threshold=0.75,
                 ransac_threshold=4.0, min_inliers=20,
                 temporal_sigma=1.5, do_colour_normalise=True):
        self.n_features = n_features
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.temporal_sigma = temporal_sigma
        self.do_colour_normalise = do_colour_normalise

    def process_sequence(self, originals, cleans, show_progress=True):
        n_frames = len(originals)
        orig_h, orig_w = originals[0].shape[:2]

        sift = cv2.SIFT_create(nfeatures=self.n_features,
                               contrastThreshold=0.03, edgeThreshold=10)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        H_mats, debug_per_frame = [], []
        it = tqdm(range(n_frames), desc="SIFT matching") if show_progress else range(n_frames)

        for idx in it:
            orig  = originals[idx]
            clean = cleans[idx]

            if self.do_colour_normalise:
                try:
                    clean_in = _colour_normalise(clean, orig)
                except Exception as e:
                    logger.warning("Colour normalise failed frame %d: %s", idx, e)
                    clean_in = clean
            else:
                clean_in = clean

            # Downscale for SIFT detection
            w = orig.shape[1]
            scale = min(1.0, _SIFT_MAX_W / float(w))
            if scale < 1.0:
                nw, nh = int(w * scale), int(orig.shape[0] * scale)
                orig_d  = cv2.resize(orig,     (nw, nh), interpolation=cv2.INTER_AREA)
                clean_d = cv2.resize(clean_in, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                orig_d, clean_d = orig, clean_in

            kp1, des1 = sift.detectAndCompute(cv2.cvtColor(orig_d,  cv2.COLOR_BGR2GRAY), None)
            kp2, des2 = sift.detectAndCompute(cv2.cvtColor(clean_d, cv2.COLOR_BGR2GRAY), None)

            dbg = {"n_kp_orig": len(kp1) if kp1 else 0,
                   "n_kp_clean": len(kp2) if kp2 else 0,
                   "n_matches_ratio": 0, "n_inliers": 0,
                   "inlier_ratio": 0.0, "model_used": "identity"}

            H = np.eye(3, dtype=np.float64)

            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                knn = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in knn if m.distance < self.ratio_threshold * n.distance]
                dbg["n_matches_ratio"] = len(good)

                if len(good) >= 4:
                    # Scale keypoint coords back to full resolution
                    inv = 1.0 / scale
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]) * inv
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]) * inv

                    H_cand, mask = cv2.findHomography(
                        pts2, pts1,  # clean → original
                        cv2.RANSAC,
                        ransacReprojThreshold=self.ransac_threshold,
                        maxIters=3000, confidence=0.995,
                    )
                    if H_cand is not None and mask is not None:
                        n_in = int(mask.sum())
                        dbg["n_inliers"] = n_in
                        dbg["inlier_ratio"] = round(n_in / len(good), 3)
                        if n_in >= self.min_inliers:
                            H = H_cand
                            dbg["model_used"] = "homography"
                        else:
                            logger.warning("Frame %d: only %d inliers, using identity", idx, n_in)

            H_mats.append(H)
            debug_per_frame.append(dbg)

        # Temporal smoothing
        H_arr = np.stack(H_mats, axis=0)  # (T, 3, 3)
        if self.temporal_sigma > 0 and n_frames > 2:
            H_arr = gaussian_filter1d(H_arr, sigma=self.temporal_sigma, axis=0, mode="nearest")

        # Warp original (un-normalised) clean frames
        warped_frames, flows, stmaps = [], [], []
        for idx in range(n_frames):
            H_i = H_arr[idx]
            warped = cv2.warpPerspective(cleans[idx], H_i, (orig_w, orig_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            warped_frames.append(warped)

            flow = self._homography_to_flow(H_i, orig_h, orig_w)
            flows.append(flow)
            stmaps.append(flow_to_stmap(flow, orig_h, orig_w))

        n_good = sum(1 for d in debug_per_frame if d["model_used"] == "homography")
        logger.info("SIFT homography: %d/%d frames valid", n_good, n_frames)

        return {"warped_frames": warped_frames, "flows": flows, "stmaps": stmaps,
                "debug_per_frame": debug_per_frame, "method_name": "sift_homography"}

    @staticmethod
    def _homography_to_flow(H_mat, H, W):
        y, x = np.mgrid[0:H, 0:W]
        ones = np.ones((H, W), dtype=np.float64)
        pts = np.stack([x, y, ones], axis=0).reshape(3, -1)
        H_inv = np.linalg.inv(H_mat)
        src = H_inv @ pts
        src_xy = src[:2] / src[2]
        flow_x = (src_xy[0] - pts[0]).reshape(H, W)
        flow_y = (src_xy[1] - pts[1]).reshape(H, W)
        return np.stack([flow_x, flow_y], axis=-1).astype(np.float32)

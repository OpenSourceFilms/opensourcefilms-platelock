import logging
import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None

try:
    from utils.flow_utils import flow_to_stmap
except ImportError:
    flow_to_stmap = None

_SIFT_MAX_W = 2048
_RANSAC_METHOD = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC

logger = logging.getLogger(__name__)


class MaskedSIFTMethod:
    """Background-only SIFT homography registration for video clean plates."""

    def __init__(
        self,
        diff_threshold: int = 55,
        mask_dilate_px: int = 40,
        min_inliers: int = 15,
        sift_n_features: int = 8000,
        ransac_thresh: float = 4.0,
        temporal_sigma: float = 1.5,
        use_gradient: bool = True,
    ):
        self.diff_threshold = diff_threshold
        self.mask_dilate_px = mask_dilate_px
        self.min_inliers = min_inliers
        self.sift_n_features = sift_n_features
        self.ransac_thresh = ransac_thresh
        self.temporal_sigma = temporal_sigma
        self.use_gradient = use_gradient

    @staticmethod
    def _grad_mag_clahe(gray_u8: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray_u8.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_u8.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(mag_u8)

    def process_sequence(self, originals, cleans, show_progress=True):
        """
        Register a list of clean plates onto original camera frames.

        Returns dict: {warped_frames, flows, stmaps, debug_per_frame, method_name}
        """
        num_frames = len(originals)
        if num_frames != len(cleans):
            raise ValueError("originals and cleans must have the same length")
        if num_frames == 0:
            return {"warped_frames": [], "flows": [], "stmaps": [],
                    "debug_per_frame": [], "method_name": "MaskedSIFT"}

        H, W = originals[0].shape[:2]
        warped_frames, flows, stmaps, detection_debugs, H_list = [], [], [], [], []

        sift = cv2.SIFT_create(nfeatures=self.sift_n_features)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.mask_dilate_px, self.mask_dilate_px)
        )

        for t in range(num_frames):
            orig = originals[t]
            clean = cleans[t]

            # Actor mask: high-diff regions are actor or AI-added content
            diff = cv2.absdiff(orig, clean).max(axis=2)
            actor_mask_u8 = (diff > self.diff_threshold).astype(np.uint8) * 255
            actor_mask_u8 = cv2.dilate(actor_mask_u8, kernel)
            bg_mask_u8 = cv2.bitwise_not(actor_mask_u8)

            actor_mask_pct = 100.0 * (bg_mask_u8 == 0).sum() / (H * W)

            # Color-invariant feature images
            gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            gray_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
            if self.use_gradient:
                feat_orig = self._grad_mag_clahe(gray_orig)
                feat_clean = self._grad_mag_clahe(gray_clean)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                feat_orig = clahe.apply(gray_orig)
                feat_clean = clahe.apply(gray_clean)

            # Downscale for SIFT
            max_dim = max(H, W)
            scale = _SIFT_MAX_W / max_dim if max_dim > _SIFT_MAX_W else 1.0
            small_w, small_h = int(W * scale), int(H * scale)

            feat_orig_s = cv2.resize(feat_orig, (small_w, small_h), interpolation=cv2.INTER_AREA)
            feat_clean_s = cv2.resize(feat_clean, (small_w, small_h), interpolation=cv2.INTER_AREA)
            bg_mask_s = cv2.resize(bg_mask_u8, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            _, bg_mask_s = cv2.threshold(bg_mask_s, 127, 255, cv2.THRESH_BINARY)

            kp_o, des_o = sift.detectAndCompute(feat_orig_s, bg_mask_s)
            kp_c, des_c = sift.detectAndCompute(feat_clean_s, bg_mask_s)

            # Scale keypoints back to full resolution
            kp_o_pts = np.float32([kp.pt for kp in kp_o]) / scale if kp_o else np.zeros((0, 2), np.float32)
            kp_c_pts = np.float32([kp.pt for kp in kp_c]) / scale if kp_c else np.zeros((0, 2), np.float32)

            # Lowe ratio matching (des_clean queries des_orig)
            if des_o is not None and des_c is not None and len(des_o) >= 2 and len(des_c) >= 2:
                raw_matches = bf.knnMatch(des_c, des_o, k=2)
                good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
            else:
                good = []

            bg_sift_matches = len(good)
            H_t = np.eye(3, dtype=np.float64)
            inlier_count = 0

            if bg_sift_matches >= 4:
                pts_clean = np.float32([kp_c_pts[m.queryIdx] for m in good])
                pts_orig = np.float32([kp_o_pts[m.trainIdx] for m in good])

                H_est, _ = cv2.findHomography(
                    pts_clean, pts_orig, _RANSAC_METHOD, self.ransac_thresh,
                    maxIters=10000, confidence=0.999,
                )
                if H_est is not None:
                    proj = cv2.perspectiveTransform(
                        pts_clean.reshape(-1, 1, 2), H_est
                    ).reshape(-1, 2)
                    inlier_count = int(np.sum(np.linalg.norm(proj - pts_orig, axis=1) < self.ransac_thresh))
                    if inlier_count >= self.min_inliers:
                        H_t = H_est
                    else:
                        logger.warning("Frame %d: %d inliers (need %d), using identity", t, inlier_count, self.min_inliers)
                else:
                    logger.warning("Frame %d: findHomography failed, using identity", t)
            else:
                logger.warning("Frame %d: %d good matches (need >=4), using identity", t, bg_sift_matches)

            detection_debugs.append({
                "actor_mask_pct": actor_mask_pct,
                "bg_sift_matches": bg_sift_matches,
                "inliers": inlier_count,
                "model_used": "homography" if inlier_count >= self.min_inliers else "identity",
            })
            H_list.append(H_t)

        # Temporal smoothing
        if self.temporal_sigma > 0 and gaussian_filter1d is not None:
            H_arr = np.stack(H_list, axis=0)
            H_arr = gaussian_filter1d(H_arr, sigma=self.temporal_sigma, axis=0)
            H_list = [H_arr[i] for i in range(num_frames)]

        # Warp, flow, stmap
        debug_per_frame = []
        for t in range(num_frames):
            H_t = H_list[t]
            warped = cv2.warpPerspective(
                cleans[t], H_t, (W, H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            warped_frames.append(warped)

            try:
                H_inv = np.linalg.inv(H_t)
            except np.linalg.LinAlgError:
                H_inv = np.eye(3)
            grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
            grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
            mapped = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H_inv).reshape(H, W, 2)
            flow = (mapped - grid).astype(np.float32)
            flows.append(flow)
            stmaps.append(flow_to_stmap(flow, H, W) if flow_to_stmap is not None else flow)

            corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
            corner_disp = float(np.mean(np.linalg.norm(
                cv2.perspectiveTransform(corners, H_t).reshape(-1, 2) - corners.reshape(-1, 2),
                axis=1,
            )))
            d = detection_debugs[t].copy()
            d["corner_displacement_px"] = corner_disp
            d["H"] = H_t.tolist()
            debug_per_frame.append(d)

        return {
            "warped_frames": warped_frames,
            "flows": flows,
            "stmaps": stmaps,
            "debug_per_frame": debug_per_frame,
            "method_name": "MaskedSIFT",
        }

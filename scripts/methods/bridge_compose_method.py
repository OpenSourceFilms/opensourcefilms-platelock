#!/usr/bin/env python3
"""
BridgeComposeMethod: VFX plate-lock via three‑step homography composition.
"""

import sys
import logging
from pathlib import Path

try:
    from utils.flow_utils import flow_to_stmap
except ImportError:
    flow_to_stmap = None

import cv2
import numpy as np

# Optional heavy dependencies – imported lazily or with graceful fallback
try:
    import torch
except ImportError:
    torch = None
try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None

# Set up package path so we can import local utils
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:
    from utils.colour_normalise import colour_normalise as _colour_normalise
except ImportError:
    colour_normalise = None
    logging.warning("utils.colour_normalise not found, colour normalisation disabled")

logger = logging.getLogger(__name__)

_SIFT_MAX_W  = 1920
_LOFTR_MAX   = 840
_LG_MAX_W    = 1600
_RANSAC_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def _downscale(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    s = max_w / w
    return cv2.resize(img, (max_w, int(h * s)), interpolation=cv2.INTER_AREA), s


def _root_sift(des):
    return np.sqrt(des / (des.sum(axis=1, keepdims=True) + 1e-8))


class BridgeComposeMethod:
    """
    Registers AI‑inpainted clean plates onto original camera footage.
    """

    def __init__(
        self,
        ref_frame=0,
        min_inliers_bridge=15,
        min_inliers_tracking=30,
        ransac_thresh=4.0,
        temporal_sigma=0.0,
        use_loftr=True,
        use_lightglue=True,
        loftr_outdoor=True,
        sift_n_features=8000,
    ):
        self.ref_frame = ref_frame
        self.min_inliers_bridge = min_inliers_bridge
        self.min_inliers_tracking = min_inliers_tracking
        self.ransac_thresh = ransac_thresh
        self.temporal_sigma = temporal_sigma
        self.use_loftr = use_loftr
        self.use_lightglue = use_lightglue
        self.loftr_outdoor = loftr_outdoor
        self.sift_n_features = sift_n_features

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def process_sequence(self, originals, cleans, show_progress=True):
        """
        Compose each clean frame onto the corresponding original frame.

        Args:
            originals : list of np.ndarray (H,W,3) uint8
            cleans    : list of np.ndarray (H,W,3) uint8
            show_progress : bool

        Returns:
            dict with keys:
                warped_frames   : list of np.ndarray (H,W,3) uint8
                flows           : list of np.ndarray (H,W,2) float32 (backward flow)
                stmaps          : list of np.ndarray (H,W,2) float32 (backward coordinates)
                debug_per_frame : list of dict
                method_name     : str
                bridge_debug    : dict
        """
        T = len(originals)
        ref_idx = min(self.ref_frame, T - 1) if T > 0 else 0
        orig_ref = originals[ref_idx]
        clean_ref = cleans[ref_idx]

        # 1. Cross‑domain bridge at reference frame
        H_bridge, bridge_debug = self._find_bridge(orig_ref, clean_ref)
        if H_bridge is None:
            logger.warning("Bridge homography failed; using identity.")
            H_bridge = np.eye(3, dtype=np.float64)
            bridge_debug = {"H_bridge": H_bridge, "inliers": 0}

        # 2. Within‑sequence tracking
        H_orig_t_to_ref = self._track_within(originals, ref_idx)  # list of 3x3
        H_clean_t_to_ref = self._track_within(cleans, ref_idx)

        # 3. Compose per frame
        H_composites = []
        for t in range(T):
            H_o2r = H_orig_t_to_ref[t]
            H_c2r = H_clean_t_to_ref[t]
            H_comp = np.linalg.inv(H_o2r) @ H_bridge @ H_c2r
            H_composites.append(H_comp)

        # Temporal smoothing if requested
        if self.temporal_sigma > 0 and gaussian_filter1d is not None:
            H_composites = self._smooth_homographies(H_composites, self.temporal_sigma)

        H, W = originals[0].shape[:2]
        warped_frames = []
        flows = []
        stmaps = []
        debug_per_frame = []

        for t in range(T):
            H_comp = H_composites[t]

            # Warp clean frame onto original coordinate system
            warped = cv2.warpPerspective(
                cleans[t],
                H_comp,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_frames.append(warped)

            # Backward flow from original frame to clean frame
            flow = self._homography_to_flow(H_comp, H, W)
            flows.append(flow)

            # STMap (absolute backward coordinates)
            if flow_to_stmap is not None:
                stmaps.append(flow_to_stmap(flow, H, W))
            else:
                stmaps.append(flow)

            is_identity = np.allclose(H_comp, np.eye(3), atol=1e-6)
            fh2, fw2 = originals[0].shape[:2]
            corners = np.float32([[0,0],[fw2,0],[fw2,fh2],[0,fh2]]).reshape(-1,1,2)
            warped_corners = cv2.perspectiveTransform(corners, H_comp).reshape(-1,2)
            corner_disp = float(np.linalg.norm(
                warped_corners - corners.reshape(-1,2), axis=1).max())
            debug_per_frame.append({
                "model_used": "identity" if is_identity else "homography",
                "corner_displacement_px": round(corner_disp, 2),
                "H_composite": H_comp.tolist(),
            })

        return {
            "warped_frames": warped_frames,
            "flows": flows,
            "stmaps": stmaps,
            "debug_per_frame": debug_per_frame,
            "method_name": "BridgeCompose",
            "bridge_debug": bridge_debug,
        }

    # -----------------------------------------------------------------
    # Bridge matching
    # -----------------------------------------------------------------
    def _find_bridge(self, orig_ref, clean_ref):
        """
        Compute cross‑domain homography H_bridge : clean_ref -> orig_ref.
        """
        # Colour normalise clean to match original
        if _colour_normalise is not None:
            clean_norm = _colour_normalise(clean_ref, orig_ref)
        else:
            clean_norm = clean_ref.copy()

        bridge_debug = {}
        all_clean_pts = []
        all_orig_pts = []

        # ---- SIFT + RootSIFT ----
        scale_sift = min(1.0, 1920.0 / max(orig_ref.shape[1], 1.0))
        if scale_sift < 1.0:
            orig_s = cv2.resize(orig_ref, None, fx=scale_sift, fy=scale_sift)
            clean_s = cv2.resize(clean_norm, None, fx=scale_sift, fy=scale_sift)
        else:
            orig_s, clean_s = orig_ref, clean_norm

        gray_orig = cv2.cvtColor(orig_s, cv2.COLOR_BGR2GRAY)
        gray_clean = cv2.cvtColor(clean_s, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=self.sift_n_features)
        kp_orig, des_orig = sift.detectAndCompute(gray_orig, None)
        kp_clean, des_clean = sift.detectAndCompute(gray_clean, None)

        if des_orig is None or des_clean is None or len(kp_orig) < 4 or len(kp_clean) < 4:
            logger.debug("SIFT: insufficient keypoints")
        else:
            # RootSIFT
            eps = 1e-8
            des_orig = np.sqrt(des_orig / (des_orig.sum(axis=1, keepdims=True) + eps))
            des_clean = np.sqrt(des_clean / (des_clean.sum(axis=1, keepdims=True) + eps))

            bf = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = bf.knnMatch(des_clean, des_orig, k=2)
            good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

            if len(good) >= 4:
                pts_clean = np.float32([kp_clean[m.queryIdx].pt for m in good])
                pts_orig = np.float32([kp_orig[m.trainIdx].pt for m in good])
                # rescale to full resolution
                pts_clean /= scale_sift
                pts_orig /= scale_sift
                all_clean_pts.append(pts_clean)
                all_orig_pts.append(pts_orig)
                bridge_debug["sift_matches"] = len(good)

        # ---- LoFTR ----
        if self.use_loftr and torch is not None:
            try:
                from kornia.feature import LoFTR

                loftr_model = LoFTR(pretrained="outdoor" if self.loftr_outdoor else "indoor")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loftr_model = loftr_model.to(device).eval()

                max_loftr = 840
                scale_lo = min(1.0, max_loftr / max(orig_ref.shape[0], orig_ref.shape[1]))
                if scale_lo < 1.0:
                    o_lo = cv2.resize(orig_ref, None, fx=scale_lo, fy=scale_lo)
                    c_lo = cv2.resize(clean_norm, None, fx=scale_lo, fy=scale_lo)
                else:
                    o_lo, c_lo = orig_ref, clean_norm

                gray_o = cv2.cvtColor(o_lo, cv2.COLOR_BGR2GRAY)
                gray_c = cv2.cvtColor(c_lo, cv2.COLOR_BGR2GRAY)

                img0 = torch.from_numpy(gray_o[None, None, ...]).float() / 255.0
                img1 = torch.from_numpy(gray_c[None, None, ...]).float() / 255.0
                img0 = img0.to(device)
                img1 = img1.to(device)

                with torch.no_grad():
                    out = loftr_model({"image0": img0, "image1": img1})

                kp0 = out["keypoints0"].cpu().numpy()  # orig
                kp1 = out["keypoints1"].cpu().numpy()  # clean
                conf = out["confidence"].cpu().numpy().flatten()
                valid = conf > 0.3
                kp0 = kp0[valid] / scale_lo
                kp1 = kp1[valid] / scale_lo

                if len(kp0) >= 4:
                    all_clean_pts.append(kp1)
                    all_orig_pts.append(kp0)
                    bridge_debug["loftr_matches"] = len(kp0)
            except Exception as e:
                logger.warning("LoFTR matching failed: %s", e)

        # ---- LightGlue ----
        if self.use_lightglue and torch is not None:
            try:
                from lightglue import LightGlue, SuperPoint
                from lightglue.utils import rbd

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
                matcher = LightGlue(features="superpoint").eval().to(device)

                max_lg = 1600
                scale_lg = min(1.0, max_lg / max(orig_ref.shape[1], 1.0))
                if scale_lg < 1.0:
                    o_lg = cv2.resize(orig_ref, None, fx=scale_lg, fy=scale_lg)
                    c_lg = cv2.resize(clean_norm, None, fx=scale_lg, fy=scale_lg)
                else:
                    o_lg, c_lg = orig_ref, clean_norm

                gray_o_lg = cv2.cvtColor(o_lg, cv2.COLOR_BGR2GRAY)
                gray_c_lg = cv2.cvtColor(c_lg, cv2.COLOR_BGR2GRAY)

                img0_lg = torch.from_numpy(gray_o_lg[None, None, ...]).float() / 255.0
                img1_lg = torch.from_numpy(gray_c_lg[None, None, ...]).float() / 255.0
                img0_lg = img0_lg.to(device)
                img1_lg = img1_lg.to(device)

                with torch.no_grad():
                    feats0 = extractor.extract(img0_lg)
                    feats1 = extractor.extract(img1_lg)
                    matches01 = matcher({"image0": feats0, "image1": feats1})
                    feats0, feats1, matches01 = [
                        rbd(x) for x in [feats0, feats1, matches01]
                    ]

                kp0 = feats0["keypoints"].cpu().numpy()
                kp1 = feats1["keypoints"].cpu().numpy()
                matched = matches01["matches"].cpu().numpy()  # (M,2)
                pts0 = kp0[matched[:, 0]] / scale_lg
                pts1 = kp1[matched[:, 1]] / scale_lg

                if len(pts0) >= 4:
                    all_clean_pts.append(pts1)  # clean image is feats1
                    all_orig_pts.append(pts0)  # orig image is feats0
                    bridge_debug["lightglue_matches"] = len(pts0)
            except Exception as e:
                logger.warning("LightGlue matching failed: %s", e)

        # Pool all matches and find final homography
        if not all_clean_pts:
            logger.warning("No matches obtained for bridge.")
            return None, bridge_debug

        pooled_clean = np.concatenate(all_clean_pts, axis=0)
        pooled_orig = np.concatenate(all_orig_pts, axis=0)
        bridge_debug["pooled_matches"] = len(pooled_clean)

        # Use USAC_MAGSAC if available, else fallback to RANSAC
        try:
            method = cv2.USAC_MAGSAC
        except AttributeError:
            method = cv2.RANSAC
            logger.debug("USAC_MAGSAC not available, using RANSAC")

        H, mask = cv2.findHomography(
            pooled_clean,
            pooled_orig,
            method,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=10000,
            confidence=0.999,
        )
        if H is None:
            return None, bridge_debug

        # Manual inlier re‑check with tighter threshold
        proj = cv2.perspectiveTransform(
            pooled_clean.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)
        err = np.linalg.norm(proj - pooled_orig, axis=1)
        mask2 = err < 5.0
        inliers = np.sum(mask2)
        bridge_debug["inliers_after_recheck"] = int(inliers)

        if inliers >= self.min_inliers_bridge:
            # Optionally refine H with only those inliers
            H_final, _ = cv2.findHomography(
                pooled_clean[mask2],
                pooled_orig[mask2],
                method,
                ransacReprojThreshold=2.0,
                maxIters=10000,
                confidence=0.999,
            )
            if H_final is not None:
                H = H_final
            return H, bridge_debug
        else:
            logger.warning(
                "Bridge homography inliers (%d) below threshold (%d).",
                inliers,
                self.min_inliers_bridge,
            )
            # fallback to identity (but we still return the computed H if desired)
            # Returning None triggers identity later.
            return None, bridge_debug

    # -----------------------------------------------------------------
    # Within‑sequence tracking
    # -----------------------------------------------------------------
    def _track_within(self, frames, ref_idx):
        """
        For each frame compute homography to the reference frame.
        Returns list of 3x3 matrices mapping frame_t -> ref coords (full res).
        """
        T = len(frames)
        sift = cv2.SIFT_create(nfeatures=self.sift_n_features)
        bf   = cv2.BFMatcher(cv2.NORM_L2)
        eps  = 1e-8
        method = _RANSAC_METHOD

        # Cache reference descriptors at downscaled resolution
        ref_s, scale_r = _downscale(frames[ref_idx], _SIFT_MAX_W)
        ref_gray = cv2.cvtColor(ref_s, cv2.COLOR_BGR2GRAY)
        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
        if des_ref is None or len(kp_ref) < 4:
            logger.warning("Reference frame has too few keypoints; tracking will be identity.")
            return [np.eye(3, dtype=np.float64)] * T
        des_ref = np.sqrt(des_ref / (des_ref.sum(axis=1, keepdims=True) + eps))
        # Scale reference kp coords to full resolution
        pts_ref_full = np.float32([k.pt for k in kp_ref]) / scale_r

        H_list = []
        for t in range(T):
            if t == ref_idx:
                H_list.append(np.eye(3, dtype=np.float64))
                continue

            fr_s, scale_t = _downscale(frames[t], _SIFT_MAX_W)
            gray = cv2.cvtColor(fr_s, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            if des is None or len(kp) < 4:
                logger.debug("Frame %d: too few keypoints, using identity.", t)
                H_list.append(np.eye(3, dtype=np.float64))
                continue

            des = np.sqrt(des / (des.sum(axis=1, keepdims=True) + eps))
            raw = bf.knnMatch(des, des_ref, k=2)
            good = [m for m, n in raw if m.distance < 0.75 * n.distance]

            if len(good) < 4:
                logger.debug("Frame %d: too few matches, using identity.", t)
                H_list.append(np.eye(3, dtype=np.float64))
                continue

            # Scale kp coords to full resolution before homography
            pts_t   = np.float32([kp[m.queryIdx].pt for m in good]) / scale_t
            pts_ref = pts_ref_full[[m.trainIdx for m in good]]

            H, mask = cv2.findHomography(
                pts_t,
                pts_ref,
                method,
                ransacReprojThreshold=3.0,
                maxIters=10000,
                confidence=0.999,
            )
            if H is None:
                logger.debug("Frame %d: homography estimation failed, using identity.", t)
                H_list.append(np.eye(3, dtype=np.float64))
                continue

            # Manual re‑check
            proj = cv2.perspectiveTransform(
                pts_t.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            err = np.linalg.norm(proj - pts_ref, axis=1)
            inlr = err < 4.0

            if np.sum(inlr) >= self.min_inliers_tracking:
                # Optionally refine
                H, _ = cv2.findHomography(
                    pts_t[inlr],
                    pts_ref[inlr],
                    method,
                    ransacReprojThreshold=2.0,
                    maxIters=10000,
                    confidence=0.999,
                )
                H_list.append(H if H is not None else np.eye(3, dtype=np.float64))
            else:
                logger.debug("Frame %d: insufficient inliers, using identity.", t)
                H_list.append(np.eye(3, dtype=np.float64))

        return H_list

    # -----------------------------------------------------------------
    # Utility: flow from homography
    # -----------------------------------------------------------------
    def _homography_to_flow(self, H_mat, H, W):
        """
        Convert a homography (clean->original) to a backward flow (orig->clean).
        Flow shape (H,W,2) float32, displacement in pixel coordinates.
        """
        H_inv = np.linalg.inv(H_mat)
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)  # (H,W,2)
        grid_flat = grid.reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(grid_flat, H_inv).reshape(H, W, 2)
        flow = mapped - grid
        return flow.astype(np.float32)

    # -----------------------------------------------------------------
    # Temporal smoothing of homographies
    # -----------------------------------------------------------------
    @staticmethod
    def _smooth_homographies(H_list, sigma):
        """Apply 1‑D Gaussian smoothing over the 8‑parameter representation."""
        if gaussian_filter1d is None:
            logger.warning("scipy not available, skipping temporal smoothing")
            return H_list

        stacked = np.stack(H_list, axis=0)   # (T,3,3)
        smoothed = gaussian_filter1d(stacked, sigma=sigma, axis=0, mode="nearest")
        return [smoothed[t] for t in range(len(H_list))]


# -------------------------------
# Notes:
# - All imports that may fail (torch, kornia, lightglue) are wrapped in try/except
#   and only used when the corresponding flags are enabled.
# - Colour normalisation is applied to the clean reference before *all* feature extractions,
#   which yields more consistent cross‑domain matching.
# - LoFTR and LightGlue use fixed maximum dimensions to balance speed and quality;
#   keypoints are scaled back to full resolution before pooling.
# - Temporal smoothing, when requested, operates on the 8‑parameter normalised homographies.
# - Returns backward flow and STMap from original to clean space, suitable for remapping.
# - If bridge or within‑tracking homographies fail, identity matrices are used gracefully.
# - Method ‘USAC_MAGSAC’ is preferred when available (OpenCV ≥ 4.4), otherwise RANSAC.
# - The ‘debug_per_frame’ dictionary stores the raw homographies for inspection.

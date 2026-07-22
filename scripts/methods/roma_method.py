"""RoMa v2 direct orig↔clean correspondence matcher for plate lock.

Per-frame approach: match each (orig_i, clean_i) pair using learned dense
features, sample high-certainty background correspondences, feed into the
shared robust_fit pipeline.

DO NOT trust the raw RoMa dense warp everywhere — hallucinated regions produce
spurious matches. Only high-certainty background correspondences drive the warp.

To enable:
  pip install romatch
  # or: git clone https://github.com/Parskatt/RoMa repos/roma
  #     pip install -e repos/roma

RoMa v2 downloads its DINOv3 backbone from HuggingFace on first use.
"""
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from utils.flow_utils import compute_bg_mask, warp_image_with_flow, flow_to_stmap
from utils.robust_fit import fit_robust, smooth_flows_temporal

logger = logging.getLogger(__name__)


class RoMANotAvailable(RuntimeError):
    pass


class RoMAMethod:
    """Per-frame RoMa v2 dense matcher -> robust constrained warp."""

    INSTALL_INSTRUCTIONS = "pip install romatch"

    def __init__(self, model_variant="outdoor", certainty_threshold=0.02,
                 max_matches=4000, device=None, temporal_smooth_sigma=1.5):
        self.model_variant = model_variant
        self.certainty_threshold = certainty_threshold
        self.max_matches = max_matches
        self.temporal_smooth_sigma = temporal_smooth_sigma
        self.device = device
        self._model = None

    def load_model(self):
        if self._model is not None:
            return
        try:
            from romatch import roma_outdoor, roma_indoor
        except ImportError:
            raise RoMANotAvailable(
                "romatch not installed.\n" + self.INSTALL_INSTRUCTIONS)

        import torch
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading RoMa (%s) on %s...", self.model_variant, self.device)
        if self.model_variant == "indoor":
            self._model = roma_indoor(device=self.device, use_custom_corr=False)
        else:
            self._model = roma_outdoor(device=self.device, use_custom_corr=False)
        logger.info("RoMa loaded.")

    def match_pair(self, orig, clean):
        if self._model is None:
            self.load_model()
        from PIL import Image

        H, W = orig.shape[:2]
        orig_pil  = Image.fromarray(orig)
        clean_pil = Image.fromarray(clean)

        warp_field, certainty = self._model.match(orig_pil, clean_pil, device=self.device)
        if hasattr(warp_field, "cpu"):
            warp_field = warp_field.cpu().numpy()
        if hasattr(certainty, "cpu"):
            certainty = certainty.cpu().numpy()

        cert_flat = certainty.ravel()
        wf_flat   = warp_field.reshape(-1, 4)

        good = cert_flat > self.certainty_threshold
        if not good.any():
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)

        idxs = np.where(good)[0]
        if len(idxs) > self.max_matches:
            idxs = idxs[np.argsort(cert_flat[idxs])[-self.max_matches:]]

        wf_sel   = wf_flat[idxs]
        cert_sel = cert_flat[idxs]

        def denorm(xy_norm):
            x = (xy_norm[:, 0] + 1) / 2 * W - 0.5
            y = (xy_norm[:, 1] + 1) / 2 * H - 0.5
            return np.stack([x, y], axis=1).astype(np.float32)

        dst_pts = denorm(wf_sel[:, :2])
        src_pts = denorm(wf_sel[:, 2:])
        return src_pts, dst_pts, cert_sel.astype(np.float32)

    def process_sequence(self, originals, cleans, show_progress=True):
        if self._model is None:
            self.load_model()

        H, W = originals[0].shape[:2]
        flows, warped_frames, debug_list = [], [], []

        it = tqdm(range(len(originals)), desc="RoMa v2") if show_progress else range(len(originals))
        for i in it:
            orig, clean = originals[i], cleans[i]
            bg_mask = compute_bg_mask(orig, clean, threshold=25, dilation_px=20)

            try:
                src_pts, dst_pts, _ = self.match_pair(orig, clean)
            except Exception as e:
                logger.error("RoMa match_pair failed frame %d: %s", i, e)
                flows.append(np.zeros((H, W, 2), np.float32))
                warped_frames.append(clean.copy())
                debug_list.append({"model_used": "identity", "roma_error": str(e)})
                continue

            if len(dst_pts) > 0:
                dy = np.clip(dst_pts[:, 1].astype(int), 0, H - 1)
                dx = np.clip(dst_pts[:, 0].astype(int), 0, W - 1)
                bg_ok = bg_mask[dy, dx] > 127
                src_pts = src_pts[bg_ok]
                dst_pts = dst_pts[bg_ok]

            if len(src_pts) < 6:
                logger.warning("Frame %d: too few bg matches (%d); identity.", i, len(src_pts))
                flows.append(np.zeros((H, W, 2), np.float32))
                warped_frames.append(clean.copy())
                debug_list.append({"model_used": "identity", "n_bg_matches": len(src_pts)})
                continue

            flow, debug = fit_robust(src_pts, dst_pts, H, W)
            debug["n_bg_matches"] = len(src_pts)
            flows.append(flow)
            warped_frames.append(warp_image_with_flow(clean, flow))
            debug_list.append(debug)

        if self.temporal_smooth_sigma > 0 and len(flows) >= 3:
            flows = smooth_flows_temporal(flows, sigma=self.temporal_smooth_sigma)
            warped_frames = [warp_image_with_flow(cleans[i], flows[i])
                             for i in range(len(cleans))]

        stmaps = [flow_to_stmap(f, H, W) for f in flows]

        return {
            "warped_frames": warped_frames,
            "flows": flows,
            "stmaps": stmaps,
            "debug_per_frame": debug_list,
            "method_name": "roma_v2",
        }

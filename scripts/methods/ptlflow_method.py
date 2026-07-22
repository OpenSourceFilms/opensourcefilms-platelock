import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PTLFLOW_MAX_W = 1280  # cap inference width; correlation is O((H*W)^2) — 5K OOMs

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from utils.flow_utils import warp_image_with_flow, compute_bg_mask, inpaint_flow_masked

try:
    import ptlflow
except ImportError as e:
    raise ImportError("ptlflow required. Install: pip install ptlflow") from e

logger = logging.getLogger(__name__)


class PTLFlowMethod:
    def __init__(self, model_name: str = "sea_raft_l", iters: int = 20,
                 mixed_precision: bool = True, device: str | None = None):
        self.model_name = model_name
        self.iters = iters
        self.mixed_precision = mixed_precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self._model = None

    def load_model(self):
        if self._model is not None:
            return
        logger.info("Loading PTLFlow '%s' (iters=%d) ...", self.model_name, self.iters)
        # PTLFlow 0.4.x: get_model(name) with no args downloads pretrained weights
        # Passing a Namespace causes 'no attribute model' if args schema doesn't match
        model = ptlflow.get_model(self.model_name)
        model.eval().to(self.device)
        self._model = model
        logger.info("Model on %s", self.device)

    def _ensure(self):
        if self._model is None:
            self.load_model()

    def estimate_flow_pair(self, original: np.ndarray, clean: np.ndarray) -> np.ndarray:
        """Returns (H,W,2) flow from original→clean."""
        self._ensure()
        H, W = original.shape[:2]

        # Downsample before inference if needed — correlation is O((H*W)^2) and OOMs at 5K
        scale = min(1.0, _PTLFLOW_MAX_W / W)
        if scale < 1.0:
            nw, nh = int(W * scale), int(H * scale)
            orig_in = cv2.resize(original, (nw, nh), interpolation=cv2.INTER_AREA)
            clean_in = cv2.resize(clean, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            orig_in, clean_in = original, clean

        def to_t(img):
            # PTLFlow/SEA-RAFT does bgr_to_rgb=True internally — expects BGR input.
            # Our frames are RGB (loaded via cv2 + cvtColor), so reverse channels here.
            bgr = img[:, :, ::-1].copy()
            return torch.from_numpy(bgr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        pair = torch.stack([to_t(orig_in), to_t(clean_in)], dim=1)  # (1,2,3,H,W)
        with torch.no_grad():
            if self.mixed_precision and self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = self._model({"images": pair})
            else:
                out = self._model({"images": pair})

        flows = out["flows"]
        # PTLFlow 0.4.x: flows shape is (B, T, 2, H, W) — take [0,0] then permute to (H,W,2)
        ft = flows[-1] if isinstance(flows, (list, tuple)) else flows
        if ft.dim() == 5:
            flow = ft[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        else:
            flow = ft[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

        if flow.shape[:2] != (H, W):
            t = F.interpolate(torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0),
                              size=(H, W), mode="bilinear", align_corners=False)
            flow = t[0].permute(1, 2, 0).numpy().astype(np.float32)
            # Scale flow magnitudes to match original resolution
            if scale < 1.0:
                flow = flow / scale
        return flow

    def warp_frame(self, clean: np.ndarray, flow: np.ndarray) -> np.ndarray:
        return warp_image_with_flow(clean, flow)

    def process_sequence(self, originals: list, cleans: list, ecc_prepass=None,
                          flow_cfg: dict | None = None, show_progress: bool = True) -> dict:
        self._ensure()
        cfg = flow_cfg or {}
        spatial_blur = cfg.get("spatial_blur_px", 0.0)
        temporal_r = cfg.get("temporal_smooth_radius", 0)
        clamp_mult = cfg.get("clamp_p95_multiplier", 0.0)

        flows, warped_frames = [], []
        it = tqdm(range(len(originals)), desc=f"PTLFlow {self.model_name}") if show_progress else range(len(originals))
        for i in it:
            orig, cln = originals[i], cleans[i]
            current = cln
            if ecc_prepass is not None:
                try:
                    _, _, _ = ecc_prepass.align_frame(orig, cln)  # unused here — use full result
                    ecc_warped, _, _ = ecc_prepass.align_frame(orig, cln)
                    current = ecc_warped
                except Exception as e:
                    logger.warning("ECC pre-pass failed frame %d: %s", i, e)

            flow = self.estimate_flow_pair(orig, current)

            # Inpaint flow in AI-changed regions before any other processing.
            # Dense optical flow produces garbage where content differs between plates;
            # propagating reliable background flow into those regions fixes the warp.
            bg_mask = compute_bg_mask(orig, current, threshold=30, dilation_px=20)
            flow = inpaint_flow_masked(flow, bg_mask)

            if spatial_blur > 0:
                from scipy.ndimage import gaussian_filter
                flow = gaussian_filter(flow, (spatial_blur, spatial_blur, 0)).astype(np.float32)
            if clamp_mult > 0:
                mag = np.linalg.norm(flow, axis=-1)
                p95 = np.percentile(mag, 95)
                factor = np.minimum(p95 * clamp_mult / (mag + 1e-6), 1.0)[..., None]
                flow = (flow * factor).astype(np.float32)

            flows.append(flow)
            warped_frames.append(self.warp_frame(current, flow))

        if temporal_r > 0:
            from utils.flow_utils import smooth_flow_temporal
            flows = smooth_flow_temporal(flows, temporal_r)
            warped_frames = [self.warp_frame(cleans[i], flows[i]) for i in range(len(cleans))]

        return {"warped_frames": warped_frames, "flows": flows,
                "model_name": self.model_name, "method_name": f"ptlflow_{self.model_name}"}

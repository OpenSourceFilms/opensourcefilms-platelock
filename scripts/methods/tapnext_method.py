"""TAPNext++ / BootsTAPNext background grid tracker → robust mesh warp.

Drop-in alternative to CoTrackerMeshMethod. Uses the same downstream
robust_fit + flow_to_stmap pipeline so outputs are directly comparable.

To enable:
  git clone https://github.com/google-deepmind/tapnet repos/tapnet
  pip install -e repos/tapnet[torch]
  mkdir -p models/tapnet
  # Download bootstapnext_torch.pth into models/tapnet/

Reference: BootsTAPNext / TAPNext++ — DeepMind TAPNet (2024/2025)
  https://github.com/google-deepmind/tapnet
"""
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from utils.flow_utils import compute_bg_mask, warp_image_with_flow, flow_to_stmap
from utils.robust_fit import fit_robust

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tapnet"


class TAPNextNotAvailable(RuntimeError):
    pass


class TAPNextMethod:
    """Same interface as CoTrackerMeshMethod — swap in for direct comparison."""

    INSTALL_INSTRUCTIONS = (
        "git clone https://github.com/google-deepmind/tapnet repos/tapnet\n"
        "pip install -e repos/tapnet[torch]\n"
        "mkdir -p models/tapnet\n"
        "# Download bootstapnext_torch.pth into models/tapnet/"
    )

    def __init__(self, grid_size=30, min_vis_ratio=0.3, smooth_trajectories=True,
                 device=None, tracking_wh=(960, 540), checkpoint_path=None):
        self.grid_size = grid_size
        self.min_vis_ratio = min_vis_ratio
        self.smooth_trajectories = smooth_trajectories
        self.tracking_wh = tracking_wh
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self._model = None

    def load_model(self):
        if self._model is not None:
            return
        try:
            from tapnet.torch import tapir_model
        except ImportError:
            raise TAPNextNotAvailable(
                "tapnet not installed.\n" + self.INSTALL_INSTRUCTIONS)

        import torch
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = self.checkpoint_path
        if ckpt is None or not ckpt.is_file():
            candidates = (list(_CHECKPOINT_DIR.glob("*.pth")) +
                          list(_CHECKPOINT_DIR.glob("*.pt")))
            if not candidates:
                raise TAPNextNotAvailable(
                    f"No checkpoint found in {_CHECKPOINT_DIR}.\n" +
                    self.INSTALL_INSTRUCTIONS)
            ckpt = candidates[0]

        logger.info("Loading BootsTAPNext from %s on %s", ckpt, self.device)
        # TAPIR is the shared backbone; BootsTAPNext uses use_causal_conv=False
        self._model = tapir_model.TAPIR(use_casual_conv=False,
                                         bilinear_interp_with_depthwise_conv=False)
        state = torch.load(str(ckpt), map_location=self.device)
        params = state.get("params", state.get("model_state_dict", state))
        self._model.load_state_dict(params, strict=False)
        self._model.to(self.device).eval()
        logger.info("BootsTAPNext loaded on %s", self.device)

    def track_sequence(self, frames, query_pts=None):
        """
        Track query_pts through frames. Returns (tracks T×N×2, vis T×N) in (x,y).

        TAPNet returns (y,x) convention; converted here to (x,y) to match CoTracker.
        Visibility: occlusion < 0.5 means the point is visible.
        """
        if self._model is None:
            self.load_model()
        import torch

        T = len(frames)
        H, W = frames[0].shape[:2]
        device = next(self._model.parameters()).device

        # (1, T, H, W, 3) float32 in [0,1] — TAPNet convention
        video = torch.stack(
            [torch.from_numpy(f.astype(np.float32) / 255.0) for f in frames], 0
        ).unsqueeze(0).to(device)

        if query_pts is not None:
            q = np.zeros((1, len(query_pts), 3), np.float32)
            q[0, :, 0] = 0                 # start frame index
            q[0, :, 1] = query_pts[:, 1]  # y
            q[0, :, 2] = query_pts[:, 0]  # x
        else:
            step = max(1, W // self.grid_size)
            ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
            pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
            q = np.zeros((1, len(pts), 3), np.float32)
            q[0, :, 0] = 0
            q[0, :, 1] = pts[:, 1]
            q[0, :, 2] = pts[:, 0]
        q_t = torch.from_numpy(q).to(device)

        with torch.no_grad():
            out = self._model(video, q_t)

        # out["tracks"]: (1, T, N, 2) in (y, x); out["occlusion"]: (1, T, N)
        tracks = out["tracks"][0].cpu().numpy()      # (T, N, 2)
        occlusion = out["occlusion"][0].cpu().numpy()  # (T, N)

        # Convert (y,x) → (x,y) to match CoTracker convention
        tracks = tracks[..., ::-1].copy().astype(np.float32)
        visible = (occlusion < 0.5)

        return tracks, visible

    def process_sequence(self, originals, cleans, show_progress=True):
        if self._model is None:
            self.load_model()

        H, W = originals[0].shape[:2]
        tw, th = self.tracking_wh
        do_downscale = (tw != W or th != H)
        scale_x, scale_y = W / tw, H / th

        def downscale(frames):
            return [cv2.resize(f, (tw, th), cv2.INTER_AREA) for f in frames]

        track_cleans = downscale(cleans)    if do_downscale else cleans
        track_origs  = downscale(originals) if do_downscale else originals

        bg_mask = compute_bg_mask(track_origs[0], track_cleans[0], threshold=30, dilation_px=10)
        step = max(1, tw // self.grid_size)
        ys, xs = np.mgrid[step // 2:th:step, step // 2:tw:step]
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        in_bg = bg_mask[pts[:, 1].astype(int).clip(0, th - 1),
                        pts[:, 0].astype(int).clip(0, tw - 1)] > 127
        query_pts = pts[in_bg] if in_bg.sum() >= 9 else pts
        logger.info("TAPNext: %d background query pts at %dx%d", int(in_bg.sum()), tw, th)

        logger.info("TAPNext: tracking clean (%d frames)...", len(cleans))
        tc, vc = self.track_sequence(track_cleans, query_pts=query_pts)
        logger.info("TAPNext: tracking original (%d frames)...", len(originals))
        to_, vo = self.track_sequence(track_origs, query_pts=query_pts)

        if do_downscale:
            tc  = tc  * np.array([scale_x, scale_y], np.float32)
            to_ = to_ * np.array([scale_x, scale_y], np.float32)

        if self.smooth_trajectories:
            tc  = gaussian_filter1d(tc,  sigma=1.0, axis=0).astype(np.float32)
            to_ = gaussian_filter1d(to_, sigma=1.0, axis=0).astype(np.float32)

        keep = (vc.mean(0) >= self.min_vis_ratio) & (vo.mean(0) >= self.min_vis_ratio)
        tc, to_ = tc[:, keep, :], to_[:, keep, :]
        vc, vo  = vc[:, keep], vo[:, keep]

        flows, warped_frames, debug_list = [], [], []
        it = tqdm(range(len(cleans)), desc="TAPNext mesh") if show_progress else range(len(cleans))
        for t in it:
            fvis = vc[t] & vo[t]
            if fvis.sum() < 3:
                flow = np.zeros((H, W, 2), np.float32)
                warped_frames.append(cleans[t].copy())
                debug_list.append({"model_used": "identity", "n_inliers": 0})
            else:
                flow, debug = fit_robust(tc[t][fvis], to_[t][fvis], H, W)
                warped_frames.append(warp_image_with_flow(cleans[t], flow))
                debug_list.append(debug)
            flows.append(flow)

        stmaps = [flow_to_stmap(f, H, W) for f in flows]

        return {
            "warped_frames": warped_frames,
            "flows": flows,
            "stmaps": stmaps,
            "tracks_clean": tc,
            "tracks_orig": to_,
            "debug_per_frame": debug_list,
            "method_name": "tapnext_mesh",
        }

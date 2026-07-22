import sys
import logging
import numpy as np
import cv2
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from utils.flow_utils import compute_bg_mask, warp_image_with_flow, flow_to_stmap
from utils.robust_fit import fit_robust

logger = logging.getLogger(__name__)


class CoTrackerMeshMethod:
    """CoTracker3 point tracking → Delaunay mesh warp.

    Tracking runs at tracking_wh resolution to fit VRAM (default 960x540).
    Tracks are scaled back to full resolution before the mesh warp step.

    Sequences are processed in short overlapping chunks. At the start of every
    chunk, fresh background query points are sampled from original[chunk_start]
    and a masked ECC alignment maps them into clean-space so both tracking calls
    start from the same physical features — not the same pixel coordinates.
    Chunk flows are blended with a smoothstep in the overlap region to avoid
    hard discontinuities at boundaries.
    """

    def __init__(self, grid_size: int = 30, min_vis_ratio: float = 0.3,
                 smooth_trajectories: bool = True, device=None,
                 tracking_wh: tuple = (960, 540),
                 chunk_size: int = 40, overlap: int = 8,
                 per_chunk_seed_ecc: bool = True,
                 bg_mask_threshold: int = 30):
        self.grid_size = grid_size
        self.min_vis_ratio = min_vis_ratio
        self.smooth_trajectories = smooth_trajectories
        self.tracking_wh = tracking_wh          # (W, H) at which CoTracker runs
        self.chunk_size = chunk_size            # frames per tracking chunk
        self.overlap = overlap                  # frames of blend overlap between chunks
        self.per_chunk_seed_ecc = per_chunk_seed_ecc
        self.bg_mask_threshold = bg_mask_threshold
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        if self.model is not None:
            return
        logger.info("Loading CoTracker3 from torch hub...")
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.model.to(self.device).eval()
        logger.info("CoTracker3 loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Raw tracking
    # ------------------------------------------------------------------

    def _track_chunk(self, frames: list, query_pts: np.ndarray | None) -> tuple:
        """Track query_pts through frames. Returns (T,N,2) tracks and (T,N) visibility."""
        video = torch.stack([torch.from_numpy(f).float() for f in frames], 0)
        video = video.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)  # 1,T,C,H,W
        with torch.no_grad():
            if query_pts is not None:
                q = np.zeros((1, len(query_pts), 3), dtype=np.float32)
                q[0, :, 1:] = query_pts          # time=0, x, y
                q_t = torch.from_numpy(q).to(self.device)
                tracks_t, vis_t = self.model(video, queries=q_t)
            else:
                tracks_t, vis_t = self.model(video, grid_size=self.grid_size)
        tracks = tracks_t[0].cpu().numpy().astype(np.float32)  # T,N,2
        vis    = vis_t[0].cpu().numpy().astype(bool)            # T,N
        del video, tracks_t, vis_t
        torch.cuda.empty_cache()
        return tracks, vis

    # kept for backwards compatibility — not called by process_sequence
    def track_sequence(self, frames: list, query_pts: np.ndarray | None = None,
                       chunk_size: int = 40) -> tuple:
        """Single-sequence tracking with internal carry-forward chunking."""
        T = len(frames)
        if T <= chunk_size:
            return self._track_chunk(frames, query_pts)
        overlap_inner = chunk_size // 4
        all_tracks, all_vis = [], []
        current_query = query_pts
        i = 0
        while i < T:
            end = min(i + chunk_size, T)
            tr, vi = self._track_chunk(frames[i:end], current_query)
            torch.cuda.empty_cache()
            if i == 0:
                all_tracks.append(tr)
                all_vis.append(vi)
            else:
                all_tracks.append(tr[overlap_inner:])
                all_vis.append(vi[overlap_inner:])
            carry_frame = min(chunk_size - overlap_inner, len(tr)) - 1
            current_query = tr[carry_frame]
            i += chunk_size - overlap_inner
        return np.concatenate(all_tracks, axis=0)[:T], np.concatenate(all_vis, axis=0)[:T]

    # ------------------------------------------------------------------
    # Per-frame warp
    # ------------------------------------------------------------------

    def warp_frame(self, clean: np.ndarray, src_pts: np.ndarray,
                   dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """RANSAC + piecewise-affine warp via fit_robust. Returns (warped, flow, debug)."""
        H, W = clean.shape[:2]
        ib = lambda pts: ((pts[:, 0] >= 0) & (pts[:, 0] < W) &
                          (pts[:, 1] >= 0) & (pts[:, 1] < H))
        valid = ib(src_pts) & ib(dst_pts)
        if valid.sum() < 3:
            logger.warning("Frame has <3 valid correspondences; returning identity flow.")
            return clean.copy(), np.zeros((H, W, 2), np.float32), {"model_used": "identity", "n_inliers": 0}
        flow, debug = fit_robust(src_pts[valid], dst_pts[valid], H, W)
        return warp_image_with_flow(clean, flow), flow, debug

    # ------------------------------------------------------------------
    # Frame downscaling
    # ------------------------------------------------------------------

    def _downscale_frames(self, frames: list) -> list:
        tw, th = self.tracking_wh
        return [cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA) for f in frames]

    # ------------------------------------------------------------------
    # Cross-frame ECC alignment
    # ------------------------------------------------------------------

    def _estimate_cross_frame_affine(self, orig_frame: np.ndarray,
                                      clean_frame: np.ndarray) -> tuple:
        """
        Estimate affine transform orig → clean using background-masked ECC.

        Returns (M_2x3, cc) on success, (None, 0.0) on failure.
        Uses only pixels where |orig-clean| < bg_mask_threshold so that
        the actor / WAN-inpainted region does not corrupt the alignment.
        """
        diff = np.abs(orig_frame.astype(np.float32) -
                      clean_frame.astype(np.float32)).mean(axis=2)
        changed = (diff > self.bg_mask_threshold).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        bg_mask = np.where(cv2.dilate(changed, k), np.uint8(0), np.uint8(255))

        if bg_mask.mean() < 10:                # almost nothing surviving
            return None, 0.0

        gray_o = cv2.cvtColor(orig_frame,  cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_c = cv2.cvtColor(clean_frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

        M_init = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-4)
        try:
            cc, M = cv2.findTransformECC(gray_o, gray_c, M_init, cv2.MOTION_AFFINE,
                                          criteria, bg_mask, 5)
        except cv2.error as e:
            logger.debug("ECC failed: %s", e)
            return None, 0.0

        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if cc < 0.5 or abs(det) < 0.5 or abs(det) > 2.0:
            logger.debug("ECC degenerate (cc=%.3f det=%.3f)", cc, det)
            return None, float(cc)

        return M.astype(np.float32), float(cc)

    def _map_query_to_clean_space(self, orig_frame: np.ndarray, clean_frame: np.ndarray,
                                   query_pts: np.ndarray) -> tuple:
        """
        Project query_pts from original-space into clean-space via masked ECC.

        Returns (clean_pts, cc). Falls back to (query_pts.copy(), 0.0) if ECC fails.
        """
        M, cc = self._estimate_cross_frame_affine(orig_frame, clean_frame)
        if M is None:
            return query_pts.copy(), 0.0

        tw, th = clean_frame.shape[1], clean_frame.shape[0]
        sx = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        sy = float(np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2))
        logger.info("ECC orig→clean: cc=%.4f  scale=(%.4f,%.4f)  t=(%.1f,%.1f)px",
                    cc, sx, sy, M[0, 2], M[1, 2])

        q_hom = np.column_stack([query_pts, np.ones(len(query_pts), np.float32)])
        clean_pts = (q_hom @ M.T).astype(np.float32)
        clean_pts[:, 0] = clean_pts[:, 0].clip(0, tw - 1)
        clean_pts[:, 1] = clean_pts[:, 1].clip(0, th - 1)
        return clean_pts, cc

    # ------------------------------------------------------------------
    # Query point sampling
    # ------------------------------------------------------------------

    def _sample_query_pts(self, orig_frame: np.ndarray,
                           clean_frame: np.ndarray) -> np.ndarray:
        """Sample background grid points from orig_frame at tracking resolution."""
        tw, th = orig_frame.shape[1], orig_frame.shape[0]
        bg_mask = compute_bg_mask(orig_frame, clean_frame,
                                  threshold=self.bg_mask_threshold, dilation_px=10)
        step = max(1, tw // self.grid_size)
        ys, xs = np.mgrid[step // 2:th:step, step // 2:tw:step]
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        in_bg = bg_mask[pts[:, 1].astype(int).clip(0, th - 1),
                        pts[:, 0].astype(int).clip(0, tw - 1)] > 127
        return pts[in_bg] if in_bg.sum() >= 9 else pts

    # ------------------------------------------------------------------
    # Chunk flow merging
    # ------------------------------------------------------------------

    def _merge_chunk_flows(self, chunk_records: list, T: int, H: int, W: int) -> list:
        """
        Merge per-chunk flow lists into a single T-length list.

        chunk_records: [(start, end, list_of_flows), ...]
        Overlap frames are blended with smoothstep between adjacent chunks.
        """
        flows = [None] * T
        overlap = self.overlap

        for idx, (start, end, chunk_flows) in enumerate(chunk_records):
            chunk_len = end - start

            if idx == 0:
                for t in range(chunk_len):
                    flows[start + t] = chunk_flows[t]
            else:
                prev_start, prev_end, prev_flows = chunk_records[idx - 1]
                actual_overlap = prev_end - start   # how many frames actually overlap

                # Blend in the overlap region
                for j in range(min(actual_overlap, chunk_len)):
                    gframe = start + j
                    if gframe >= T:
                        break
                    alpha = (j + 1) / (actual_overlap + 1)
                    sm = alpha * alpha * (3.0 - 2.0 * alpha)      # smoothstep
                    prev_j = len(prev_flows) - actual_overlap + j
                    if 0 <= prev_j < len(prev_flows):
                        blended = ((1.0 - sm) * prev_flows[prev_j] +
                                   sm * chunk_flows[j]).astype(np.float32)
                    else:
                        blended = chunk_flows[j]
                    flows[gframe] = blended

                # Non-overlap frames
                for t in range(actual_overlap, chunk_len):
                    gframe = start + t
                    if gframe < T:
                        flows[gframe] = chunk_flows[t]

        # Fill any gaps (shouldn't happen, but be safe)
        zero = np.zeros((H, W, 2), np.float32)
        for t in range(T):
            if flows[t] is None:
                logger.warning("Flow at frame %d is None; filling with zeros", t)
                flows[t] = zero

        return flows

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_sequence(self, originals: list, cleans: list,
                         show_progress: bool = True) -> dict:
        if self.model is None:
            self.load_model()

        H, W = originals[0].shape[:2]
        tw, th = self.tracking_wh
        scale_x, scale_y = W / tw, H / th
        do_downscale = (tw != W or th != H)
        if do_downscale:
            logger.info("Downscaling to %dx%d for tracking (scale %.2fx %.2fy)",
                        tw, th, scale_x, scale_y)

        track_cleans = self._downscale_frames(cleans) if do_downscale else cleans
        track_origs  = self._downscale_frames(originals) if do_downscale else originals

        T = len(originals)
        chunk_size = self.chunk_size
        overlap = self.overlap

        # Build chunk boundaries
        chunk_starts = []
        i = 0
        while i < T:
            end = min(i + chunk_size, T)
            chunk_starts.append((i, end))
            if end == T:
                break
            i += chunk_size - overlap

        logger.info("Processing %d frames in %d chunks (size=%d overlap=%d) ecc=%s",
                    T, len(chunk_starts), chunk_size, overlap, self.per_chunk_seed_ecc)

        chunk_records = []   # (start, end, flows_list)

        progress = tqdm(chunk_starts, desc="CoTracker chunks") if show_progress else chunk_starts
        for (start, end) in progress:
            chunk_len = end - start
            orig_chunk  = track_origs[start:end]
            clean_chunk = track_cleans[start:end]

            # Fresh query points at this chunk's start frame
            query_pts = self._sample_query_pts(orig_chunk[0], clean_chunk[0])

            # Per-chunk ECC seed: map query_pts from orig-space into clean-space
            if self.per_chunk_seed_ecc:
                clean_query_pts, cc = self._map_query_to_clean_space(
                    orig_chunk[0], clean_chunk[0], query_pts)
            else:
                clean_query_pts, cc = query_pts.copy(), 0.0

            logger.info("Chunk [%d:%d] len=%d  query=%d  ECC cc=%.3f",
                        start, end, chunk_len, len(query_pts), cc)

            # Track both sequences for this chunk
            tc, vc = self._track_chunk(clean_chunk, clean_query_pts)
            to_, vo = self._track_chunk(orig_chunk, query_pts)
            torch.cuda.empty_cache()

            # Scale tracks back to full resolution
            if do_downscale:
                tc  = tc  * np.array([scale_x, scale_y], dtype=np.float32)
                to_ = to_ * np.array([scale_x, scale_y], dtype=np.float32)

            if self.smooth_trajectories:
                tc  = gaussian_filter1d(tc,  sigma=1.0, axis=0).astype(np.float32)
                to_ = gaussian_filter1d(to_, sigma=1.0, axis=0).astype(np.float32)

            # Filter points that are visible for enough of the chunk
            keep = ((vc.mean(0) >= self.min_vis_ratio) &
                    (vo.mean(0) >= self.min_vis_ratio))
            n_keep = int(keep.sum())
            logger.info("  visible tracks: %d/%d", n_keep, len(keep))

            chunk_flows = []
            if n_keep < 3:
                logger.warning("Chunk [%d:%d]: only %d reliable tracks; using identity flows",
                               start, end, n_keep)
                for _ in range(chunk_len):
                    chunk_flows.append(np.zeros((H, W, 2), np.float32))
            else:
                tc_k  = tc[:, keep, :]
                to_k  = to_[:, keep, :]
                vc_k  = vc[:, keep]
                vo_k  = vo[:, keep]
                for t in range(chunk_len):
                    fvis = vc_k[t] & vo_k[t]
                    if fvis.sum() < 3:
                        chunk_flows.append(np.zeros((H, W, 2), np.float32))
                    else:
                        _, flow, _ = self.warp_frame(
                            cleans[start + t], tc_k[t][fvis], to_k[t][fvis])
                        chunk_flows.append(flow)

            chunk_records.append((start, end, chunk_flows))

        # Merge chunk flows with smoothstep blending at boundaries
        final_flows = self._merge_chunk_flows(chunk_records, T, H, W)

        warped_frames = [warp_image_with_flow(cleans[t], final_flows[t]) for t in range(T)]
        stmaps = [flow_to_stmap(f, H, W) for f in final_flows]

        return {
            "warped_frames": warped_frames,
            "flows": final_flows,
            "stmaps": stmaps,
            "method_name": "cotracker_mesh",
        }

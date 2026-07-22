import cv2
import numpy as np
import logging

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None

try:
    from utils.flow_utils import flow_to_stmap
except ImportError:
    flow_to_stmap = None

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    _HAS_LIGHTGLUE = True
except ImportError:
    _HAS_LIGHTGLUE = False

logger = logging.getLogger(__name__)
_RANSAC_METHOD = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
_SIFT_MAX_W = 2048

# Tile / spatial balance constants
N_COLS = 8
N_ROWS = 5
MAX_PER_TILE = 100
MIN_TILES_OCCUPIED = 4
MAX_TILE_FRACTION = 0.30
MIN_INLIERS = 15
EDGE_IMPROVEMENT = 0.90


# ---------------------------------------------------------------------------
# Structural representations
# ---------------------------------------------------------------------------

def _make_structural_reps(img_bgr):
    """CLAHE grey, gradient magnitude, DoG — all colour-invariant."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe_obj.apply(gray)

    gx = cv2.Sobel(clahe_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clahe_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradmag = cv2.normalize(np.sqrt(gx**2 + gy**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 4.0)
    dog = cv2.normalize(g1 - g2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {'clahe': clahe_img, 'gradmag': gradmag, 'dog': dog}


def _negative_mask(img_bgr, border_px=15):
    """Hard-reject mask (255 = reject): only frame borders and near-black painted regions.
    Does NOT reject based on diff with the other frame — that would remove common structure."""
    H, W = img_bgr.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    mask[:border_px, :] = 255
    mask[-border_px:, :] = 255
    mask[:, :border_px] = 255
    mask[:, -border_px:] = 255
    # Near-black (actor painted out to darkness in the clean plate)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    near_black = gray < 10
    if near_black.mean() < 0.40:   # don't apply if scene is genuinely very dark
        mask[near_black] = 255
    return mask


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------

def _sift_match_pair(rep_orig, rep_clean, pos_mask_orig, pos_mask_clean, n_features=8000):
    """SIFT + RootSIFT + Lowe ratio on one representation pair.
    pos_mask_* are uint8 (255 = allowed). Returns (pts_clean, pts_orig) or (None, None)."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_o, des_o = sift.detectAndCompute(rep_orig, pos_mask_orig)
    kp_c, des_c = sift.detectAndCompute(rep_clean, pos_mask_clean)
    if des_o is None or des_c is None or len(des_o) < 4 or len(des_c) < 4:
        return None, None
    eps = 1e-8
    des_o = np.sqrt(des_o / (des_o.sum(axis=1, keepdims=True) + eps))
    des_c = np.sqrt(des_c / (des_c.sum(axis=1, keepdims=True) + eps))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des_c, des_o, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None, None
    pts_c = np.float32([kp_c[m.queryIdx].pt for m in good])
    pts_o = np.float32([kp_o[m.trainIdx].pt for m in good])
    return pts_c, pts_o


def _lightglue_match_pair(orig_bgr, clean_bgr, scale=1.0, device='cuda'):
    """LightGlue SuperPoint matching. Returns (pts_clean, pts_orig) scaled to full res, or (None, None)."""
    if not _HAS_LIGHTGLUE:
        return None, None
    import torch
    try:
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)

        def to_t(gray):
            return torch.from_numpy(gray[None, None]).float().to(device) / 255.0

        gray_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
        with torch.no_grad():
            f0 = extractor.extract(to_t(gray_o))
            f1 = extractor.extract(to_t(gray_c))
            m01 = matcher({'image0': f0, 'image1': f1})
            f0, f1, m01 = rbd(f0), rbd(f1), rbd(m01)

        matches = m01['matches'].cpu().numpy()   # (M, 2)
        if len(matches) < 4:
            return None, None
        kp0 = f0['keypoints'].cpu().numpy()
        kp1 = f1['keypoints'].cpu().numpy()
        pts_o = (kp0[matches[:, 0]] / scale).astype(np.float32)
        pts_c = (kp1[matches[:, 1]] / scale).astype(np.float32)
        return pts_c, pts_o
    except Exception as e:
        logger.warning("LightGlue failed: %s", e)
        return None, None


# ---------------------------------------------------------------------------
# Spatial tile balancing
# ---------------------------------------------------------------------------

def _tile_balance(pts_clean, pts_orig, frame_H, frame_W,
                  n_cols=N_COLS, n_rows=N_ROWS, max_per=MAX_PER_TILE):
    if len(pts_clean) == 0:
        return pts_clean, pts_orig, {}
    tile_w = frame_W / n_cols
    tile_h = frame_H / n_rows
    col_ids = np.floor(pts_clean[:, 0] / tile_w).astype(int).clip(0, n_cols - 1)
    row_ids = np.floor(pts_clean[:, 1] / tile_h).astype(int).clip(0, n_rows - 1)
    tile_ids = col_ids * n_rows + row_ids

    keep_idxs = []
    tile_counts = {}
    for tid in np.unique(tile_ids):
        idxs = np.where(tile_ids == tid)[0]
        tile_counts[int(tid)] = len(idxs)
        if len(idxs) > max_per:
            idxs = np.random.choice(idxs, max_per, replace=False)
        keep_idxs.extend(idxs.tolist())

    keep = np.array(keep_idxs)
    return pts_clean[keep], pts_orig[keep], tile_counts


def _inlier_coverage(pts_inliers, frame_H, frame_W, n_cols=N_COLS, n_rows=N_ROWS):
    """Returns (n_occupied_tiles, max_tile_fraction)."""
    if len(pts_inliers) == 0:
        return 0, 1.0
    tile_w = frame_W / n_cols
    tile_h = frame_H / n_rows
    col_ids = np.floor(pts_inliers[:, 0] / tile_w).astype(int).clip(0, n_cols - 1)
    row_ids = np.floor(pts_inliers[:, 1] / tile_h).astype(int).clip(0, n_rows - 1)
    tile_ids = col_ids * n_rows + row_ids
    unique, counts = np.unique(tile_ids, return_counts=True)
    return len(unique), float(counts.max() / len(pts_inliers))


# ---------------------------------------------------------------------------
# Edge score validation
# ---------------------------------------------------------------------------

def _edge_score(orig_bgr, warped_bgr, downscale_to=1280):
    """Median chamfer distance: how far are warped-clean edges from original edges.
    Lower is better. Returns float (px at downscaled res), inf if too few edges."""
    H, W = orig_bgr.shape[:2]
    s = min(1.0, downscale_to / W)
    o_s = cv2.resize(orig_bgr, (int(W * s), int(H * s)))
    w_s = cv2.resize(warped_bgr, (int(W * s), int(H * s)))

    def canny(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        return cv2.Canny(c, 60, 140)

    o_edges = canny(o_s)
    w_edges = canny(w_s)
    dt = cv2.distanceTransform(255 - o_edges, cv2.DIST_L2, 3)
    ys, xs = np.where(w_edges > 0)
    if len(xs) < 100:
        return np.inf
    return float(np.median(dt[ys, xs]))


# ---------------------------------------------------------------------------
# Geometry fitting
# ---------------------------------------------------------------------------

def _fit_and_validate(pts_clean, pts_orig, model, ransac_thresh,
                      frame_H, frame_W, min_inliers=MIN_INLIERS):
    """Fit model (translation|affine|homography) and enforce spatial constraints.
    Returns (M, inlier_count, n_tiles, max_frac, inlier_mask) or None."""
    if len(pts_clean) < 4:
        return None

    if model == 'translation':
        shifts = pts_orig - pts_clean
        med = np.median(shifts, axis=0)
        M = np.float64([[1, 0, med[0]], [0, 1, med[1]]])
        err = np.linalg.norm((pts_clean + med) - pts_orig, axis=1)
        inlier_mask = err < ransac_thresh

    elif model == 'affine':
        M, _ = cv2.estimateAffine2D(
            pts_clean.reshape(-1, 1, 2), pts_orig.reshape(-1, 1, 2),
            method=_RANSAC_METHOD, ransacReprojThreshold=ransac_thresh,
            maxIters=10000, confidence=0.999)
        if M is None:
            return None
        proj = cv2.transform(pts_clean.reshape(-1, 1, 2), M).reshape(-1, 2)
        inlier_mask = np.linalg.norm(proj - pts_orig, axis=1) < ransac_thresh

    else:  # homography
        M, _ = cv2.findHomography(
            pts_clean.reshape(-1, 1, 2), pts_orig.reshape(-1, 1, 2),
            _RANSAC_METHOD, ransac_thresh, maxIters=10000, confidence=0.999)
        if M is None:
            return None
        proj = cv2.perspectiveTransform(pts_clean.reshape(-1, 1, 2), M).reshape(-1, 2)
        inlier_mask = np.linalg.norm(proj - pts_orig, axis=1) < ransac_thresh

    inlier_mask = inlier_mask.astype(bool).flatten()
    inlier_count = int(inlier_mask.sum())
    if inlier_count < min_inliers:
        return None

    n_tiles, max_frac = _inlier_coverage(pts_clean[inlier_mask], frame_H, frame_W)
    if n_tiles < MIN_TILES_OCCUPIED:
        logger.debug("  model=%s rejected: %d tiles < %d required", model, n_tiles, MIN_TILES_OCCUPIED)
        return None
    if max_frac > MAX_TILE_FRACTION:
        logger.debug("  model=%s rejected: max_frac=%.2f > %.2f (matches too concentrated)", model, max_frac, MAX_TILE_FRACTION)
        return None

    return M, inlier_count, n_tiles, max_frac, inlier_mask


def _homogenize(M, model):
    """Promote any fit matrix to 3×3 float64 for warpPerspective."""
    if model == 'translation':
        H3 = np.eye(3, dtype=np.float64)
        H3[0, 2] = float(M[0, 2])
        H3[1, 2] = float(M[1, 2])
        return H3
    elif model == 'affine':
        H3 = np.eye(3, dtype=np.float64)
        H3[:2, :] = M.astype(np.float64)
        return H3
    else:
        return M.astype(np.float64)


def _flow_from_H(H3, frame_H, frame_W):
    """Backward flow (H, W, 2): for each orig pixel, offset to sample in clean."""
    H_inv = np.linalg.inv(H3)
    gx, gy = np.meshgrid(np.arange(frame_W), np.arange(frame_H))
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)
    mapped = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H_inv).reshape(frame_H, frame_W, 2)
    return (mapped - grid).astype(np.float32)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StructuralMatchMethod:
    """
    Per-frame plate registration.

    DESIGN PRINCIPLE: never use a diff mask to identify background.
    WAN changes colors/textures everywhere — a diff mask deletes exactly the common
    structure we need. Instead:
      - Match everywhere using structural representations (CLAHE, gradmag, DoG)
      - Enforce spatial tile balance so actor-dominated correspondences can't win
      - Gate on edge-score improvement: only accept transforms that measurably
        improve structural alignment
    """
    MODELS = ['translation', 'affine', 'homography']

    def __init__(
        self,
        sift_n_features: int = 8000,
        representations=None,           # ['clahe', 'gradmag', 'dog']
        ransac_thresh: float = 4.0,
        min_inliers: int = 15,
        temporal_sigma: float = 1.5,
        use_lightglue: bool = True,
        lightglue_max_w: int = 1600,
        border_reject_px: int = 15,
        edge_improvement_threshold: float = EDGE_IMPROVEMENT,
    ):
        self.sift_n_features = sift_n_features
        self.representations = representations or ['clahe', 'gradmag', 'dog']
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.temporal_sigma = temporal_sigma
        self.use_lightglue = use_lightglue
        self.lightglue_max_w = lightglue_max_w
        self.border_reject_px = border_reject_px
        self.edge_improvement_threshold = edge_improvement_threshold

    def process_sequence(self, originals, cleans, show_progress=True):
        T = len(originals)
        if T == 0 or T != len(cleans):
            return {"warped_frames": [], "flows": [], "stmaps": [],
                    "debug_per_frame": [], "method_name": "StructuralMatch"}

        fH, fW = originals[0].shape[:2]
        H_list, debug_list = [], []

        for t in range(T):
            if show_progress and t % 10 == 0:
                logger.info("StructuralMatch frame %d/%d", t, T)
            H3, dbg = self._process_frame(originals[t], cleans[t], t, fH, fW)
            H_list.append(H3)
            debug_list.append(dbg)

        # Temporal smoothing
        if self.temporal_sigma > 0 and gaussian_filter1d is not None and T > 1:
            arr = gaussian_filter1d(np.stack(H_list, axis=0), sigma=self.temporal_sigma, axis=0, mode='nearest')
            H_list = [arr[i] for i in range(T)]
        elif self.temporal_sigma > 0 and gaussian_filter1d is None:
            logger.warning("scipy unavailable — temporal smoothing skipped")

        warped_frames, flows, stmaps, debug_per_frame = [], [], [], []
        for t in range(T):
            H3 = H_list[t]
            warped = cv2.warpPerspective(cleans[t], H3, (fW, fH),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            warped_frames.append(warped)
            flow = _flow_from_H(H3, fH, fW)
            flows.append(flow)
            stmaps.append(flow_to_stmap(flow, fH, fW) if flow_to_stmap else flow)

            corners = np.float32([[0, 0], [fW, 0], [fW, fH], [0, fH]]).reshape(-1, 1, 2)
            corner_disp = float(np.mean(np.linalg.norm(
                cv2.perspectiveTransform(corners, H3).reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))
            d = debug_list[t].copy()
            d['corner_displacement_px'] = corner_disp
            d['H'] = H3.tolist()
            debug_per_frame.append(d)

        return {"warped_frames": warped_frames, "flows": flows, "stmaps": stmaps,
                "debug_per_frame": debug_per_frame, "method_name": "StructuralMatch"}

    def _process_frame(self, orig, clean, frame_idx, fH, fW):
        reps_o = _make_structural_reps(orig)
        reps_c = _make_structural_reps(clean)

        rej_o = _negative_mask(orig, self.border_reject_px)
        rej_c = _negative_mask(clean, self.border_reject_px)
        pos_o = cv2.bitwise_not(rej_o)
        pos_c = cv2.bitwise_not(rej_c)

        max_dim = max(fH, fW)
        scale = _SIFT_MAX_W / max_dim if max_dim > _SIFT_MAX_W else 1.0
        sw, sh = int(fW * scale), int(fH * scale)

        def rs(r): return cv2.resize(r, (sw, sh), interpolation=cv2.INTER_AREA)
        def rm(m):
            m_s = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
            _, m_s = cv2.threshold(m_s, 127, 255, cv2.THRESH_BINARY)
            return m_s

        pos_o_s, pos_c_s = rm(pos_o), rm(pos_c)
        all_pts_c, all_pts_o, match_counts = [], [], {}

        for rep_name in self.representations:
            pc, po = _sift_match_pair(rs(reps_o[rep_name]), rs(reps_c[rep_name]),
                                      pos_o_s, pos_c_s, self.sift_n_features)
            if pc is not None and len(pc) > 0:
                all_pts_c.append(pc / scale)
                all_pts_o.append(po / scale)
                match_counts[rep_name] = len(pc)
            else:
                match_counts[rep_name] = 0

        lg_matches = 0
        if self.use_lightglue and _HAS_LIGHTGLUE:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            lg_s = min(1.0, self.lightglue_max_w / fW)
            pc_lg, po_lg = _lightglue_match_pair(
                cv2.resize(orig, (int(fW*lg_s), int(fH*lg_s))),
                cv2.resize(clean, (int(fW*lg_s), int(fH*lg_s))),
                scale=lg_s, device=device)
            if pc_lg is not None and len(pc_lg) > 0:
                all_pts_c.append(pc_lg)
                all_pts_o.append(po_lg)
                lg_matches = len(pc_lg)
            match_counts['lightglue'] = lg_matches

        total_raw = sum(len(p) for p in all_pts_c)
        if total_raw == 0:
            logger.warning("Frame %d: zero matches from all matchers, using identity", frame_idx)
            return np.eye(3, dtype=np.float64), {
                'model_used': 'identity', 'total_matches': 0, 'balanced_matches': 0,
                'inliers': 0, 'n_tiles': 0, 'max_tile_frac': 1.0,
                'edge_before': None, 'edge_after': None, 'match_counts': match_counts}

        pts_c_all = np.concatenate(all_pts_c, axis=0)
        pts_o_all = np.concatenate(all_pts_o, axis=0)
        pts_c_bal, pts_o_bal, _ = _tile_balance(pts_c_all, pts_o_all, fH, fW)

        edge_before = _edge_score(orig, clean)
        best_H3 = None
        best_edge = edge_before
        best_model = 'identity'
        best_dbg = {'inliers': 0, 'n_tiles': 0, 'max_tile_frac': 1.0}
        # Fallback: best model passing spatial checks even if edge gate fails.
        # Edge score is unreliable when actor dominates (clean background edges land
        # near dense actor edges in the distance transform, masking true improvement).
        spatial_fallback_H3 = None
        spatial_fallback_model = 'identity'
        spatial_fallback_dbg = {'inliers': 0, 'n_tiles': 0, 'max_tile_frac': 1.0}

        for model_name in self.MODELS:
            result = _fit_and_validate(pts_c_bal, pts_o_bal, model_name,
                                       self.ransac_thresh, fH, fW, self.min_inliers)
            if result is None:
                continue
            M_raw, n_inl, n_tiles, max_frac, _ = result
            H3c = _homogenize(M_raw, model_name)
            warped_cand = cv2.warpPerspective(clean, H3c, (fW, fH),
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            edge_after = _edge_score(orig, warped_cand)
            logger.info("f%03d  %s  inliers=%d  tiles=%d  max_frac=%.2f  edge %.2f→%.2f",
                        frame_idx, model_name, n_inl, n_tiles, max_frac, edge_before, edge_after)

            # Primary: edge score improvement gate
            if edge_after < best_edge * self.edge_improvement_threshold:
                best_H3 = H3c
                best_edge = edge_after
                best_model = model_name
                best_dbg = {'inliers': n_inl, 'n_tiles': n_tiles, 'max_tile_frac': max_frac}

            # Spatial fallback: keep highest-inlier spatially-valid model
            if n_inl > spatial_fallback_dbg['inliers']:
                spatial_fallback_H3 = H3c
                spatial_fallback_model = model_name
                spatial_fallback_dbg = {'inliers': n_inl, 'n_tiles': n_tiles, 'max_tile_frac': max_frac}

        if best_H3 is None and spatial_fallback_H3 is not None:
            logger.warning("f%03d  edge gate failed (before=%.2f); using spatial fallback: %s  inliers=%d  tiles=%d",
                           frame_idx, edge_before, spatial_fallback_model,
                           spatial_fallback_dbg['inliers'], spatial_fallback_dbg['n_tiles'])
            best_H3 = spatial_fallback_H3
            best_model = spatial_fallback_model + '_sfb'
            best_dbg = spatial_fallback_dbg

        if best_H3 is None:
            logger.warning("f%03d  no valid model found, using identity", frame_idx)
            best_H3 = np.eye(3, dtype=np.float64)

        return best_H3, {
            'model_used': best_model,
            'total_matches': total_raw,
            'balanced_matches': len(pts_c_bal),
            'match_counts': match_counts,
            'edge_before': float(edge_before) if np.isfinite(edge_before) else None,
            'edge_after': float(best_edge) if np.isfinite(best_edge) else None,
            **best_dbg,
        }

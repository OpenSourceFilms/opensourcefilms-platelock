import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter


def warp_image_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp source image using forward flow (dx,dy pixel displacements).

    flow[y,x] = (dx,dy) means source pixel (x,y) maps to (x+dx, y+dy) in the other frame.
    To reconstruct output(x,y) from source: sample source at (x+dx, y+dy).
    This is consistent with flow_to_stmap which also uses +flow.
    """
    H, W = flow.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(W, dtype=np.float32),
                                      np.arange(H, dtype=np.float32))
    map_x = x_coords + flow[..., 0]
    map_y = y_coords + flow[..., 1]
    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def flow_to_stmap(flow: np.ndarray, H: int, W: int, direction: str = 'backward') -> np.ndarray:
    """
    Convert flow to Nuke-compatible STMap (H,W,2) with U,V in [0,1].

    Nuke STMap origin is BOTTOM-LEFT; optical flow is TOP-LEFT.
    Apply V-flip: V = 1 - (y_src / H)
    """
    if direction != 'backward':
        raise NotImplementedError("Only 'backward' direction implemented.")
    x_src = np.arange(W, dtype=np.float32).reshape(1, W) + flow[..., 0]
    y_src = np.arange(H, dtype=np.float32).reshape(H, 1) + flow[..., 1]
    U = x_src / W
    V = 1.0 - (y_src / H)  # V-flip: Nuke is bottom-left origin, flow is top-left
    return np.stack([U, V], axis=-1).astype(np.float32)


def visualise_flow(flow: np.ndarray) -> np.ndarray:
    """HSV colour-wheel visualisation (hue=direction, value=magnitude). Returns BGR uint8."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    mag_99 = np.percentile(mag, 99)
    mag = np.clip(mag / mag_99 * 255, 0, 255) if mag_99 > 0 else np.zeros_like(mag)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang.astype(np.float32) / 360.0 * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = mag.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def write_flow_npy(flow: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), flow)


def read_flow_npy(path: Path) -> np.ndarray:
    return np.load(str(path))


def smooth_flow_spatial(flow: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur each channel independently."""
    return gaussian_filter(flow, sigma=(sigma_px, sigma_px, 0)).astype(np.float32)


def smooth_flow_temporal(flows: list, radius: int) -> list:
    """Per-frame median across a sliding window."""
    n = len(flows)
    smoothed = []
    for i in range(n):
        window = np.stack(flows[max(0, i - radius) : i + radius + 1], axis=0)
        smoothed.append(np.median(window, axis=0).astype(np.float32))
    return smoothed


def clamp_flow(flow: np.ndarray, p: int = 95, multiplier: float = 3.0) -> np.ndarray:
    """Clamp flow magnitudes to multiplier * pth-percentile, preserving direction."""
    mag = np.linalg.norm(flow, axis=-1)
    threshold = multiplier * np.percentile(mag, p)
    exceed = mag > threshold
    if np.any(exceed) and threshold > 0:
        scale = threshold / mag[exceed]
        flow = flow.copy()
        flow[exceed, 0] *= scale
        flow[exceed, 1] *= scale
    return flow


def blend_flows(flow_a: np.ndarray, flow_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Linear blend: (1-alpha)*a + alpha*b."""
    return ((1.0 - alpha) * flow_a + alpha * flow_b).astype(np.float32)


def affine_to_flow(M: np.ndarray, H: int, W: int) -> np.ndarray:
    """Convert a 2×3 or 3×3 warp matrix (WARP_INVERSE_MAP convention) to a flow field (H,W,2).

    With WARP_INVERSE_MAP: dst(x,y) = src(M*(x,y)), so
    flow[y,x] = (src_x - x, src_y - y).
    Accepts 2×3 affine or 3×3 homography matrices.
    """
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    if M.shape == (3, 3):
        pts = np.stack([x.ravel(), y.ravel()], axis=-1).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M).reshape(H, W, 2)
        src_x, src_y = dst[..., 0], dst[..., 1]
    else:
        src_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        src_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return np.stack([src_x - x, src_y - y], axis=-1).astype(np.float32)


def compute_bg_mask(orig: np.ndarray, clean: np.ndarray,
                    threshold: int = 30, dilation_px: int = 20) -> np.ndarray:
    """
    Return a uint8 mask (255=reliable background, 0=AI-changed region).

    Pixels where |orig - clean| > threshold are AI-changed content and
    unreliable for registration. Dilated to avoid edge contamination.
    """
    diff = np.abs(orig.astype(np.float32) - clean.astype(np.float32)).mean(axis=2)
    changed = (diff > threshold).astype(np.uint8)
    if dilation_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        changed = cv2.dilate(changed, k)
    return np.where(changed, np.uint8(0), np.uint8(255))


def inpaint_flow_masked(flow: np.ndarray, bg_mask: np.ndarray,
                        propagation_sigma: float = 60.0) -> np.ndarray:
    """
    Replace flow in AI-changed regions (bg_mask==0) with a weighted Gaussian
    propagation of the reliable background flow (bg_mask==255).

    Uses distance-weighted averaging so nearby reliable vectors dominate.
    """
    weights = (bg_mask.astype(np.float32) / 255.0)  # 1=reliable, 0=unreliable
    result = flow.copy()
    for c in range(2):
        ch = flow[..., c]
        num = gaussian_filter(ch * weights, sigma=propagation_sigma)
        den = gaussian_filter(weights, sigma=propagation_sigma) + 1e-7
        smooth = num / den
        # keep original values where reliable, use propagated where not
        result[..., c] = np.where(weights < 0.5, smooth, ch)
    return result.astype(np.float32)

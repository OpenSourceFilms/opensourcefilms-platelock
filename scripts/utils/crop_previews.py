"""
Crop-based preview generation for VFX plate-lock quality assessment.

Generates tight crops of background reference regions so alignment quality
is visible without the full-frame distraction of actor or inpainted content.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Crop regions defined in 1920×1080 space (x, y, w, h).
# Auto-scaled to the actual working resolution in generate_crop_previews.
DEFAULT_CROPS_1080P = {
    "right_wall":        (1388,  80, 450, 640),
    "right_edge_window": (1650,   0, 270, 560),
    "center_background": ( 640, 160, 640, 480),
    "top_strip":         ( 320,   0, 1280, 200),
}


def _scale_crop(crop: tuple, src_w: int, src_h: int) -> tuple:
    x, y, w, h = crop
    return (int(x * src_w / 1920), int(y * src_h / 1080),
            max(1, int(w * src_w / 1920)), max(1, int(h * src_h / 1080)))


def _extract_crop(frame: np.ndarray, crop: tuple) -> np.ndarray:
    x, y, w, h = crop
    H, W = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    return frame[y1:y2, x1:x2]


def _make_edge_map(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    return np.stack([edges, edges, edges], axis=-1)


def _overlay_edges_green_red(orig_crop: np.ndarray,
                              warped_crop: np.ndarray) -> np.ndarray:
    """Original edges → green channel; warped edges → red channel."""
    e_o = _make_edge_map(orig_crop)
    e_w = _make_edge_map(warped_crop)
    out = np.zeros_like(orig_crop)
    out[..., 1] = e_o[..., 0]   # green = original
    out[..., 0] = e_w[..., 0]   # red   = warped
    return out


def _write_mp4(frames: list, out_path: Path, fps: float = 24.0, crf: int = 18) -> None:
    import imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                 output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"])
    try:
        for f in frames:
            writer.append_data(f if f.dtype == np.uint8 else f.clip(0, 255).astype(np.uint8))
    finally:
        writer.close()


def generate_crop_previews(originals: list, warped_frames: list,
                            out_dir, fps: float = 24.0,
                            crops: dict | None = None) -> None:
    """
    Write per-crop alignment preview videos into out_dir.

    For each named crop region:
      <crop>_overlay.mp4        50% alpha blend of orig and warped
      <crop>_edges.mp4          original edges (green) vs warped edges (red)
      <crop>_side_by_side.mp4   original | warped horizontal stack
    """
    if not originals:
        return
    fh, fw = originals[0].shape[:2]
    if crops is None:
        crops = {n: _scale_crop(c, fw, fh) for n, c in DEFAULT_CROPS_1080P.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, crop_box in crops.items():
        oc, wc_list, blend, edges, sbs = [], [], [], [], []
        for orig, warped in zip(originals, warped_frames):
            c_o = _extract_crop(orig,   crop_box)
            c_w = _extract_crop(warped, crop_box)
            h = min(c_o.shape[0], c_w.shape[0])
            w = min(c_o.shape[1], c_w.shape[1])
            c_o, c_w = c_o[:h, :w], c_w[:h, :w]
            blend.append(((c_o.astype(np.float32) + c_w.astype(np.float32)) * 0.5)
                         .clip(0, 255).astype(np.uint8))
            edges.append(_overlay_edges_green_red(c_o, c_w))
            sbs.append(np.concatenate([c_o, c_w], axis=1))

        _write_mp4(blend, out_dir / f"{name}_overlay.mp4",       fps=fps)
        _write_mp4(edges, out_dir / f"{name}_edges.mp4",          fps=fps)
        _write_mp4(sbs,   out_dir / f"{name}_side_by_side.mp4",   fps=fps)
        logger.info("Crop previews done: %s → %s", name, out_dir)

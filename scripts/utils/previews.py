import imageio
import numpy as np
import cv2
from pathlib import Path


def make_side_by_side(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    out = np.hstack((frame_a, frame_b))
    w = frame_a.shape[1]
    out[:, w - 1 : w + 1] = 255
    return out


def make_difference(original: np.ndarray, warped: np.ndarray, amplify: float = 5.0) -> np.ndarray:
    diff = cv2.absdiff(original, warped).astype(np.float32) * amplify
    return np.clip(diff, 0, 255).astype(np.uint8)


def make_checkerboard(frame_a: np.ndarray, frame_b: np.ndarray, tile_size: int = 64) -> np.ndarray:
    h, w = frame_a.shape[:2]
    out = np.zeros_like(frame_a)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            src = frame_a if ((y // tile_size) + (x // tile_size)) % 2 == 0 else frame_b
            ht = min(tile_size, h - y)
            wt = min(tile_size, w - x)
            out[y:y+ht, x:x+wt] = src[y:y+ht, x:x+wt]
    return out


def make_wipe(frame_a: np.ndarray, frame_b: np.ndarray, position: float = 0.5) -> np.ndarray:
    h, w = frame_a.shape[:2]
    pivot = int(position * w)
    out = frame_a.copy()
    if pivot < w:
        out[:, pivot:] = frame_b[:, pivot:]
    if 0 < pivot < w:
        out[:, max(0, pivot-1) : pivot+1] = 255
    return out


def make_overlay_50(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(frame_a, 0.5, frame_b, 0.5, 0)


_PREVIEW_MAX_W = 1920  # cap single-frame width; side-by-side will be 2×this


def _scale_frame(frame: np.ndarray, max_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def generate_all_previews(originals: list, warpeds: list, out_dir: Path,
                           fps: float = 24.0, crf: int = 23) -> None:
    """Stream all preview videos frame-by-frame to avoid accumulating large lists in memory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _open(name):
        p = out_dir / name
        return imageio.get_writer(str(p), fps=fps, format="FFMPEG",
                                   codec="libx264", output_params=["-crf", str(crf)])

    ws = {
        "warped_clean": _open("warped_clean.mp4"),
        "side_by_side": _open("side_by_side.mp4"),
        "difference":   _open("difference.mp4"),
        "checkerboard": _open("checkerboard.mp4"),
        "wipe_preview": _open("wipe_preview.mp4"),
        "overlay_50":   _open("overlay_50.mp4"),
    }
    for orig, warp in zip(originals, warpeds):
        o = _scale_frame(orig, _PREVIEW_MAX_W)
        w_ = _scale_frame(warp, _PREVIEW_MAX_W)
        ws["warped_clean"].append_data(w_)
        ws["side_by_side"].append_data(make_side_by_side(o, w_))
        ws["difference"].append_data(make_difference(o, w_))
        ws["checkerboard"].append_data(make_checkerboard(o, w_))
        ws["wipe_preview"].append_data(make_wipe(o, w_))
        ws["overlay_50"].append_data(make_overlay_50(o, w_))
    for w in ws.values():
        w.close()


def save_comparison_stills(originals: list, warpeds: list, out_dir: Path, frame_indices: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in frame_indices:
        if idx < 0 or idx >= len(originals):
            continue
        a, b = originals[idx], warpeds[idx]
        cv2.imwrite(str(out_dir / f"frame{idx:04d}_side_by_side.png"),
                    cv2.cvtColor(make_side_by_side(a, b), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"frame{idx:04d}_difference.png"),
                    cv2.cvtColor(make_difference(a, b), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"frame{idx:04d}_wipe.png"),
                    cv2.cvtColor(make_wipe(a, b), cv2.COLOR_RGB2BGR))

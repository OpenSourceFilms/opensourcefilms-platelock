import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import sys

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    ssim_func = None
    print("[WARN] skimage.metrics.structural_similarity unavailable; SSIM=0.0", file=sys.stderr)


_SSIM_MAX_PX = 1920 * 1080  # downsample for SSIM only; compositing metrics stay at full res


def compute_frame_metrics(original: np.ndarray, warped: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Per-frame quality metrics. Both inputs are HxWxC uint8 or float32."""
    orig = original.astype(np.float32) / 255.0 if original.dtype == np.uint8 else original.astype(np.float32)
    warp = warped.astype(np.float32) / 255.0 if warped.dtype == np.uint8 else warped.astype(np.float32)

    mask_bool = (mask > 0) if mask is not None else None
    orig_m = orig[mask_bool].ravel() if mask_bool is not None else orig.ravel()
    warp_m = warp[mask_bool].ravel() if mask_bool is not None else warp.ravel()
    n_pixels = orig_m.size

    if n_pixels == 0:
        return dict(mae=0.0, mse=0.0, rmse=0.0, psnr=float('inf'), ssim=1.0, n_pixels=0)

    diff = orig_m - warp_m
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    psnr = 20.0 * float(np.log10(1.0 / rmse)) if rmse > 0 else float('inf')

    channel_axis = 2 if orig.ndim == 3 else None
    if ssim_func is not None:
        # Downsample to ≤1080p for SSIM — full 5K float64 is ~20s/frame on CPU
        h, w = orig.shape[:2]
        if h * w > _SSIM_MAX_PX:
            scale = (_SSIM_MAX_PX / (h * w)) ** 0.5
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            o_s = cv2.resize(orig, (nw, nh), interpolation=cv2.INTER_AREA)
            w_s = cv2.resize(warp, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            o_s, w_s = orig, warp
        ssim = float(ssim_func(o_s, w_s, data_range=1.0, channel_axis=channel_axis))
    else:
        ssim = 0.0

    return dict(mae=mae, mse=mse, rmse=rmse, psnr=psnr, ssim=ssim, n_pixels=n_pixels)


def compute_sequence_metrics(originals: list, warpeds: list, masks: list | None = None,
                              method_name: str = '', output_dir: 'Path | None' = None) -> pd.DataFrame:
    rows = []
    for i, (orig, warp) in enumerate(zip(originals, warpeds)):
        m = compute_frame_metrics(orig, warp, masks[i] if masks else None)
        rows.append({'frame': i, 'method': method_name, **m})
    df = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'per_frame.csv', index=False)
    return df


def summarise_metrics(df: pd.DataFrame) -> dict:
    return dict(mean_psnr=float(df['psnr'].mean()), median_psnr=float(df['psnr'].median()),
                mean_ssim=float(df['ssim'].mean()), mean_mae=float(df['mae'].mean()),
                worst_5_frames=find_worst_frames(df, 5, 'psnr'))


def find_worst_frames(df: pd.DataFrame, n: int = 5, metric: str = 'psnr') -> list:
    """Return frame indices of n worst frames (lowest psnr/ssim or highest mae)."""
    worst = df.nsmallest(n, metric) if metric in ('psnr', 'ssim') else df.nlargest(n, metric)
    return worst['frame'].tolist()

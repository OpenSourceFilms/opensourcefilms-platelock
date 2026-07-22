"""VFX-appropriate quality metrics for plate-lock evaluation.

PSNR/SSIM measure pixel accuracy. For compositing, what matters is:
  1. Background edge alignment  — are edges lining up in the safe region?
  2. Temporal warp jerk         — does the lock swim or jitter over time?
  3. STMap spatial smoothness   — does the warp field have discontinuities?
"""
import numpy as np
import cv2


def masked_edge_alignment(orig, warped, bg_mask, canny_low=40, canny_high=120):
    """Mean chamfer distance (px) between background edges. Lower = better."""
    gray_o = cv2.cvtColor(orig,   cv2.COLOR_RGB2GRAY)
    gray_w = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

    safe = (bg_mask > 127).astype(np.uint8) * 255
    edges_o = cv2.bitwise_and(cv2.Canny(gray_o, canny_low, canny_high), safe)
    edges_w = cv2.bitwise_and(cv2.Canny(gray_w, canny_low, canny_high), safe)

    dist = cv2.distanceTransform((edges_w == 0).astype(np.uint8), cv2.DIST_L2, 5)

    ys, xs = np.where(edges_o > 0)
    n = len(ys)
    if n == 0:
        return dict(mean_chamfer_px=float("inf"),
                    recall_1px=0.0, recall_2px=0.0, recall_3px=0.0, n_edge_pts=0)

    dists = dist[ys, xs]
    return dict(
        mean_chamfer_px=float(dists.mean()),
        recall_1px=float((dists < 1.0).mean()),
        recall_2px=float((dists < 2.0).mean()),
        recall_3px=float((dists < 3.0).mean()),
        n_edge_pts=n,
    )


def temporal_jerk(warp_params, kind="matrix"):
    """2nd derivative of warp magnitude. Higher = more swimming/jitter."""
    if len(warp_params) < 3:
        return dict(mean_jerk=0.0, max_jerk=0.0)

    first = warp_params[0]
    if isinstance(first, np.ndarray) and first.ndim == 3:
        mag = np.array([float(np.mean(np.linalg.norm(f, axis=-1))) for f in warp_params])
    else:
        tx = np.array([float(m[0, 2]) for m in warp_params])
        ty = np.array([float(m[1, 2]) for m in warp_params])
        mag = np.sqrt(tx ** 2 + ty ** 2)

    jerk = np.abs(np.diff(mag, n=2))
    return dict(mean_jerk=float(jerk.mean()), max_jerk=float(jerk.max()))


def stmap_smoothness(stmap):
    """Spatial gradient magnitude of U/V channels. Lower = smoother warp."""
    u = stmap[..., 0].astype(np.float32)
    v = stmap[..., 1].astype(np.float32)

    def grad_mag(ch):
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    gu = grad_mag(u)
    gv = grad_mag(v)
    return dict(
        mean_grad_u=float(gu.mean()), mean_grad_v=float(gv.mean()),
        p99_grad_u=float(np.percentile(gu, 99)),
        p99_grad_v=float(np.percentile(gv, 99)),
    )


def compute_compositing_metrics(originals, warpeds, bg_masks=None,
                                 flows_or_warps=None, stmaps=None):
    """Aggregate compositing metrics over a full sequence."""
    edge_results = [
        masked_edge_alignment(orig, warped,
                              bg_masks[i] if bg_masks else
                              np.full(orig.shape[:2], 255, np.uint8))
        for i, (orig, warped) in enumerate(zip(originals, warpeds))
    ]

    jerk = temporal_jerk(flows_or_warps) if flows_or_warps else {}

    stmap_r = {}
    if stmaps:
        sm_list = [stmap_smoothness(s) for s in stmaps]
        stmap_r = dict(
            mean_stmap_grad_u=float(np.mean([s["mean_grad_u"] for s in sm_list])),
            mean_stmap_grad_v=float(np.mean([s["mean_grad_v"] for s in sm_list])),
        )

    return dict(
        mean_chamfer_px=float(np.mean([r["mean_chamfer_px"] for r in edge_results
                                        if r["mean_chamfer_px"] != float("inf")])),
        mean_edge_recall_2px=float(np.mean([r["recall_2px"] for r in edge_results])),
        **jerk,
        **stmap_r,
    )

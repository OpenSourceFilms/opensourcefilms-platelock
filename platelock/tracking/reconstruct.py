"""
platelock.tracking.reconstruct — Stage B: per-frame decomposed reconstruction
+ temporal parameter smoothing + spatial residual low-pass.

Consumes Stage A's anchor pairs (trO, trC, vi) and produces a savgol/blur-
smoothed (M, cres) sequence -- the SAME assembly unit as the RoMa-dense branch
(platelock.dense_field.assemble_map), so downstream export/microlock/hole-fill
code doesn't care which correspondence source produced it.
"""
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from platelock import dense_field as df, diagnostics as dg, geometry as g


def _hole_pin_anchors(o, M, W, H, stride, margin):
    """Affine-pin virtual anchors on a coarse lattice far from any real anchor,
    so the TPS residual can never extrapolate unboundedly in an unsupported hole."""
    hx, hy = np.meshgrid(np.arange(stride // 2, W, stride), np.arange(stride // 2, H, stride))
    holes = np.stack([hx.ravel(), hy.ravel()], 1).astype(np.float32)
    dist, _ = cKDTree(o).query(holes)
    hp = holes[dist > margin]
    if len(hp) == 0 or M is None:
        return None, None
    return hp, (hp @ M[:, :2].T) + M[:, 2]


def reconstruct_sequence(pairs, working_W, working_H, rcfg):
    """pairs: dict from track_keyframes (or an equivalent anchor source).
    rcfg: the `reconstruct:` config block. Returns raw (M, cres) per frame
    BEFORE temporal smoothing -- worth checkpointing on its own, since Stage B's
    per-frame reconstruction (not the smoothing) is the slow part."""
    trO, trC, vi = pairs["trO"], pairs["trC"], pairs["vi"]
    TW, TH = pairs["track_res"]
    SX, SY = working_W / TW, working_H / TH
    T = trO.shape[0]
    t0 = time.time()
    Ms, creses = [], []
    for fr in range(T):
        o = (trO[fr][vi[fr]] * [SX, SY]).astype(np.float32)
        c = (trC[fr][vi[fr]] * [SX, SY]).astype(np.float32)
        M, _ = df.fit_affine(o, c, rcfg["affine_thresh"])
        hp, hc = _hole_pin_anchors(o, M, working_W, working_H,
                                   rcfg["hole_pin_stride"], rcfg["hole_pin_margin"])
        if hp is not None:
            o = np.vstack([o, hp]); c = np.vstack([c, hc])
        M, cres, _ = df.reconstruct_decomposed(
            o, c, working_H, working_W, M, grid=tuple(rcfg["grid"]), neighbors=rcfg["neighbors"],
            smoothing=rcfg["smoothing"], max_anchors=rcfg["max_anchors"], boundary=tuple(rcfg["boundary"]))
        if cres is None:
            cres = np.zeros((rcfg["grid"][1], rcfg["grid"][0], 2), np.float32)
            print(f"f{fr}: degenerate reconstruction", flush=True)
        Ms.append(M); creses.append(cres)
        if fr % 10 == 0:
            print(f"recon f{fr} ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(Ms), np.stack(creses)


def smooth_temporal(Ms, creses, working_W, working_H, rcfg,
                    grab_orig=None, grab_clean=None, mask_loader=None):
    """Joint M+cres savgol (M and cres MUST be smoothed together -- smoothing
    one alone breaks per-frame consistency and re-introduces wobble), then a
    spatial low-pass on the residual lattice (kills per-track frozen-bias
    "noodling" without touching the real parallax signal, which lives at a
    much coarser spatial scale). `cres_blur_sigma` may be a single value or a
    list to auto-sweep against `sigma_check_frames` using the gated grad-NCC
    lock -- requires grab_orig/grab_clean/mask_loader callables (fr -> image
    or mask) when sweeping more than one sigma."""
    Mt = Ms
    for w in rcfg["temporal_windows"]:
        Mt = savgol_filter(Mt, w, 2, axis=0)

    sigmas = rcfg["cres_blur_sigma"]
    if not isinstance(sigmas, (list, tuple)):
        sigmas = [sigmas]

    def blur_and_smooth(sig):
        ct = gaussian_filter(creses, sigma=(0, sig, sig, 0))
        for w in rcfg["temporal_windows"]:
            ct = savgol_filter(ct, w, 2, axis=0)
        return ct

    if len(sigmas) == 1 or grab_orig is None:
        return Mt, blur_and_smooth(sigmas[0])

    check_frames = rcfg["sigma_check_frames"]
    imgs = {fr: (grab_orig(fr), grab_clean(fr), mask_loader(fr)) for fr in check_frames}
    best = None
    for sig in sigmas:
        ct = blur_and_smooth(sig)
        score = 0.0
        for fr in check_frames:
            O, C, mask = imgs[fr]
            mx, my = df.assemble_map(Mt[fr], ct[fr], working_H, working_W)
            wa = g.apply_backward_map(C, mx, my)
            score += dg.grad_ncc_gated(O, wa, exclude=mask)
        print(f"sigma {sig}: lock-sum {score:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, sig, ct)
    _, sig, ct = best
    print(f"chosen cres_blur_sigma={sig}", flush=True)
    return Mt, ct

"""
platelock.tracking.track — Stage A: same-domain multi-keyframe tracking.

Seeds a grid at several ORIGINAL-space keyframes (mask-filtered), bootstrap-
transfers seed positions to CLEAN-space via an already-solved backward map
(no new cross-domain matching), tracks both plates with CoTracker3 offline,
merges all keyframes' anchor pairs, then screens out what isn't usable:

  wiggle screen         -> drop chaotically-noisy tracks outright
  gap bridging (PCHIP)  -> fill short occlusion gaps from the track's own samples
  per-frame liar screen -> drop anchor INSTANCES whose residual disagrees with
                           their local neighbourhood (smoothly-wrong correspondences
                           that a track-level filter would miss -- the make-or-
                           break stage; found empirically after three other fixes
                           for the same symptom failed)
  per-track savgol      -> smooth surviving contiguous runs

Returns per-frame (orig_xy, clean_xy) anchor pairs ready for
platelock.tracking.reconstruct.
"""
import time

import cv2
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from platelock import dense_field as df
from platelock import masks as pm


def _load_video(path, W, H):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (W, H)))
    cap.release()
    return np.stack(frames)


def _load_masks(mask_dir, T, side):
    return np.stack([pm.load_mask(f"{mask_dir}/{side}_mask_f{fr:04d}.png") for fr in range(T)])


def _keyframes(T, stride):
    keys = list(range(0, T, stride))
    if keys[-1] != T - 1:
        keys.append(T - 1)
    return keys


def _runs(mask):
    out, s = [], None
    for t in range(len(mask)):
        if mask[t] and s is None:
            s = t
        if (not mask[t] or t == len(mask) - 1) and s is not None:
            out.append((s, t + 1 if mask[t] else t))
            s = None
    return out


def track_keyframes(cfg, working_W, working_H, T, device="cuda"):
    """Stage A driver. `cfg` is the full pipeline config (uses `inputs`,
    `masks`, `bootstrap`, `tracking` blocks). Returns a dict with trO, trC, vi
    (bool), trO_raw, trC_raw, keys, track_res -- ready to np.savez as a
    checkpoint and to feed platelock.tracking.reconstruct.reconstruct_sequence."""
    inp_orig, inp_clean = cfg["inputs"]["original"], cfg["inputs"]["clean"]
    tcfg = cfg["tracking"]
    mask_dir = cfg["masks"]["dir"]
    M_boot = np.load(cfg["bootstrap"]["M"])
    cres_boot = np.load(cfg["bootstrap"]["cres"])
    keys = _keyframes(T, tcfg["keyframe_stride"])
    gx_n, gy_n = tcfg["seed_grid"]
    # opt-in border boost: the uniform grid + short-run screens structurally
    # starve edge/corner coverage (late-revealed content under the pan lives
    # < min_run frames and gets wig=9e9 outright). No change when absent.
    bcfg = tcfg.get("border_boost", {})
    boost_on = bcfg.get("enabled", False)
    bband = bcfg.get("band_px", 100)

    maskO = _load_masks(mask_dir, T, "orig")
    maskC = _load_masks(mask_dir, T, "clean")

    model = None
    t0 = time.time()
    TW = TH = SX = SY = None
    for TW, TH in tcfg["resolutions"]:
        torch.cuda.empty_cache()
        try:
            vidO = _load_video(inp_orig, TW, TH)
            vidC = _load_video(inp_clean, TW, TH)
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM loading at {TW}x{TH}, falling back", flush=True)
            model = None
            continue

        def track(vid, queries):
            v = torch.from_numpy(vid[..., ::-1].copy()).permute(0, 3, 1, 2)[None].float().to(device)
            q = torch.from_numpy(queries)[None].float().to(device)
            with torch.no_grad():
                tr, vis = model(v, queries=q, backward_tracking=True)
            del v
            torch.cuda.empty_cache()
            return tr[0].cpu().numpy(), vis[0].cpu().numpy() > 0.5

        SX, SY = working_W / TW, working_H / TH
        allO, allC, allviO, allviC = [], [], [], []
        try:
            for k in keys:
                gx, gy = np.meshgrid(np.linspace(8, TW - 9, gx_n), np.linspace(8, TH - 9, gy_n))
                seeds = np.stack([gx.ravel(), gy.ravel()], 1)
                mx, my = df.assemble_map(M_boot[k], cres_boot[k], working_H, working_W)

                def cull_and_pair(sds):
                    fx = (sds[:, 0] * SX).astype(int)
                    fy = (sds[:, 1] * SY).astype(int)
                    keep = ~maskO[k][fy, fx]
                    sds, fx, fy = sds[keep], fx[keep], fy[keep]
                    cs = np.stack([mx[fy, fx] / SX, my[fy, fx] / SY], 1)
                    inb = ((cs[:, 0] > 4) & (cs[:, 0] < TW - 5)
                           & (cs[:, 1] > 4) & (cs[:, 1] < TH - 5))
                    ccx = np.clip(cs[:, 0] * SX, 0, working_W - 1).astype(int)
                    ccy = np.clip(cs[:, 1] * SY, 0, working_H - 1).astype(int)
                    inb &= ~maskC[k][ccy, ccx]
                    return sds[inb], cs[inb]

                seeds, cseeds = cull_and_pair(seeds)
                trO_, viO_ = track(vidO, np.concatenate([np.full((len(seeds), 1), k), seeds], 1))
                trC_, viC_ = track(vidC, np.concatenate([np.full((len(cseeds), 1), k), cseeds], 1))
                allO.append(trO_); allC.append(trC_); allviO.append(viO_); allviC.append(viC_)
                print(f"K={k}: {len(seeds)} pairs ({time.time()-t0:.0f}s)", flush=True)
                if boost_on:
                    # border seeds tracked in a SEPARATE CoTracker pass: keeps the
                    # base pass inputs bitwise identical to the un-boosted run
                    # (purely additive experiment) and halves peak GPU memory.
                    m = bcfg.get("density_mult", 2)
                    gxd, gyd = np.meshgrid(np.linspace(8, TW - 9, gx_n * m),
                                           np.linspace(8, TH - 9, gy_n * m))
                    dense = np.stack([gxd.ravel(), gyd.ravel()], 1)
                    inband = ((dense[:, 0] < 8 + bband) | (dense[:, 0] > TW - 9 - bband)
                              | (dense[:, 1] < 8 + bband) | (dense[:, 1] > TH - 9 - bband))
                    bsds = dense[inband]
                    bsds, bcs = cull_and_pair(bsds)
                    if len(bsds):
                        btrO_, bviO_ = track(vidO, np.concatenate([np.full((len(bsds), 1), k), bsds], 1))
                        btrC_, bviC_ = track(vidC, np.concatenate([np.full((len(bcs), 1), k), bcs], 1))
                        allO.append(btrO_); allC.append(btrC_); allviO.append(bviO_); allviC.append(bviC_)
                        print(f"[border boost] K={k}: +{len(bsds)} border pairs", flush=True)
            print(f"TRACKING OK at {TW}x{TH}", flush=True)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"OOM mid-tracking at {TW}x{TH}, falling back", flush=True)
            del model
            model = None
            continue
    if model is None:
        raise RuntimeError("CoTracker ran out of memory at every resolution in tracking.resolutions")

    trO = np.concatenate(allO, 1); trC = np.concatenate(allC, 1)
    viO = np.concatenate(allviO, 1); viC = np.concatenate(allviC, 1)
    N = trO.shape[1]
    inb = ((trO[..., 0] >= 0) & (trO[..., 0] < TW) & (trO[..., 1] >= 0) & (trO[..., 1] < TH)
           & (trC[..., 0] >= 0) & (trC[..., 0] < TW) & (trC[..., 1] >= 0) & (trC[..., 1] < TH))
    vi = inb & viO & viC
    print(f"merged {N} tracks; active/frame min {vi.sum(1).min()} med {int(np.median(vi.sum(1)))}", flush=True)

    # border tracks (median valid position in the band) get screen leniency
    # below when border boost is on: the liar screen still gates them per-frame.
    border = np.zeros(N, bool)
    if boost_on:
        for i in np.where(vi.any(0))[0]:
            mx_, my_ = np.median(trO[vi[:, i], i], 0)
            border[i] = (mx_ < 8 + bband or mx_ > TW - 9 - bband
                         or my_ < 8 + bband or my_ > TH - 9 - bband)

    # wiggle screen: drop chaotically-noisy tracks outright (2nd-derivative accel outlier)
    acc = np.linalg.norm(trC[2:] - 2 * trC[1:-1] + trC[:-2], axis=2)
    okk = vi[2:] & vi[1:-1] & vi[:-2]
    wig = np.array([np.median(acc[okk[:, i], i]) if okk[:, i].sum() > 4 else 9e9 for i in range(N)])
    if boost_on:
        # short border tracks (too few accel samples for a wiggle verdict) get
        # a neutral score instead of auto-cull; the liar screen still applies
        exempt = border & (wig >= 9e9) & vi.any(0)
        if (wig < 9e9).any():
            wig[exempt] = np.median(wig[wig < 9e9])
        print(f"[border boost] wiggle-exempted {int(exempt.sum())} border tracks", flush=True)
    good = wig < np.median(wig[wig < 9e9]) * tcfg["wiggle_mult"]
    trO, trC, vi = trO[:, good], trC[:, good], vi[:, good]
    border = border[good]
    N = trO.shape[1]
    print(f"wiggle screen: keep {N}", flush=True)

    # PCHIP gap bridging: fill short occlusion gaps from the track's own visible samples
    nbridged = 0
    gap_max = tcfg["gap_max"]
    for i in range(N):
        rs = _runs(vi[:, i])
        if len(rs) < 2:
            continue
        visidx = np.where(vi[:, i])[0]
        if len(visidx) < 8:
            continue
        fO = PchipInterpolator(visidx, trO[visidx, i], axis=0)
        fC = PchipInterpolator(visidx, trC[visidx, i], axis=0)
        for (s1, e1), (s2, e2) in zip(rs[:-1], rs[1:]):
            gap = np.arange(e1, s2)
            if len(gap) == 0 or len(gap) >= gap_max:
                continue
            if (e1 - s1) < 5 or (e2 - s2) < 5:
                continue
            trO[gap, i] = fO(gap); trC[gap, i] = fC(gap); vi[gap, i] = True
            nbridged += len(gap)
    print(f"bridged {nbridged}", flush=True)

    # per-frame iterative local-consistency screen: catches correspondences that
    # are wrong for only PART of their lifetime, invisible to any track-level threshold
    lcfg = tcfg["liar_screen"]
    t0 = time.time()
    nexcl = 0
    for fr in range(T):
        for _ in range(lcfg["iterations"]):
            ids = np.where(vi[fr])[0]
            if len(ids) < lcfg["min_active"]:
                break
            o = (trO[fr][ids] * [SX, SY]).astype(np.float32)
            c = (trC[fr][ids] * [SX, SY]).astype(np.float32)
            M, _ = df.fit_affine(o, c, 3.0)
            if M is None:
                break
            resid = c - ((o @ M[:, :2].T) + M[:, 2])
            _, nn = cKDTree(o).query(o, k=min(lcfg["neighbors"], len(o)))
            nmed = np.median(resid[nn[:, 1:]], axis=1)
            bad = np.linalg.norm(resid - nmed, axis=1) > lcfg["threshold_px"]
            if not bad.any():
                break
            vi[fr, ids[bad]] = False
            nexcl += int(bad.sum())
    print(f"per-frame liar screen: excluded {nexcl} ({time.time()-t0:.0f}s)", flush=True)

    # per-track savgol over surviving contiguous runs
    trO_raw, trC_raw = trO.copy(), trC.copy()
    vi2 = np.zeros_like(vi)
    min_run, twin = tcfg["min_run"], tcfg["temporal_window"]
    mrb = bcfg.get("min_run_border", 5) if boost_on else min_run
    nshort = 0
    for i in range(N):
        mr = mrb if border[i] else min_run
        for s, e in _runs(vi[:, i]):
            if e - s < mr:
                continue
            if e - s < min_run:
                nshort += 1               # kept only by the border leniency
            if e - s < 5:                 # savgol poly2 needs odd window >= 5
                vi2[s:e, i] = True
                continue
            wl = min(twin, (e - s) if (e - s) % 2 == 1 else (e - s) - 1)
            trO[s:e, i] = savgol_filter(trO[s:e, i], wl, 2, axis=0)
            trC[s:e, i] = savgol_filter(trC[s:e, i], wl, 2, axis=0)
            vi2[s:e, i] = True
    if boost_on:
        print(f"[border boost] kept {nshort} short border runs (< min_run)", flush=True)
    print(f"post-screen anchors/frame: min {vi2.sum(1).min()} med {int(np.median(vi2.sum(1)))}", flush=True)

    return dict(trO=trO, trC=trC, trO_raw=trO_raw, trC_raw=trC_raw, vi=vi2,
                keys=np.array(keys), track_res=np.array([TW, TH]))

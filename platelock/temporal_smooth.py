"""
platelock.temporal_smooth — temporal smoothing sweep on the crux cluster.

Chat's step 3: represent each frame's warp as (global affine M, control-grid
residual cres) via dense_field.reconstruct_decomposed, then smooth those
PARAMETERS across time with a Savitzky-Golay filter (NOT pixels). Reject bad
frames before smoothing (a collapsed frame is not signal) by replacing their
M/cres with a linear interpolation of the nearest good neighbours.

Runs the sweep over CONTIGUOUS frames 110..120 (11 frames) so savgol windows
5/7/9 are all valid and the time axis is uniformly sampled.

Metrics per sweep (unsmoothed + each window):
  mean_lock  — mean left_window gradient-NCC (higher = better lock)
  jitter     — mean |2nd temporal difference| of the backward map over the ROI
               (lower = less swim/flicker)
  mean_fold  — mean fold fraction (must not rise vs unsmoothed)

Pass condition (Chat): lock equal-or-better, folds not higher, jitter lower,
no visible lag (smoothing should be LIGHT if unsmoothed is already stable — do
not smear away good local correction).

Usage:
    python -m platelock.temporal_smooth \
        --frames 110-120 --windows 5,7,9 --out output/temporal_sweep_110_120
"""

import os
import json
import argparse

import cv2
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

from platelock import geometry as g, diagnostics as dg, dense_field as df
from platelock.run_dense_cluster import load_config, build_model, grab


def collect(cfg, frames, device="cuda"):
    """Per frame -> decomposed warp + health. Returns list of dicts keeping the
    affine M, control-grid residual cres, and the crops needed to re-score."""
    model = build_model(cfg, device)
    inp = cfg["inputs"]
    width = cfg["working"]["width"]
    box = cfg["crux"]["rois"]["left_window"]
    a, rb, rc = cfg["anchors"], cfg["anchors"]["roi_balanced"], cfg["reconstruct"]
    recs = []
    for fr in frames:
        O = grab(inp["original"], fr, width)
        C = grab(inp["clean"], fr, width)
        H, W = O.shape[:2]
        oi = Image.fromarray(cv2.cvtColor(O, cv2.COLOR_BGR2RGB))
        ci = Image.fromarray(cv2.cvtColor(C, cv2.COLOR_BGR2RGB))
        warp, cert = model.match(oi, ci, device=device)
        orig, clean, c = df.extract_anchors(warp, cert, H, W, model)
        o0, cl0, c0, _ = df.filter_anchors(orig, clean, c, thresh=a["cert_thresh"])
        o, cl, cc = df.roi_balanced_topk(o0, cl0, c0, H, W, tiles=tuple(rb["tiles"]),
                                         frac=rb["frac"], max_total=rb["max_total"])
        M, _ = df.fit_affine(o, cl, thresh=cfg["affine"]["ransac_thresh"])
        M, cres, grid = df.reconstruct_decomposed(
            o, cl, H, W, M, grid=tuple(rc["grid"]), neighbors=rc["neighbors"],
            smoothing=rc["smoothing"], max_anchors=rc["max_anchors"],
            boundary=tuple(rc["boundary"]))
        mx, my = df.assemble_map(M, cres, H, W)
        fh = df.field_health(mx, my)
        x, y, w, h = box
        lock = dg.grad_ncc(O[y:y + h, x:x + w],
                           g.apply_backward_map(C, mx, my)[y:y + h, x:x + w])
        recs.append(dict(fr=fr, M=M, cres=cres, grid=grid, HW=(H, W),
                         anchors=int(len(o)), fold=fh["fold_frac"], lock=float(lock),
                         O=O, C=C, degenerate=(M is None or cres is None)))
        print(f"  f{fr}: anchors {len(o)} fold {fh['fold_frac']:.4f} lock {lock:+.3f}"
              f"{'  [DEGENERATE]' if recs[-1]['degenerate'] else ''}", flush=True)
    return recs, box


def reject_before_smooth(recs, min_anchor=250, max_fold=0.05):
    """Mark degenerate / low-anchor / high-fold frames as BAD and replace their
    M and cres by linear interpolation of the nearest good neighbours (in frame
    order), so smoothing sees signal, not a collapsed frame. Returns bad mask.
    Absolute thresholds (Chat's step-5 spec): anchors<250 or fold>5% => bad."""
    n = len(recs)
    bad = np.array([bool(r["degenerate"]) or r["anchors"] < min_anchor
                    or r["fold"] > max_fold for r in recs])
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return bad  # nothing to lean on; leave as-is
    for i in np.where(bad)[0]:
        left = good_idx[good_idx < i]
        right = good_idx[good_idx > i]
        if len(left) and len(right):
            lo, hi = left[-1], right[0]
            t = (i - lo) / (hi - lo)
            recs[i]["M"] = (1 - t) * recs[lo]["M"] + t * recs[hi]["M"]
            recs[i]["cres"] = (1 - t) * recs[lo]["cres"] + t * recs[hi]["cres"]
        else:                                   # end frame -> copy nearest good
            j = (left[-1] if len(left) else right[0])
            recs[i]["M"] = recs[j]["M"].copy()
            recs[i]["cres"] = recs[j]["cres"].copy()
    return bad


def temporal_jitter(maps, box):
    """mean |2nd temporal difference| of the backward map over the ROI."""
    x, y, w, h = box
    sx = np.stack([mx[y:y + h, x:x + w] for mx, my in maps])
    sy = np.stack([my[y:y + h, x:x + w] for mx, my in maps])
    d2x = sx[2:] - 2 * sx[1:-1] + sx[:-2]
    d2y = sy[2:] - 2 * sy[1:-1] + sy[:-2]
    return float(np.mean(np.abs(d2x)) + np.mean(np.abs(d2y)))


def evaluate(recs, box, Ms, creses):
    """Assemble maps from (possibly smoothed) params and score lock/fold/jitter."""
    x, y, w, h = box
    maps, locks, folds = [], [], []
    for i, r in enumerate(recs):
        H, W = r["HW"]
        mx, my = df.assemble_map(Ms[i], creses[i], H, W)
        maps.append((mx, my))
        locks.append(dg.grad_ncc(r["O"][y:y + h, x:x + w],
                                 g.apply_backward_map(r["C"], mx, my)[y:y + h, x:x + w]))
        folds.append(df.field_health(mx, my)["fold_frac"])
    return dict(mean_lock=round(float(np.mean(locks)), 4),
                mean_fold=round(float(np.mean(folds)), 5),
                jitter=round(temporal_jitter(maps, box), 4),
                per_frame_lock=[round(float(v), 3) for v in locks])


def sweep(recs, box, windows):
    Ms = np.stack([r["M"] for r in recs])            # (T,2,3)
    creses = np.stack([r["cres"] for r in recs])     # (T,gy,gx,2)
    out = {"unsmoothed": evaluate(recs, box, Ms, creses)}
    for wlen in windows:
        if wlen > len(recs) or wlen % 2 == 0:
            out[f"savgol_w{wlen}"] = {"skipped": f"invalid window for {len(recs)} frames"}
            continue
        po = min(2, wlen - 1)
        Ms_s = savgol_filter(Ms, wlen, po, axis=0)
        cr_s = savgol_filter(creses, wlen, po, axis=0)
        out[f"savgol_w{wlen}"] = evaluate(recs, box, Ms_s, cr_s)
    return out


def render_compare(recs, box, out_dir, window, device="cuda"):
    """Visual sign-off: per-frame CLAHE left_window overlay comparing RAW vs
    DENSE-unsmoothed vs DENSE-savgol_w{window}, so the steadier field can be
    checked for genuine lock (not lag) by eye. Writes jpgs + a compare video."""
    import glob
    x, y, w, h = box
    Ms = np.stack([r["M"] for r in recs])
    creses = np.stack([r["cres"] for r in recs])
    po = min(2, window - 1)
    Ms_s = savgol_filter(Ms, window, po, axis=0)
    cr_s = savgol_filter(creses, window, po, axis=0)
    for i, r in enumerate(recs):
        H, W = r["HW"]
        oc = r["O"][y:y + h, x:x + w]
        mxu, myu = df.assemble_map(r["M"], r["cres"], H, W)
        mxs, mys = df.assemble_map(Ms_s[i], cr_s[i], H, W)
        wu = g.apply_backward_map(r["C"], mxu, myu)[y:y + h, x:x + w]
        ws = g.apply_backward_map(r["C"], mxs, mys)[y:y + h, x:x + w]
        cc = r["C"][y:y + h, x:x + w]
        panels = [
            dg.label(dg.edge_overlay(oc, cc), f"f{r['fr']} RAW ncc={dg.grad_ncc(oc, cc):+.2f}"),
            dg.label(dg.edge_overlay(oc, wu), f"DENSE unsm ncc={dg.grad_ncc(oc, wu):+.2f}"),
            dg.label(dg.edge_overlay(oc, ws), f"DENSE w{window} ncc={dg.grad_ncc(oc, ws):+.2f}")]
        cv2.imwrite(f"{out_dir}/cmp_f{r['fr']:03d}_w{window}.jpg", dg.hstack_pad(panels, pad=6))
    files = sorted(glob.glob(f"{out_dir}/cmp_f*_w{window}.jpg"))
    if files:
        im = cv2.imread(files[0]); hh, ww = im.shape[:2]
        vw = cv2.VideoWriter(f"{out_dir}/cmp_w{window}.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), 4, (ww, hh))
        for f in files:
            vw.write(cv2.resize(cv2.imread(f), (ww, hh)))
        vw.release()
    print(f"WROTE compare overlays + cmp_w{window}.mp4 to {out_dir}")


def parse_frames(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Temporal smoothing sweep (crux).")
    ap.add_argument("--config", default="platelock/configs/dense_crux.yaml")
    ap.add_argument("--out", default="output/temporal_sweep_110_120")
    ap.add_argument("--frames", default="110-120")
    ap.add_argument("--windows", default="5,7,9")
    ap.add_argument("--render-window", type=int, default=0,
                    help="if set, render RAW/unsmoothed/w{N} compare overlays")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frames = parse_frames(args.frames)
    windows = [int(x) for x in args.windows.split(",")]
    os.makedirs(args.out, exist_ok=True)

    print(f"collecting decomposed warps for frames {frames[0]}..{frames[-1]} ...")
    recs, box = collect(cfg, frames, args.device)
    bad = reject_before_smooth(recs)
    bad_frames = [recs[i]["fr"] for i in np.where(bad)[0]]
    print(f"reject-before-smooth flagged: {bad_frames or 'none'}")

    result = sweep(recs, box, windows)
    result["_meta"] = {"frames": frames, "windows": windows,
                       "bad_frames": bad_frames, "metric": "gradient_ncc_left_window"}
    json.dump(result, open(f"{args.out}/sweep.json", "w"), indent=2)

    print("\n=== temporal smoothing sweep (left_window) ===")
    print(f"{'variant':>14} | {'mean_lock':>9} | {'jitter':>8} | {'mean_fold':>9}")
    for k in ["unsmoothed"] + [f"savgol_w{w}" for w in windows]:
        r = result.get(k, {})
        if "skipped" in r:
            print(f"{k:>14} | {r['skipped']}")
        else:
            print(f"{k:>14} | {r['mean_lock']:>+9.4f} | {r['jitter']:>8.3f} | {r['mean_fold']:>9.5f}")
    print(f"\nWROTE {args.out}/sweep.json")

    if args.render_window:
        render_compare(recs, box, args.out, args.render_window, args.device)


if __name__ == "__main__":
    main()

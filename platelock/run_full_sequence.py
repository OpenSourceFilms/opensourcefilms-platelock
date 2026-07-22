"""
platelock.run_full_sequence — full 0-125 dense plate-lock at working resolution.

Chat's step 4 (+ step-5 failure handling), built after the crux cluster survived
temporal smoothing (w7). TWO PASS so 126 full frames never sit in RAM at once:

  PASS 1 (RoMa, expensive): per frame -> decomposed warp (affine M, control-grid
     residual cres) + health (anchor_count, fold). Store ONLY those compact params.
  reject-before-smooth (step 5): frames with anchors<250 / fold>5% / degenerate get
     their M,cres replaced by linear interp of nearest good neighbours (a collapsed
     frame is never fed to the smoother as signal). Softer warn tier: <500 / >2%.
  temporal smoothing: savgol over time on the affine params + residual grids (config
     window, default 7). Never smooths pixels. BOTH unsmoothed and smoothed params
     are saved, so w5/w9/unsmoothed can be re-tested WITHOUT rerunning RoMa.

  PASS 2 (re-grab frames, NO RoMa): assemble_map from smoothed params -> backward
     remap. Emits overlay_50 / wipe / checkerboard / displacement / crux crop videos,
     saves each backward map (map_x/map_y .npy) for the later 5K upscale + EXR step,
     and writes gated-NCC (dense vs raw), anchor-count, fold%, and jitter curves.

Global metric = diagnostics.grad_ncc_gated (scores only where the ORIGINAL has
structure) — valid at any pan position, unlike fixed crux ROIs. CAVEAT (Chat):
this is a HEALTH SIGNAL, not sole sign-off — original-gradient gating can still
score actor edges / fire / non-target foreground. Trusted visual overlays remain
the verdict; actor/fire/generated exclusion masks are a later refinement.

Output tree (default output/dense_full_workingres/):
  params/  M_unsmoothed.npy cres_unsmoothed.npy M_savgol_w{N}.npy cres_savgol_w{N}.npy frame_health.json
  maps/    map_x_f0000.npy map_y_f0000.npy ...
  previews/ overlay_50.mp4 wipe_preview.mp4 checkerboard.mp4 disp_field.mp4 crux_left_window.mp4 crux_sconce_support.mp4
  metrics/ gated_ncc.csv anchor_count.csv fold_percent.csv jitter.csv rejected_frames.json
  plots/   gated_ncc.png anchor_count.png fold_percent.png jitter.png

Usage:
    python -m platelock.run_full_sequence --frames 0-125 --out output/dense_full_workingres
"""

import os
import csv
import json
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter

from platelock import geometry as g, diagnostics as dg, dense_field as df, masks as mk
from platelock.run_dense_cluster import load_config, build_model, build_segmenter, frame_masks, grab
from platelock.temporal_smooth import reject_before_smooth, parse_frames

CRUX = (110, 120)


def pass1_collect(cfg, frames, model, device, out=None):
    """RoMa + anchor pipeline per frame -> compact decomposed params only.
    If cfg["masks"]["enabled"], builds the SAM3 segmenter ONCE, applies
    mk.exclude_anchors before certainty filtering, and (if `out` given) caches
    orig_mask/clean_mask PNGs to {out}/masks/ so pass2_render doesn't need to
    rerun SAM3 for the metric-exclusion mask."""
    inp = cfg["inputs"]; width = cfg["working"]["width"]
    a, rb, rc = cfg["anchors"], cfg["anchors"]["roi_balanced"], cfg["reconstruct"]
    segmenter = build_segmenter(cfg, device)
    if segmenter is not None and out is not None:
        os.makedirs(f"{out}/masks", exist_ok=True)
    recs = []
    for fr in frames:
        O = grab(inp["original"], fr, width)
        C = grab(inp["clean"], fr, width)
        H, W = O.shape[:2]
        oi = Image.fromarray(cv2.cvtColor(O, cv2.COLOR_BGR2RGB))
        ci = Image.fromarray(cv2.cvtColor(C, cv2.COLOR_BGR2RGB))
        warp, cert = model.match(oi, ci, device=device)
        orig, clean, c = df.extract_anchors(warp, cert, H, W, model)
        if segmenter is not None:
            orig_mask, clean_mask = frame_masks(O, C, segmenter, cfg)
            orig, clean, c, _ = mk.exclude_anchors(orig, clean, c,
                                                    orig_mask=orig_mask, clean_mask=clean_mask)
            if out is not None:
                mk.save_mask(f"{out}/masks/orig_mask_f{fr:04d}.png", orig_mask)
                mk.save_mask(f"{out}/masks/clean_mask_f{fr:04d}.png", clean_mask)
        o0, cl0, c0, _ = df.filter_anchors(orig, clean, c, thresh=a["cert_thresh"])
        o, cl, cc = df.roi_balanced_topk(o0, cl0, c0, H, W, tiles=tuple(rb["tiles"]),
                                         frac=rb["frac"], max_total=rb["max_total"])
        M, _ = df.fit_affine(o, cl, thresh=cfg["affine"]["ransac_thresh"])
        M, cres, grid = df.reconstruct_decomposed(
            o, cl, H, W, M, grid=tuple(rc["grid"]), neighbors=rc["neighbors"],
            smoothing=rc["smoothing"], max_anchors=rc["max_anchors"],
            boundary=tuple(rc["boundary"]))
        mx, my = df.assemble_map(M, cres, H, W)
        fold = df.field_health(mx, my)["fold_frac"]
        recs.append(dict(fr=fr, M=M, cres=cres, grid=grid, HW=(H, W),
                         anchors=int(len(o)), fold=float(fold),
                         degenerate=(M is None or cres is None)))
        print(f"  pass1 f{fr}: anchors {len(o)} fold {fold*100:.2f}%"
              f"{'  [DEGEN]' if recs[-1]['degenerate'] else ''}", flush=True)
    return recs


def checkerboard(a, b, tile=140):
    out = a.copy()
    H, W = a.shape[:2]
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            if ((x // tile) + (y // tile)) % 2:
                out[y:y + tile, x:x + tile] = b[y:y + tile, x:x + tile]
    return out


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def plot_curve(x, series, ylabel, path, hlines=()):
    fig, ax = plt.subplots(figsize=(12, 3.4))
    for label, y, style in series:
        ax.plot(x, y, style, ms=2, label=label)
    for yv, txt in hlines:
        ax.axhline(yv, color="k", lw=0.5, ls="--")
    ax.axvspan(CRUX[0], CRUX[1], color="orange", alpha=0.15, label="crux 110-120")
    ax.set_xlabel("frame"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def pass2_render(cfg, recs, Ms, creses, out, save_maps, device):
    """exclude= for grad_ncc_gated is the ORIGINAL-space mask only (actor + real
    dim sconce). clean_mask is never used here — it's clean-space anchor exclusion,
    not original-space metric exclusion (would need explicit projection through the
    map to mean anything in O's coordinate frame, which we don't do)."""
    inp = cfg["inputs"]; width = cfg["working"]["width"]
    rois = cfg["crux"]["rois"]
    H, W = recs[0]["HW"]
    masks_enabled = bool(cfg.get("masks", {}).get("enabled"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ov = cv2.VideoWriter(f"{out}/previews/overlay_50.mp4", fourcc, 24, (W, H))
    wp = cv2.VideoWriter(f"{out}/previews/wipe_preview.mp4", fourcc, 24, (W, H))
    cb = cv2.VideoWriter(f"{out}/previews/checkerboard.mp4", fourcc, 24, (W, H))
    dv = cv2.VideoWriter(f"{out}/previews/disp_field.mp4", fourcc, 24, (W, H))
    crux_vw = {}
    for n in ("left_window", "sconce_support", "middle_window"):
        if n not in rois:
            continue
        x, y, w, h = rois[n]
        crux_vw[n] = (cv2.VideoWriter(f"{out}/previews/crux_{n}.mp4", fourcc, 8,
                                      (w * 3 + 12, h)), (x, y, w, h))

    frames = [r["fr"] for r in recs]
    lock_d, lock_r, jitter = [], [], [float("nan")] * len(recs)
    mbuf = []  # rolling buffer of last 3 (mx,my) for temporal 2nd-derivative
    for i, r in enumerate(recs):
        fr = r["fr"]
        O = grab(inp["original"], fr, width)
        C = grab(inp["clean"], fr, width)
        mx, my = df.assemble_map(Ms[i], creses[i], H, W)
        warped = g.apply_backward_map(C, mx, my)

        exclude = None
        if masks_enabled:
            mask_path = f"{out}/masks/orig_mask_f{fr:04d}.png"
            if os.path.exists(mask_path):
                exclude = mk.load_mask(mask_path)

        lock_d.append(dg.grad_ncc_gated(O, warped, exclude=exclude))
        lock_r.append(dg.grad_ncc_gated(O, C, exclude=exclude))

        ov.write(cv2.addWeighted(O, 0.5, warped, 0.5, 0))
        wipe = O.copy(); wipe[:, W // 2:] = warped[:, W // 2:]
        cv2.line(wipe, (W // 2, 0), (W // 2, H), (0, 255, 255), 1); wp.write(wipe)
        cb.write(checkerboard(O, warped))
        dv.write(df.displacement_preview(mx, my, H, W, step=44))

        for n, (vw, (x, y, wd, ht)) in crux_vw.items():
            oc = O[y:y + ht, x:x + wd]
            panel = dg.hstack_pad([
                dg.label(dg.edge_overlay(oc, C[y:y + ht, x:x + wd]), f"f{fr} RAW"),
                dg.label(dg.edge_overlay(oc, warped[y:y + ht, x:x + wd]), "DENSE"),
                dg.label(oc, "orig")], pad=6)
            vw.write(cv2.resize(panel, (wd * 3 + 12, ht)))

        mbuf.append((mx.copy(), my.copy()))
        if len(mbuf) == 3:
            (ax, ay), (bx, by), (cx2, cy2) = mbuf
            jitter[i - 1] = float(np.mean(np.abs(cx2 - 2 * bx + ax))
                                  + np.mean(np.abs(cy2 - 2 * by + ay)))
            mbuf.pop(0)

        if save_maps:
            np.save(f"{out}/maps/map_x_f{fr:04d}.npy", mx)
            np.save(f"{out}/maps/map_y_f{fr:04d}.npy", my)
        if fr % 15 == 0:
            print(f"  pass2 f{fr}: dense {lock_d[-1]:+.3f} raw {lock_r[-1]:+.3f}"
                  f" fold {r['fold']*100:.2f}%", flush=True)

    for vw in (ov, wp, cb, dv, *[v for v, _ in crux_vw.values()]):
        vw.release()
    return dict(frame=frames, lock_dense=lock_d, lock_raw=lock_r,
                jitter=jitter, fold=[r["fold"] * 100 for r in recs],
                anchors=[r["anchors"] for r in recs])


def main():
    ap = argparse.ArgumentParser(description="Full 0-125 dense plate-lock (working res).")
    ap.add_argument("--config", default="platelock/configs/dense_crux.yaml")
    ap.add_argument("--out", default="output/dense_full_workingres")
    ap.add_argument("--frames", default="0-125")
    ap.add_argument("--skip-maps", action="store_true", help="don't save per-frame maps (~2GB)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frames = parse_frames(args.frames)
    for sub in ("params", "maps", "previews", "metrics", "plots", "masks"):
        os.makedirs(f"{args.out}/{sub}", exist_ok=True)
    model = build_model(cfg, args.device)

    print(f"PASS 1: RoMa + decompose, frames {frames[0]}..{frames[-1]}")
    recs = pass1_collect(cfg, frames, model, args.device, out=args.out)

    # frame health BEFORE repair, + save unsmoothed params for later re-testing
    health = {r["fr"]: {"anchors": r["anchors"], "fold_pct": round(r["fold"] * 100, 3),
                        "degenerate": bool(r["degenerate"])} for r in recs}
    json.dump(health, open(f"{args.out}/params/frame_health.json", "w"), indent=2)
    np.save(f"{args.out}/params/M_unsmoothed.npy", np.stack([r["M"] for r in recs]))
    np.save(f"{args.out}/params/cres_unsmoothed.npy", np.stack([r["cres"] for r in recs]))

    bad = reject_before_smooth(recs)
    bad_frames = [recs[i]["fr"] for i in np.where(bad)[0]]
    warn_frames = [r["fr"] for r in recs if r["anchors"] < 500 or r["fold"] > 0.02]
    json.dump({"bad_interpolated": bad_frames, "warn": warn_frames,
               "thresholds": {"bad": "anchors<250 or fold>5%",
                              "warn": "anchors<500 or fold>2%"}},
              open(f"{args.out}/metrics/rejected_frames.json", "w"), indent=2)
    print(f"reject-before-smooth: bad(interp)={bad_frames or 'none'} | warn={warn_frames or 'none'}")

    tcfg = cfg.get("temporal", {})
    window = tcfg.get("savgol_window", 7) if tcfg.get("enabled", True) else 0
    Ms = np.stack([r["M"] for r in recs]); creses = np.stack([r["cres"] for r in recs])
    wused = 0
    if window >= 3:
        n = len(recs); wused = min(window, n if n % 2 else n - 1)
        po = min(tcfg.get("savgol_polyorder", 2), wused - 1)
        Ms = savgol_filter(Ms, wused, po, axis=0)
        creses = savgol_filter(creses, wused, po, axis=0)
    np.save(f"{args.out}/params/M_savgol_w{wused}.npy", Ms)
    np.save(f"{args.out}/params/cres_savgol_w{wused}.npy", creses)

    print(f"PASS 2: temporal window={wused or 'OFF'}, render + score")
    cur = pass2_render(cfg, recs, Ms, creses, args.out, not args.skip_maps, args.device)

    # metrics CSVs
    save_csv(f"{args.out}/metrics/gated_ncc.csv", ["frame", "dense", "raw"],
             list(zip(cur["frame"], cur["lock_dense"], cur["lock_raw"])))
    save_csv(f"{args.out}/metrics/anchor_count.csv", ["frame", "anchors"],
             list(zip(cur["frame"], cur["anchors"])))
    save_csv(f"{args.out}/metrics/fold_percent.csv", ["frame", "fold_pct"],
             list(zip(cur["frame"], cur["fold"])))
    save_csv(f"{args.out}/metrics/jitter.csv", ["frame", "jitter"],
             list(zip(cur["frame"], cur["jitter"])))
    # plots
    plot_curve(cur["frame"], [("dense", cur["lock_dense"], "-o"),
                              ("raw", cur["lock_raw"], "-o")],
               "gated NCC lock", f"{args.out}/plots/gated_ncc.png", hlines=[(0, "0")])
    plot_curve(cur["frame"], [("anchors", cur["anchors"], "-o")],
               "anchor count", f"{args.out}/plots/anchor_count.png",
               hlines=[(500, "warn"), (250, "bad")])
    plot_curve(cur["frame"], [("fold %", cur["fold"], "-o")],
               "fold %", f"{args.out}/plots/fold_percent.png",
               hlines=[(2, "warn"), (5, "bad")])
    jj = [0 if (v != v) else v for v in cur["jitter"]]
    plot_curve(cur["frame"], [("jitter", jj, "-o")], "temporal 2nd-deriv (px)",
               f"{args.out}/plots/jitter.png")

    d = np.array(cur["lock_dense"]); rw = np.array(cur["lock_raw"])
    report = {"frames": [frames[0], frames[-1]], "temporal_window": wused,
              "bad_frames": bad_frames, "warn_frames": warn_frames,
              "metric": "full_frame_gated_ncc (HEALTH SIGNAL, not sole sign-off)",
              "dense_lock_mean": round(float(d.mean()), 4),
              "dense_lock_min": round(float(d.min()), 4),
              "dense_lock_min_frame": int(cur["frame"][int(d.argmin())]),
              "raw_lock_mean": round(float(rw.mean()), 4),
              "dense_beats_raw_frac": round(float((d > rw).mean()), 3),
              "fold_max_pct": round(float(np.max(cur["fold"])), 3),
              "fold_max_frame": int(cur["frame"][int(np.argmax(cur["fold"]))])}
    json.dump(report, open(f"{args.out}/report.json", "w"), indent=2)
    print(f"\ndense lock mean {report['dense_lock_mean']:+.3f} (raw {report['raw_lock_mean']:+.3f}); "
          f"dense>raw on {report['dense_beats_raw_frac']*100:.0f}% of frames; "
          f"min dense {report['dense_lock_min']:+.3f}@f{report['dense_lock_min_frame']}; "
          f"fold max {report['fold_max_pct']:.2f}%@f{report['fold_max_frame']}")
    print(f"WROTE {args.out}/  (previews/ metrics/ plots/ params/ maps/ report.json)")


if __name__ == "__main__":
    main()

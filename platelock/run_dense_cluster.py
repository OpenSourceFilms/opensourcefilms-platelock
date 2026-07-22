"""
platelock.run_dense_cluster — MAINLINE dense RoMa-anchor reconstruction runner.

Promoted 2026-07-01 from the proven prototype _cluster_run.pyx after the dense
branch beat every method on the crux cluster (frames 110-120): dense gradient-NCC
+0.39..+0.51 on left_window vs raw +0.06..+0.33 and CoTracker -0.09..-0.03, with
folds 0.3-1.1%. See platelock/configs/dense_crux_baseline.json.

Pipeline (per frame):
    RoMa match (original -> clean)
    -> extract_anchors -> filter_anchors(thresh) -> roi_balanced_topk
    -> fit_affine (global prior) -> reconstruct (affine + local TPS residual)
    -> apply_backward_map -> per-ROI gradient-NCC (raw / dense / cotracker)

CoTracker is a VALIDATOR/COMPARISON baseline only; it never drives the warp.

Outputs (under --out): summary_table.json (gradient_ncc + field_health),
per-frame ROI overlay jpgs, displacement previews, anchor previews, and
cluster preview mp4s.

Usage:
    python -m platelock.run_dense_cluster [--config platelock/configs/dense_crux.yaml]
                                          [--out output/dense_cluster_110_120]
                                          [--frames 110,112,114,115,116,118,120]
"""

import os
import glob
import json
import argparse

import cv2
import numpy as np
import yaml
from PIL import Image
from romatch import roma_outdoor

from platelock import geometry as g, diagnostics as dg, dense_field as df, masks as mk


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def grab(path, idx, width):
    """Read frame `idx` from a video, resize to `width` preserving aspect."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx} from {path}")
    h, w = frame.shape[:2]
    s = width / w
    return cv2.resize(frame, (width, int(round(h * s))))


def build_model(cfg, device="cuda"):
    r = cfg["roma"]
    return roma_outdoor(device=device, use_custom_corr=r["use_custom_corr"],
                        coarse_res=r["coarse_res"], upsample_res=r["upsample_res"])


def build_segmenter(cfg, device="cuda"):
    """Returns a Sam3Segmenter if cfg["masks"]["enabled"], else None. Built once,
    reused across all frames (SAM3 model load is ~7s, do not rebuild per frame)."""
    mcfg = cfg.get("masks")
    if not mcfg or not mcfg.get("enabled"):
        return None
    from platelock.segmenters.sam3_segmenter import Sam3Segmenter
    return Sam3Segmenter(checkpoint_path=mcfg["checkpoint"], device=device)


def frame_masks(O, C, segmenter, cfg):
    """Build (orig_mask, clean_mask) at O/C's own resolution, or (None, None) if
    segmenter is disabled. orig_mask = actor UNION real dim sconce (scored on
    ORIGINAL). clean_mask = WAN-invented flame/sconce glow (scored on CLEAN).
    The two live in independent coordinate spaces and are never combined."""
    if segmenter is None:
        return None, None
    mcfg = cfg["masks"]
    orig_mask = segmenter.segment(O, mcfg["actor_prompts"], max_area_frac=1.0)
    if mcfg.get("orig_concept_prompts"):
        orig_mask = orig_mask | segmenter.segment(
            O, mcfg["orig_concept_prompts"],
            max_area_frac=mcfg["concept_max_area_frac"],
            confidence_threshold=mcfg["concept_conf_thresh"])
    clean_mask = segmenter.segment(
        C, mcfg["clean_concept_prompts"],
        max_area_frac=mcfg["concept_max_area_frac"],
        confidence_threshold=mcfg["concept_conf_thresh"])
    return orig_mask, clean_mask


def dense_map_for_frame(O, C, model, cfg, device="cuda", segmenter=None):
    """Run the full anchor->reconstruct pipeline; return (map_x, map_y, n_anchors)."""
    H, W = O.shape[:2]
    oi = Image.fromarray(cv2.cvtColor(O, cv2.COLOR_BGR2RGB))
    ci = Image.fromarray(cv2.cvtColor(C, cv2.COLOR_BGR2RGB))
    warp, cert = model.match(oi, ci, device=device)
    orig, clean, c = df.extract_anchors(warp, cert, H, W, model)
    orig_mask, clean_mask = frame_masks(O, C, segmenter, cfg)
    if orig_mask is not None or clean_mask is not None:
        orig, clean, c, _ = mk.exclude_anchors(orig, clean, c, orig_mask=orig_mask, clean_mask=clean_mask)
    a = cfg["anchors"]
    o0, cl0, c0, _ = df.filter_anchors(orig, clean, c, thresh=a["cert_thresh"])
    rb = a["roi_balanced"]
    o, cl, cc = df.roi_balanced_topk(o0, cl0, c0, H, W, tiles=tuple(rb["tiles"]),
                                     frac=rb["frac"], max_total=rb["max_total"])
    M, inl = df.fit_affine(o, cl, thresh=cfg["affine"]["ransac_thresh"])
    rc = cfg["reconstruct"]
    mx, my = df.reconstruct(o, cl, H, W, M, grid=tuple(rc["grid"]),
                            neighbors=rc["neighbors"], smoothing=rc["smoothing"],
                            max_anchors=rc["max_anchors"], boundary=tuple(rc["boundary"]))
    return mx, my, o


def make_vid(pattern, out, fps=4):
    files = sorted(glob.glob(pattern))
    if not files:
        return
    im = cv2.imread(files[0])
    h, w = im.shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in files:
        vw.write(cv2.resize(cv2.imread(f), (w, h)))
    vw.release()


def run(cfg, out_dir, frames, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    inp = cfg["inputs"]
    width = cfg["working"]["width"]
    rois = cfg["crux"]["rois"]
    model = build_model(cfg, device)
    segmenter = build_segmenter(cfg, device)

    summary, health = {}, {}
    for fr in frames:
        O = grab(inp["original"], fr, width)
        C = grab(inp["clean"], fr, width)
        H, W = O.shape[:2]
        ct = cv2.resize(grab(inp["cotracker"], fr, width), (W, H))

        mx, my, anchors = dense_map_for_frame(O, C, model, cfg, device, segmenter=segmenter)
        warped = g.apply_backward_map(C, mx, my)
        health[fr] = df.field_health(mx, my)
        health[fr]["anchors"] = int(len(anchors))

        rec = {}
        for n, (x, y, w, h) in rois.items():
            oc = O[y:y + h, x:x + w]
            rec[n] = {"raw": round(dg.grad_ncc(oc, C[y:y + h, x:x + w]), 3),
                      "dense": round(dg.grad_ncc(oc, warped[y:y + h, x:x + w]), 3),
                      "cotracker": round(dg.grad_ncc(oc, ct[y:y + h, x:x + w]), 3)}
        summary[fr] = rec

        for n in ("left_window", "sconce_support"):
            x, y, w, h = rois[n]
            oc = O[y:y + h, x:x + w]
            panels = [
                dg.label(dg.edge_overlay(oc, C[y:y + h, x:x + w]),
                         f"f{fr} {n} RAW ncc={rec[n]['raw']:+.2f}"),
                dg.label(dg.edge_overlay(oc, warped[y:y + h, x:x + w]),
                         f"DENSE ncc={rec[n]['dense']:+.2f}"),
                dg.label(dg.edge_overlay(oc, ct[y:y + h, x:x + w]),
                         f"cotracker ncc={rec[n]['cotracker']:+.2f}")]
            cv2.imwrite(f"{out_dir}/f{fr:03d}_{n}.jpg", dg.hstack_pad(panels, pad=6))

        cv2.imwrite(f"{out_dir}/f{fr:03d}_disp.jpg",
                    df.displacement_preview(mx, my, H, W, step=44))
        av = O.copy()
        for (px, py) in anchors.astype(int):
            cv2.circle(av, (px, py), 1, (0, 255, 255), -1)
        cv2.imwrite(f"{out_dir}/f{fr:03d}_anchors.jpg", av)

        print(f"f{fr}: anchors {len(anchors)} fold {health[fr]['fold_frac']:.4f} | "
              f"left raw {rec['left_window']['raw']:+.2f} dense {rec['left_window']['dense']:+.2f} "
              f"ct {rec['left_window']['cotracker']:+.2f} | "
              f"sconce dense {rec['sconce_support']['dense']:+.2f}", flush=True)

    json.dump({"gradient_ncc": summary, "field_health": health},
              open(f"{out_dir}/summary_table.json", "w"), indent=2)
    make_vid(f"{out_dir}/f*_left_window.jpg", f"{out_dir}/cluster_left_window.mp4")
    make_vid(f"{out_dir}/f*_disp.jpg", f"{out_dir}/cluster_disp.mp4")
    print("WROTE", out_dir, "summary_table.json + per-frame jpgs + cluster mp4s")


def main():
    ap = argparse.ArgumentParser(description="Dense RoMa-anchor crux runner (mainline).")
    ap.add_argument("--config", default="platelock/configs/dense_crux.yaml")
    ap.add_argument("--out", default="output/dense_cluster_110_120")
    ap.add_argument("--frames", default=None, help="comma-separated override")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frames = ([int(x) for x in args.frames.split(",")] if args.frames
              else cfg["crux"]["frames"])
    run(cfg, args.out, frames, args.device)


if __name__ == "__main__":
    main()

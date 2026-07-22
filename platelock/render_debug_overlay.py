"""
platelock.render_debug_overlay — regenerate CLAHE edge-overlay panels for any
frames/ROIs without rerunning the whole cluster.

Promoted 2026-07-01 from _regen_overlay.pyx. Uses the SAME mainline dense
pipeline as run_dense_cluster (imported, not duplicated), so overlays always
reflect the frozen crux config (platelock/configs/dense_crux.yaml).

Each panel row is: RAW (clean vs original) | DENSE (warped-clean vs original) |
cotracker (comparison baseline), CLAHE-normalised so dim crops are legible,
each labelled with its gradient-NCC. Green = original edges, red = candidate,
yellow = lock.

Usage:
    python -m platelock.render_debug_overlay \
        --frames 115,120 --rois left_window --out output/dense_cluster_110_120
    # --rois names must exist in the config's crux.rois block; default = all.
"""

import argparse

import cv2
from PIL import Image  # noqa: F401  (kept for parity w/ pipeline import surface)

from platelock import geometry as g, diagnostics as dg
from platelock.run_dense_cluster import (
    load_config, build_model, grab, dense_map_for_frame,
)


def render(cfg, frames, roi_names, out_dir, device="cuda", suffix="clahe"):
    inp = cfg["inputs"]
    width = cfg["working"]["width"]
    all_rois = cfg["crux"]["rois"]
    roi_names = roi_names or list(all_rois.keys())
    model = build_model(cfg, device)

    for fr in frames:
        O = grab(inp["original"], fr, width)
        C = grab(inp["clean"], fr, width)
        H, W = O.shape[:2]
        ct = cv2.resize(grab(inp["cotracker"], fr, width), (W, H))

        mx, my, _ = dense_map_for_frame(O, C, model, cfg, device)
        warped = g.apply_backward_map(C, mx, my)

        for n in roi_names:
            x, y, w, h = all_rois[n]
            oc = O[y:y + h, x:x + w]
            cc = C[y:y + h, x:x + w]
            wc = warped[y:y + h, x:x + w]
            tc = ct[y:y + h, x:x + w]
            panels = [
                dg.label(dg.edge_overlay(oc, cc),
                         f"f{fr} {n} RAW ncc={dg.grad_ncc(oc, cc):+.2f}"),
                dg.label(dg.edge_overlay(oc, wc),
                         f"DENSE ncc={dg.grad_ncc(oc, wc):+.2f}"),
                dg.label(dg.edge_overlay(oc, tc),
                         f"cotracker ncc={dg.grad_ncc(oc, tc):+.2f}")]
            path = f"{out_dir}/f{fr:03d}_{n}_{suffix}.jpg"
            cv2.imwrite(path, dg.hstack_pad(panels, pad=6))
            print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Regenerate CLAHE debug overlays.")
    ap.add_argument("--config", default="platelock/configs/dense_crux.yaml")
    ap.add_argument("--out", default="output/dense_cluster_110_120")
    ap.add_argument("--frames", required=True, help="comma-separated frame indices")
    ap.add_argument("--rois", default=None,
                    help="comma-separated ROI names (default: all in config)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suffix", default="clahe")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frames = [int(x) for x in args.frames.split(",")]
    roi_names = [x for x in args.rois.split(",")] if args.rois else None
    render(cfg, frames, roi_names, args.out, args.device, args.suffix)


if __name__ == "__main__":
    main()

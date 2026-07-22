"""
platelock.export_5k — upscale the working-res smoothed backward map to native
5K and emit the shippable deliverables: warped clean frames (PNG), the 32-bit
float EXR STMap sequence, and trusted-crop diagnostic overlays at 5K.

Reuses the ALREADY-SOLVED working-res geometry (params/M_savgol_w{N}.npy +
cres_savgol_w{N}.npy from run_full_sequence.py) — no RoMa rerun. Per the
Phase 2 resolution sweep (2026-07-02), solving at higher RoMa upsample_res
does not improve lock quality on this shot, so the working-res solve is
already the practical ceiling; this stage only changes the SAMPLING density
of that same solved field, via geometry.upscale_backward_map.

Usage:
    python -m platelock.export_5k --run-dir output/dense_full_workingres_masked \
        --out output/export_5k --frames 0-125 --window 7
"""

import os
import argparse

import cv2
import numpy as np

from platelock import geometry as g, diagnostics as dg, dense_field as df
from platelock.run_dense_cluster import load_config, grab
from platelock.temporal_smooth import parse_frames

CRUX_ROIS_WORKING = {
    "left_window":    (20, 20, 280, 500),
    "sconce_support": (460, 160, 220, 230),
    "right_window":   (1080, 320, 280, 320),
}


def scale_roi(roi, sx, sy):
    x, y, w, h = roi
    return (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy)))


def export_frame(fr, M, cres, cfg, native_H, native_W, working_H, working_W, out_dir, overlay=False):
    inp = cfg["inputs"]
    O_native = grab(inp["original"], fr, native_W)
    C_native = grab(inp["clean"], fr, native_W)

    mx_w, my_w = df.assemble_map(M, cres, working_H, working_W)
    mx_5k, my_5k = g.upscale_backward_map(mx_w, my_w, native_H, native_W)

    warped_5k = g.apply_backward_map(C_native, mx_5k, my_5k)
    cv2.imwrite(f"{out_dir}/warped/warped_f{fr:04d}.png", warped_5k)

    stmap = g.backward_map_to_stmap(mx_5k, my_5k, native_W, native_H)
    g.write_stmap_exr(stmap, f"{out_dir}/stmaps/stmap_f{fr:04d}.exr")

    if overlay:
        sx, sy = native_W / working_W, native_H / working_H
        for name, roi in CRUX_ROIS_WORKING.items():
            x, y, w, h = scale_roi(roi, sx, sy)
            oc = O_native[y:y + h, x:x + w]
            wc = warped_5k[y:y + h, x:x + w]
            panel = dg.hstack_pad([
                dg.label(dg.edge_overlay(oc, C_native[y:y + h, x:x + w]), f"f{fr} {name} RAW"),
                dg.label(dg.edge_overlay(oc, wc), "DENSE 5K"),
            ], pad=6)
            cv2.imwrite(f"{out_dir}/overlays_5k/f{fr:04d}_{name}.jpg", panel)

    return warped_5k.shape


def main():
    ap = argparse.ArgumentParser(description="Upscale working-res masked solve to native 5K deliverables.")
    ap.add_argument("--run-dir", default="output/dense_full_workingres_masked",
                     help="dir containing params/M_savgol_w{N}.npy + cres_savgol_w{N}.npy")
    ap.add_argument("--config", default="platelock/configs/dense_crux_masked.yaml")
    ap.add_argument("--out", default="output/export_5k")
    ap.add_argument("--frames", default="0-125")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--overlay", action="store_true", help="also write trusted-crop overlays at 5K")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frames = parse_frames(args.frames)
    for sub in ("warped", "stmaps", "overlays_5k"):
        os.makedirs(f"{args.out}/{sub}", exist_ok=True)

    Ms = np.load(f"{args.run_dir}/params/M_savgol_w{args.window}.npy")
    creses = np.load(f"{args.run_dir}/params/cres_savgol_w{args.window}.npy")

    inp = cfg["inputs"]
    probe = cv2.VideoCapture(inp["original"])
    native_W = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_H = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    working_W = cfg["working"]["width"]
    working_H = int(round(native_H * (working_W / native_W)))
    print(f"native resolution: {native_W}x{native_H}  working: {working_W}x{working_H}")

    for fr in frames:
        shape = export_frame(fr, Ms[fr], creses[fr], cfg, native_H, native_W,
                             working_H, working_W, args.out, overlay=args.overlay)
        print(f"  f{fr}: exported warped {shape[1]}x{shape[0]} + stmap EXR"
              f"{' + overlay' if args.overlay else ''}", flush=True)

    print(f"WROTE {args.out}/  (warped/ stmaps/ overlays_5k/)")


if __name__ == "__main__":
    main()

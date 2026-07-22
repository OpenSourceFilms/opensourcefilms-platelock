#!/usr/bin/env python3
"""
Recover ECC metrics and previews from existing flow .npy files.
Used after the main benchmark ran ECC but crashed before metrics/previews
due to memory exhaustion.

Usage:
    source env/platewarp/bin/activate
    python scripts/recover_ecc_results.py \
        --original input/original \
        --clean input/clean/clean_24fps_288frames.mp4 \
        --flows output/full_sequence/ecc/flows \
        --out output/full_sequence/ecc \
        --resize 1920x1080
"""
import argparse, json, sys, time
from pathlib import Path

import imageio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.frame_io import normalise_pair
from utils.metrics import compute_frame_metrics
from utils.previews import (make_side_by_side, make_difference, make_checkerboard,
                             make_wipe, make_overlay_50)
from utils.flow_utils import warp_image_with_flow


def parse_resize(s):
    if not s or s.lower() == "none":
        return None
    w, h = s.lower().split("x")
    return int(w), int(h)


def _writer(path: Path, fps: float, crf: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, format="FFMPEG",
                               codec="libx264", output_params=["-crf", str(crf)])


def _save_still(frame: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), frame)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--clean", required=True)
    p.add_argument("--flows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resize", default="1920x1080")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--crf", type=int, default=18)
    args = p.parse_args()

    resize = parse_resize(args.resize)
    flows_dir = Path(args.flows)
    out_dir = Path(args.out)

    flow_paths = sorted(flows_dir.glob("flow_*.npy"))
    if not flow_paths:
        sys.exit(f"No flow_*.npy files in {flows_dir}")
    n = len(flow_paths)
    print(f"Found {n} flow files")

    print("Loading frames...")
    originals, cleans, meta = normalise_pair(args.original, args.clean, None, resize)
    if len(originals) != n:
        mn = min(len(originals), n)
        print(f"  Truncating to {mn} (orig={len(originals)}, flows={n})")
        originals = originals[:mn]
        cleans = cleans[:mn]
        flow_paths = flow_paths[:mn]
        n = mn
    fps = meta.get("fps") or args.fps
    H, W = originals[0].shape[:2]
    print(f"  {n} frames  {W}x{H}  fps={fps}")

    prev_dir = out_dir / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "warped_clean": _writer(prev_dir / "warped_clean.mp4",  fps, args.crf),
        "side_by_side": _writer(prev_dir / "side_by_side.mp4",  fps, args.crf),
        "difference":   _writer(prev_dir / "difference.mp4",    fps, args.crf),
        "checkerboard": _writer(prev_dir / "checkerboard.mp4",  fps, args.crf),
        "wipe_preview": _writer(prev_dir / "wipe_preview.mp4",  fps, args.crf),
        "overlay_50":   _writer(prev_dir / "overlay_50.mp4",    fps, args.crf),
    }

    rows = []
    still_frames = {0, n // 2, n - 1}
    t0 = time.perf_counter()

    print("Processing frames...")
    for i, (orig, clean, fp) in enumerate(zip(originals, cleans, flow_paths)):
        flow = np.load(str(fp))
        warped = warp_image_with_flow(clean, flow)

        m = compute_frame_metrics(orig, warped)
        rows.append({"frame": i, "method": "ecc", **m})

        writers["warped_clean"].append_data(warped)
        writers["side_by_side"].append_data(make_side_by_side(orig, warped))
        writers["difference"].append_data(make_difference(orig, warped))
        writers["checkerboard"].append_data(make_checkerboard(orig, warped))
        writers["wipe_preview"].append_data(make_wipe(orig, warped))
        writers["overlay_50"].append_data(make_overlay_50(orig, warped))

        if i in still_frames:
            _save_still(make_side_by_side(orig, warped), prev_dir / "stills" / f"frame_{i:04d}_sbs.png")
            _save_still(make_wipe(orig, warped),         prev_dir / "stills" / f"frame_{i:04d}_wipe.png")
            _save_still(make_difference(orig, warped),   prev_dir / "stills" / f"frame_{i:04d}_diff.png")

        if (i + 1) % 48 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            tx = float(flow[H // 2, W // 2, 0])
            ty = float(flow[H // 2, W // 2, 1])
            print(f"  [{i+1}/{n}]  PSNR={m['psnr']:.2f}  offset=({tx:.2f},{ty:.2f})px  "
                  f"{(i+1)/elapsed:.1f} fr/s")

    for w in writers.values():
        w.close()

    df = pd.DataFrame(rows)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "per_frame.csv", index=False)

    worst5 = df.nsmallest(5, "psnr")["frame"].tolist()
    for i in worst5:
        if i not in still_frames:
            flow = np.load(str(flow_paths[i]))
            warped = warp_image_with_flow(cleans[i], flow)
            _save_still(make_side_by_side(originals[i], warped),
                        prev_dir / "stills" / f"frame_{i:04d}_sbs.png")
            _save_still(make_wipe(originals[i], warped),
                        prev_dir / "stills" / f"frame_{i:04d}_wipe.png")

    summary = {
        "mean_psnr":      float(df["psnr"].mean()),
        "median_psnr":    float(df["psnr"].median()),
        "mean_ssim":      float(df["ssim"].mean()),
        "mean_mae":       float(df["mae"].mean()),
        "worst_5_frames": worst5,
    }
    result_meta = {
        "method": "ecc",
        "runtime_seconds": round(time.perf_counter() - t0, 2),
        "metrics_summary": summary,
        "status": "success (recovered from flows)",
    }
    json.dump(result_meta, open(out_dir / "method_result.json", "w"), indent=2)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s")
    print(f"  mean_psnr={summary['mean_psnr']:.4f}  mean_ssim={summary['mean_ssim']:.4f}  "
          f"mean_mae={summary['mean_mae']:.6f}")
    print(f"  worst frames: {worst5}")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()

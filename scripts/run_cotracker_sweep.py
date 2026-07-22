#!/usr/bin/env python3
"""Sweep runner: CoTracker mesh across chunk sizes.

Usage:
  python scripts/run_cotracker_sweep.py \
    --original input/original/ExampleShot_5K8bit_24fps_Pan_RETIMED_01.mp4 \
    --clean    input/clean/ExampleShot_WAN2700p_24fps_Pan_RETIMED_01.mp4 \
    --out      output/cotracker_sweep_01 \
    --chunk-sizes 16,24,32,40,56 \
    --resize 1920x1080 --crop-previews true
"""
import argparse, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sweep")


def parse_wh(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def to_bool(s):
    return s.lower() in ("true", "1", "yes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original",           required=True)
    p.add_argument("--clean",              required=True)
    p.add_argument("--out",                required=True)
    p.add_argument("--chunk-sizes",        default="16,24,32,40,56")
    p.add_argument("--tracking-wh",        default="960x540")
    p.add_argument("--grid-size",          type=int, default=40)
    p.add_argument("--overlap",            type=int, default=8)
    p.add_argument("--per-chunk-seed-ecc", default="true")
    p.add_argument("--bg-mask-threshold",  type=int, default=30)
    p.add_argument("--resize",             default="1920x1080")
    p.add_argument("--start",              type=int, default=None)
    p.add_argument("--end",                type=int, default=None)
    p.add_argument("--crop-previews",      default="true")
    args = p.parse_args()

    from utils.frame_io import normalise_pair
    from utils.metrics import compute_sequence_metrics, summarise_metrics
    from utils.previews import generate_all_previews
    from utils.crop_previews import generate_crop_previews
    from methods.cotracker_mesh import CoTrackerMeshMethod

    resize       = parse_wh(args.resize) if args.resize.lower() != "none" else None
    tracking_wh  = parse_wh(args.tracking_wh)
    chunk_sizes  = [int(x) for x in args.chunk_sizes.split(",")]
    per_ecc      = to_bool(args.per_chunk_seed_ecc)
    do_crops     = to_bool(args.crop_previews)
    frame_range  = (args.start or 0, args.end or 999999)
    out_dir      = Path(args.out)

    log.info("Loading frames…")
    orig_frames, clean_frames, meta = normalise_pair(
        args.original, args.clean, frame_range, resize)
    fps = meta.get("fps") or 24.0
    log.info("Loaded %d frames  %dx%d  fps=%.2f",
             meta["frame_count"], meta["width"], meta["height"], fps)

    results = []
    for cs in chunk_sizes:
        log.info("=== chunk_size=%d ===", cs)
        m = CoTrackerMeshMethod(
            grid_size=args.grid_size, tracking_wh=tracking_wh,
            chunk_size=cs, overlap=args.overlap,
            per_chunk_seed_ecc=per_ecc,
            bg_mask_threshold=args.bg_mask_threshold,
        )
        t0 = time.perf_counter()
        result = m.process_sequence(orig_frames, clean_frames)
        elapsed = time.perf_counter() - t0
        warped = result["warped_frames"]

        summary = summarise_metrics(compute_sequence_metrics(orig_frames, warped))
        psnr = summary.get("mean_psnr", float("nan"))
        ssim = summary.get("mean_ssim", float("nan"))
        mae  = summary.get("mean_mae",  float("nan"))
        log.info("chunk=%d  PSNR=%.4f  SSIM=%.4f  MAE=%.5f  t=%.1fs",
                 cs, psnr, ssim, mae, elapsed)

        method_dir = out_dir / f"chunk{cs:03d}" / "cotracker_mesh"
        generate_all_previews(orig_frames, warped, method_dir / "previews", fps=fps)
        if do_crops:
            generate_crop_previews(orig_frames, warped,
                                   method_dir / "crop_previews", fps=fps)
        (method_dir / "summary.txt").parent.mkdir(parents=True, exist_ok=True)
        (method_dir / "summary.txt").write_text(
            f"chunk_size={cs} overlap={args.overlap} tracking_wh={tracking_wh} "
            f"grid_size={args.grid_size} per_chunk_seed_ecc={per_ecc}\n"
            f"PSNR={psnr:.4f}  SSIM={ssim:.4f}  MAE={mae:.5f}  t={elapsed:.1f}s\n"
        )
        results.append(dict(chunk_size=cs, psnr=psnr, ssim=ssim, mae=mae, t=elapsed))

    log.info("\n=== Sweep results ===")
    log.info("%-10s  %-8s  %-8s  %-9s  %s", "chunk", "PSNR", "SSIM", "MAE", "time")
    for r in results:
        log.info("%-10d  %-8.4f  %-8.4f  %-9.5f  %.1fs",
                 r["chunk_size"], r["psnr"], r["ssim"], r["mae"], r["t"])


if __name__ == "__main__":
    main()

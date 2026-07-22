#!/usr/bin/env python3
"""
Run a single warp method on a pair of video/image sequences.

Usage:
    python scripts/run_method.py \
        --method searaft \
        --original input/original \
        --clean input/clean \
        --out output/searaft/test_001 \
        [--config configs/default.yaml] \
        [--start 0] [--end 120] [--resize none|WxH]
"""
import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.frame_io import normalise_pair, video_from_frames
from utils.flow_utils import flow_to_stmap, write_flow_npy, smooth_flow_spatial, smooth_flow_temporal, clamp_flow
from utils.metrics import compute_sequence_metrics, summarise_metrics, find_worst_frames
from utils.previews import generate_all_previews, save_comparison_stills

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_method")

METHOD_MODEL = {
    "searaft": "sea_raft_l", "raft": "raft_large",
    "flowseek": "flowseek", "dpflow": "dpflow",
}


def parse_resize(s):
    if s is None or s.lower() == "none":
        return None
    w, h = s.lower().split("x")
    return (int(w), int(h))


def run(args):
    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    frame_range = None
    if args.start is not None or args.end is not None:
        frame_range = (args.start or 0, args.end or 999999)
    resize = parse_resize(args.resize)

    log.info("Loading frames...")
    originals, cleans, meta = normalise_pair(args.original, args.clean, frame_range, resize)
    fps = meta.get("fps") or 24.0
    log.info("Loaded %d frames  %dx%d", meta["frame_count"], meta["width"], meta["height"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- instantiate method ---
    m = args.method
    t0 = time.perf_counter()
    warped_frames, flows = [], []

    if m == "ecc":
        from methods.ecc_method import ECCMethod
        ecc_cfg = cfg.get("ecc", {})
        method = ECCMethod(mode=ecc_cfg.get("mode", "homography"),
                           num_iterations=ecc_cfg.get("num_iterations", 200),
                           termination_eps=ecc_cfg.get("termination_eps", 1e-5))
        warped_frames, transforms, infos = method.align_sequence(originals, cleans)
        flows = [None] * len(warped_frames)  # ECC has no dense flow
        json.dump([i for i in infos], open(out_dir / "ecc_infos.json", "w"), indent=2)

    elif m == "cotracker_mesh":
        from methods.cotracker_mesh import CoTrackerMeshMethod
        ct_cfg = cfg.get("cotracker_mesh", {})
        method = CoTrackerMeshMethod(**{k: ct_cfg[k] for k in ("grid_size","min_vis_ratio","smooth_trajectories") if k in ct_cfg})
        result = method.process_sequence(originals, cleans)
        warped_frames = result["warped_frames"]
        flows = [None] * len(warped_frames)

    else:
        from methods.ptlflow_method import PTLFlowMethod
        flow_cfg = cfg.get("flow", {})
        model_name = METHOD_MODEL.get(m) or cfg.get("ptlflow", {}).get("model", "sea_raft_l")
        pt_cfg = cfg.get("ptlflow", {})
        method = PTLFlowMethod(model_name=model_name,
                               iters=pt_cfg.get("iters", 20),
                               mixed_precision=pt_cfg.get("mixed_precision", True))
        result = method.process_sequence(originals, cleans, flow_cfg=flow_cfg)
        warped_frames = result["warped_frames"]
        flows = result["flows"]

    runtime = time.perf_counter() - t0
    log.info("Method completed in %.1fs", runtime)

    # --- apply any remaining flow post-processing ---
    flow_cfg = cfg.get("flow", {})
    if flows[0] is not None:
        sp = flow_cfg.get("spatial_blur_px", 0)
        tr = flow_cfg.get("temporal_smooth_radius", 0)
        cm = flow_cfg.get("clamp_p95_multiplier", 0)
        if sp > 0: flows = [smooth_flow_spatial(f, sp) for f in flows]
        if tr > 0: flows = smooth_flow_temporal(flows, tr)
        if cm > 0: flows = [clamp_flow(f, multiplier=cm) for f in flows]
        # save flows
        flow_dir = out_dir / "flows"
        flow_dir.mkdir(exist_ok=True)
        for i, f in enumerate(flows):
            if f is not None:
                write_flow_npy(f, flow_dir / f"flow_{i:04d}.npy")

    # --- STMap export ---
    if flows[0] is not None and cfg.get("stmap", {}).get("export", True):
        stmap_dir = out_dir / "stmaps"
        stmap_dir.mkdir(exist_ok=True)
        H, W = originals[0].shape[:2]
        for i, f in enumerate(flows):
            if f is not None:
                sm = flow_to_stmap(f, H, W)
                np.save(str(stmap_dir / f"stmap_{i:04d}.npy"), sm)

    # --- metrics ---
    log.info("Computing metrics...")
    metrics_dir = out_dir / "metrics"
    df = compute_sequence_metrics(originals, warped_frames, method_name=m, output_dir=metrics_dir)
    summary = summarise_metrics(df)
    worst = find_worst_frames(df, 5, "psnr")

    # --- previews ---
    log.info("Generating previews...")
    prev_dir = out_dir / "previews"
    preview_cfg = cfg.get("previews", {})
    generate_all_previews(originals, warped_frames, prev_dir,
                           fps=fps, crf=preview_cfg.get("crf", 23))
    n = len(originals)
    stills = list({0, n // 2, n - 1} | set(worst))
    save_comparison_stills(originals, warped_frames, prev_dir / "stills", stills)

    # --- metadata ---
    meta_out = {
        "method": m, "model": getattr(method, "model_name", m),
        "frame_count": len(originals), "runtime_seconds": round(runtime, 2),
        "metrics_summary": summary, "output_dir": str(out_dir),
    }
    json.dump(meta_out, open(out_dir / "method_result.json", "w"), indent=2)

    log.info("Done. PSNR=%.2f  SSIM=%.4f  MAE=%.5f",
             summary["mean_psnr"], summary["mean_ssim"], summary["mean_mae"])
    log.info("Output: %s", out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--clean", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--resize", default="none")
    run(p.parse_args())


if __name__ == "__main__":
    main()

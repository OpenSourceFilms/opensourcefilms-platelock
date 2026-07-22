#!/usr/bin/env python3
"""
Run multiple warp methods on one shot pair and write comparative outputs.

Usage:
    python scripts/run_benchmark.py \
        --original /path/to/original \
        --clean /path/to/clean \
        --out output/test_001 \
        --methods ecc searaft raft cotracker_mesh \
        --config configs/full_quality.yaml
"""
import argparse, json, logging, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.frame_io import normalise_pair
from utils.flow_utils import flow_to_stmap, write_flow_npy, smooth_flow_spatial, smooth_flow_temporal, clamp_flow, affine_to_flow
from utils.metrics import compute_sequence_metrics, summarise_metrics, find_worst_frames
from utils.previews import generate_all_previews, save_comparison_stills

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_benchmark")

METHOD_MODEL = {
    "searaft": "sea_raft_l", "raft": "raft",
    "flowseek": "flowseek_m", "dpflow": "dpflow",
}
ALL_METHODS = ["ecc", "farneback", "dual_video", "raft", "searaft", "ptlflow", "flowseek", "dpflow", "cotracker_mesh"]


def parse_resize(s):
    if not s or s.lower() == "none":
        return None
    w, h = s.lower().split("x")
    return (int(w), int(h))


def run_one_method(method_name, originals, cleans, out_dir, cfg, fps):
    """Run a single method. Returns (warped_frames, flows, summary_dict)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    flows = None

    if method_name == "ecc":
        from methods.ecc_method import ECCMethod
        ec = cfg.get("ecc", {})
        m = ECCMethod(mode=ec.get("mode", "homography"),
                      num_iterations=ec.get("num_iterations", 200),
                      termination_eps=ec.get("termination_eps", 1e-5))
        warped, transforms, infos = m.align_sequence(originals, cleans)
        # Convert affine transforms to flow fields for STMap export
        H, W = originals[0].shape[:2]
        flows = [affine_to_flow(t, H, W) for t in transforms]

    elif method_name == "farneback":
        from methods.farneback_method import FarnebackMethod
        fb = cfg.get("farneback", {})
        m = FarnebackMethod(
            max_corners=fb.get("max_corners", 2000),
            quality_level=fb.get("quality_level", 0.01),
            min_distance=fb.get("min_distance", 12),
            lk_win_size=fb.get("lk_win_size", 21),
            lk_max_level=fb.get("lk_max_level", 4),
            ransac_threshold=fb.get("ransac_threshold", 2.0),
            temporal_smooth_kernel=fb.get("temporal_smooth_kernel", 5),
            max_jump_px=fb.get("max_jump_px", 0.5),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]
        # Convert affine transforms to flow fields for STMap export
        H, W = originals[0].shape[:2]
        flows = [affine_to_flow(warp, H, W) for warp in result["warps"]]

    elif method_name == "dual_video":
        from methods.dual_video_method import DualVideoMethod
        dv = cfg.get("dual_video", {})
        m = DualVideoMethod(
            temporal_smooth_kernel=dv.get("temporal_smooth_kernel", 7),
            ecc_iters=dv.get("ecc_iters", 500),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]
        H, W = originals[0].shape[:2]
        flows = [affine_to_flow(warp, H, W) for warp in result["warps"]]

    elif method_name == "cotracker_mesh":
        from methods.cotracker_mesh import CoTrackerMeshMethod
        ct = cfg.get("cotracker_mesh", {})
        _ct_keys = ("grid_size", "min_vis_ratio", "smooth_trajectories",
                    "chunk_size", "overlap", "per_chunk_seed_ecc", "bg_mask_threshold")
        ct_kwargs = {k: ct[k] for k in _ct_keys if k in ct}
        if "tracking_wh" in ct:
            ct_kwargs["tracking_wh"] = tuple(ct["tracking_wh"])
        m = CoTrackerMeshMethod(**ct_kwargs)
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]

    elif method_name == "tapnext_mesh":
        from methods.tapnext_method import TAPNextMethod as TAPNextMeshMethod
        tn = cfg.get("tapnext", {})
        m = TAPNextMeshMethod(
            grid_size=tn.get("grid_size", 30),
            min_vis_ratio=tn.get("min_vis_ratio", 0.3),
            smooth_trajectories=tn.get("smooth_trajectories", True),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]

    elif method_name == "roma_v2":
        from methods.roma_method import RoMAMethod
        rm = cfg.get("roma", {})
        m = RoMAMethod(
            model_variant=rm.get("model_variant", "outdoor"),
            certainty_threshold=rm.get("certainty_threshold", 0.02),
            max_matches=rm.get("max_matches", 4000),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]

    elif method_name == "sift_homography":
        from methods.sift_method import SIFTHomographyMethod
        sc = cfg.get("sift_homography", {})
        m = SIFTHomographyMethod(
            n_features=sc.get("n_features", 3000),
            ratio_threshold=sc.get("ratio_threshold", 0.75),
            ransac_threshold=sc.get("ransac_threshold", 4.0),
            min_inliers=sc.get("min_inliers", 20),
            temporal_sigma=sc.get("temporal_sigma", 1.5),
            do_colour_normalise=sc.get("do_colour_normalise", True),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]
        flows = result["flows"]

    elif method_name == "structural_match":
        from methods.structural_match_method import StructuralMatchMethod
        sm = cfg.get("structural_match", {})
        m = StructuralMatchMethod(
            sift_n_features=sm.get("sift_n_features", 8000),
            representations=sm.get("representations", ['clahe', 'gradmag', 'dog']),
            ransac_thresh=sm.get("ransac_thresh", 4.0),
            min_inliers=sm.get("min_inliers", 15),
            temporal_sigma=sm.get("temporal_sigma", 1.5),
            use_lightglue=sm.get("use_lightglue", True),
            lightglue_max_w=sm.get("lightglue_max_w", 1600),
            border_reject_px=sm.get("border_reject_px", 15),
            edge_improvement_threshold=sm.get("edge_improvement_threshold", 0.90),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]
        flows = result["flows"]
        identity_frames = [t for t, d in enumerate(result["debug_per_frame"]) if d.get("model_used") == "identity"]
        if identity_frames:
            log.warning("Identity fallback frames: %s", identity_frames)

    elif method_name == "masked_sift":
        from methods.masked_sift_method import MaskedSIFTMethod
        ms = cfg.get("masked_sift", {})
        m = MaskedSIFTMethod(
            diff_threshold=ms.get("diff_threshold", 55),
            mask_dilate_px=ms.get("mask_dilate_px", 40),
            min_inliers=ms.get("min_inliers", 15),
            sift_n_features=ms.get("sift_n_features", 8000),
            ransac_thresh=ms.get("ransac_thresh", 4.0),
            temporal_sigma=ms.get("temporal_sigma", 1.5),
            use_gradient=ms.get("use_gradient", True),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]
        flows = result["flows"]
        for dbg in result["debug_per_frame"]:
            if dbg.get("model_used") == "identity":
                log.warning("Frame fell back to identity: actor_pct=%.1f%% inliers=%d",
                            dbg["actor_mask_pct"], dbg["inliers"])

    elif method_name == "bridge_compose":
        from methods.bridge_compose_method import BridgeComposeMethod
        bc = cfg.get("bridge_compose", {})
        m = BridgeComposeMethod(
            ref_frame=bc.get("ref_frame", 0),
            min_inliers_bridge=bc.get("min_inliers_bridge", 15),
            min_inliers_tracking=bc.get("min_inliers_tracking", 30),
            ransac_thresh=bc.get("ransac_thresh", 4.0),
            temporal_sigma=bc.get("temporal_sigma", 0.0),
            use_loftr=bc.get("use_loftr", True),
            use_lightglue=bc.get("use_lightglue", True),
            loftr_outdoor=bc.get("loftr_outdoor", True),
            sift_n_features=bc.get("sift_n_features", 8000),
        )
        result = m.process_sequence(originals, cleans)
        warped = result["warped_frames"]
        flows = result["flows"]
        log.info("Bridge debug: %s", result.get("bridge_debug", {}))

    elif method_name == "roi_retime":
        from methods.roi_retime_method import ROIRetimeMethod
        rr = cfg.get("roi_retime", {})
        roi_config = rr.get("roi_config", "configs/rois/example_shot_rois.json")
        m = ROIRetimeMethod(
            roi_config=roi_config,
            dt_range=rr.get("dt_range", 8),
            dx_range=rr.get("dx_range", 120),
            dy_range=rr.get("dy_range", 40),
            work_scale=rr.get("work_scale", 0.25),
            transform_mode=rr.get("transform_mode", "retime+translate"),
            savgol_window=rr.get("savgol_window", 11),
            savgol_poly=rr.get("savgol_poly", 3),
            min_confidence=rr.get("min_confidence", 0.15),
            debug_frames=rr.get("debug_frames", None),
            static_pre_correction=rr.get("static_pre_correction", None),
        )
        result = m.process_sequence(originals, cleans, out_dir=out_dir)
        warped = result["warped_frames"]
        flows = result["flows"]
        log.info("dt curve  [%.2f, %.2f]", min(result["dt_smooth"]), max(result["dt_smooth"]))
        log.info("dx curve  [%.2f, %.2f]", min(result["dx_smooth"]), max(result["dx_smooth"]))
        log.info("dy curve  [%.2f, %.2f]", min(result["dy_smooth"]), max(result["dy_smooth"]))
        low_conf = [t for t, d in enumerate(result["debug_per_frame"])
                    if d["conf"] < m.min_confidence]
        if low_conf:
            log.warning("Low-confidence frames (interpolated): %s", low_conf)

    else:
        from methods.ptlflow_method import PTLFlowMethod
        model_name = METHOD_MODEL.get(method_name) or cfg.get("ptlflow", {}).get("model", "sea_raft_l")
        pc = cfg.get("ptlflow", {})
        m = PTLFlowMethod(model_name=model_name, iters=pc.get("iters", 20),
                          mixed_precision=pc.get("mixed_precision", True))
        result = m.process_sequence(originals, cleans, flow_cfg=cfg.get("flow", {}))
        warped = result["warped_frames"]
        flows = result["flows"]

    runtime = time.perf_counter() - t0

    # post-process flows
    if flows is not None:
        fc = cfg.get("flow", {})
        sp = fc.get("spatial_blur_px", 0)
        tr = fc.get("temporal_smooth_radius", 0)
        cm = fc.get("clamp_p95_multiplier", 0)
        if sp > 0: flows = [smooth_flow_spatial(f, sp) for f in flows]
        if tr > 0: flows = smooth_flow_temporal(flows, tr)
        if cm > 0: flows = [clamp_flow(f, multiplier=cm) for f in flows]

        if cfg.get("output", {}).get("keep_flow_npy", True):
            fd = out_dir / "flows"
            fd.mkdir(exist_ok=True)
            for i, f in enumerate(flows):
                write_flow_npy(f, fd / f"flow_{i:04d}.npy")

        if cfg.get("stmap", {}).get("export", True):
            H, W = originals[0].shape[:2]
            sd = out_dir / "stmaps"
            sd.mkdir(exist_ok=True)
            for i, f in enumerate(flows):
                np.save(str(sd / f"stmap_{i:04d}.npy"), flow_to_stmap(f, H, W))

    # metrics
    metrics_dir = out_dir / "metrics"
    df = compute_sequence_metrics(originals, warped, method_name=method_name, output_dir=metrics_dir)
    summary = summarise_metrics(df)
    worst = find_worst_frames(df, 5, "psnr")

    # previews
    pc = cfg.get("previews", {})
    if pc.get("generate", True):
        prev = out_dir / "previews"
        generate_all_previews(originals, warped, prev, fps=fps, crf=pc.get("crf", 23))
        n = len(originals)
        stills = list({0, n // 2, n - 1} | set(worst))
        save_comparison_stills(originals, warped, prev / "stills", stills)
        # Collect overlay_50 into a flat folder for easy grabbing
        overlay_src = prev / "overlay_50.mp4"
        if overlay_src.exists():
            overlays_dir = out_dir.parent / "overlays"
            overlays_dir.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(overlay_src, overlays_dir / f"{method_name}.mp4")

    result_meta = {"method": method_name, "runtime_seconds": round(runtime, 2),
                   "metrics_summary": summary, "status": "success"}
    json.dump(result_meta, open(out_dir / "method_result.json", "w"), indent=2)
    return warped, flows, {**result_meta, "mean_psnr": summary["mean_psnr"],
                            "mean_ssim": summary["mean_ssim"], "mean_mae": summary["mean_mae"]}


def write_readme(out_root, method_names, records, command, cfg_path):
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    cap = str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else "N/A"
    table = ["| Method | mean_psnr | mean_ssim | mean_mae | status |",
             "|--------|-----------|-----------|----------|--------|"]
    for r in records:
        if r["status"] == "success":
            table.append(f"| {r['method']} | {r['mean_psnr']:.4f} | {r['mean_ssim']:.4f} | {r['mean_mae']:.5f} | ✓ |")
        else:
            table.append(f"| {r['method']} | — | — | — | FAILED |")

    readme = f"""# Benchmark Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Environment
- **GPU**: {gpu} (capability {cap})
- **torch**: {torch.__version__}  |  **CUDA**: {torch.version.cuda}
- **Config**: `{cfg_path}`

## Command
```
{command}
```

## Methods
- Ran: {', '.join(r['method'] for r in records if r['status']=='success') or 'none'}
- Failed: {', '.join(r['method'] for r in records if r['status']=='failed') or 'none'}

## Metrics
{chr(10).join(table)}

> Note: metrics measure residual between warped clean plate and original.
> Lower MAE / higher PSNR+SSIM on background regions indicates better lock.
> Visual inspection of previews is the primary quality signal.

## Preview files
Each method has `previews/warped_clean.mp4`, `side_by_side.mp4`, `difference.mp4`,
`checkerboard.mp4`, `wipe_preview.mp4`, `overlay_50.mp4`.

## STMaps
Nuke-compatible STMaps are in `stmaps/stmap_NNNN.npy` (float32, shape HxWx2, U=ch0 V=ch1).
**Convention**: backward map; origin bottom-left (Nuke convention); V-flip applied.
Load in Nuke: `np.load("stmap.npy")` → write to EXR with R=U, G=V → use STMap node.
"""
    (out_root / "README_RESULTS.md").write_text(readme)
    log.info("README_RESULTS.md written to %s", out_root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--clean", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--resize", default="none")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {} if Path(args.config).exists() else {}
    method_names = args.methods or cfg.get("methods", ["ecc", "searaft"])

    frame_range = None
    if args.start is not None or args.end is not None:
        frame_range = (args.start or 0, args.end or 999999)
    elif cfg.get("input", {}).get("frame_range"):
        frame_range = tuple(cfg["input"]["frame_range"])

    resize = parse_resize(args.resize) or parse_resize(cfg.get("input", {}).get("resize", "none"))

    log.info("Loading frames...")
    originals, cleans, meta = normalise_pair(args.original, args.clean, frame_range, resize)
    fps = meta.get("fps") or 24.0
    log.info("Loaded %d frames  %dx%d", meta["frame_count"], meta["width"], meta["height"])

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    import sys as _sys
    command = " ".join(_sys.argv)

    for method_name in method_names:
        log.info("=== %s ===", method_name)
        try:
            _, _, rec = run_one_method(method_name, originals, cleans,
                                        out_root / method_name, cfg, fps)
            records.append(rec)
            log.info("%s  PSNR=%.2f  SSIM=%.4f  MAE=%.5f  t=%.1fs",
                     method_name, rec["mean_psnr"], rec["mean_ssim"],
                     rec["mean_mae"], rec["runtime_seconds"])
        except Exception as e:
            log.error("Method %s FAILED: %s", method_name, e, exc_info=True)
            (out_root / method_name).mkdir(parents=True, exist_ok=True)
            records.append({"method": method_name, "status": "failed",
                             "mean_psnr": None, "mean_ssim": None, "mean_mae": None})

    # summary CSV
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(metrics_dir / "summary.csv", index=False)

    write_readme(out_root, method_names, records, command, args.config)
    log.info("All methods done. Results: %s", out_root)


if __name__ == "__main__":
    main()

"""
platelock.run_roma_falsify — 5-frame RoMa-vs-CoTracker falsification.

Question this answers (and NOTHING more): can a RoMa-family dense matcher lock
this exact original/clean pair on TRUSTED BACKGROUND crops, better than the
current CoTracker output, without being driven by actor/flame/generated regions?

PASS RULE (per handover): RoMa passes only if trusted-crop edge overlays visibly
improve (lower chamfer) over the raw clean plate — NOT because match count or
RANSAC inlier count is high. The .jpgs are the sign-off; the JSON is support.

Run:
  python3 -m platelock.run_roma_falsify \
    --original input/original/ExampleShot_5K8bit_24fps_Pan_RETIMED_01.mp4 \
    --clean    input/clean/ExampleShot_WAN2700p_24fps_Pan_RETIMED_01.mp4 \
    --rois     configs/rois/example_shot_rois.json \
    --cotracker output/perchunk_full/cotracker_mesh/previews/warped_clean.mp4 \
    --out      output/roma_falsify_01
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

from platelock import geometry as g
from platelock import warp_fit as wf
from platelock import diagnostics as dg
from platelock import roi_loader as rl
from platelock.correspondence.roma_provider import RomaProvider

FRAMES = [0, 30, 60, 90, 125]
CERT_THRESH = 0.5


def grab(path, idx, work_w):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"read failed: {path} frame {idx}")
    h, w = f.shape[:2]
    s = work_w / w
    return cv2.resize(f, (work_w, int(round(h * s))), interpolation=cv2.INTER_AREA), (w, h)


def verdict(before, after):
    """Auto pre-screen only; human reads the jpgs. 'improved' if after clearly
    below before on the best-structured (last) trusted crop."""
    b = [x for x in before if np.isfinite(x)]
    a = [x for x in after if np.isfinite(x)]
    if not a or not b:
        return "no_edges"
    if np.mean(a) < np.mean(b) - 0.5:
        return "improved"
    if np.mean(a) > np.mean(b) + 0.5:
        return "worse"
    return "marginal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--rois", required=True)
    ap.add_argument("--cotracker", default=None, help="CoTracker warped_clean.mp4 for comparison")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work_w", type=int, default=1920)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rois = rl.load_rois(args.rois)
    prov = RomaProvider()
    stats = []

    for fr in FRAMES:
        orig, (fw, fh) = grab(args.original, fr, args.work_w)
        clean, _ = grab(args.clean, fr, args.work_w)
        H, W = orig.shape[:2]
        scale = args.work_w / fw          # full-res ROI coords -> working coords

        kA, kB, cert = prov.match(orig, clean)          # kA=orig, kB=clean
        in_roi = rl.points_in_rois(kA, rois, scale=scale)
        keep = in_roi & (cert >= CERT_THRESH)

        rec = {"frame": fr, "raw_matches": int(len(kA)),
               "roi_filtered_matches": int(keep.sum()),
               "ransac_inliers_similarity": 0, "ransac_inliers_affine": 0,
               "median_reprojection_error": None, "inlier_hull_coverage": 0.0,
               "trusted_crop_chamfer_before": {}, "trusted_crop_chamfer_after": {},
               "cotracker_crop_chamfer": {}, "visual_verdict": "unknown"}

        # match viz (all vs ROI-filtered)
        cv2.imwrite(f"{args.out}/f{fr:03d}_matches_all.jpg",
                    dg.draw_matches(orig, clean, kA, kB, keep=None))
        cv2.imwrite(f"{args.out}/f{fr:03d}_matches_filtered.jpg",
                    dg.draw_matches(orig, clean, kA, kB, keep=keep))

        if keep.sum() < 8:
            rec["visual_verdict"] = "too_few_matches"
            stats.append(rec)
            print(f"frame {fr}: only {int(keep.sum())} ROI matches — skipping fit")
            continue

        src, dst = kA[keep], kB[keep]                   # orig -> clean
        Msim, inl_s = wf.fit_similarity(src, dst)
        Maff, inl_a = wf.fit_affine(src, dst)
        rec["ransac_inliers_similarity"] = int(inl_s.sum())
        rec["ransac_inliers_affine"] = int(inl_a.sum())
        rec["median_reprojection_error"] = wf.reproj_error(Maff, src, dst, inl_a)
        rec["inlier_hull_coverage"] = wf.hull_coverage(src[inl_a] if inl_a.any() else src, H, W)

        cv2.imwrite(f"{args.out}/f{fr:03d}_inliers_similarity.jpg",
                    dg.draw_matches(orig, clean, src, dst, keep=inl_s))
        cv2.imwrite(f"{args.out}/f{fr:03d}_inliers_affine.jpg",
                    dg.draw_matches(orig, clean, src, dst, keep=inl_a))

        # warp clean -> original space with the affine backward map
        mx, my = g.affine_to_backward_map(Maff, H, W)
        warped = g.apply_backward_map(clean, mx, my)

        # CoTracker comparison frame (its own working space), if provided
        ct = None
        if args.cotracker and os.path.exists(args.cotracker):
            ct, _ = grab(args.cotracker, fr, args.work_w)

        before_list, after_list, rows = [], [], []
        for r in rois:
            box = (r["x"] * scale, r["y"] * scale, r["w"] * scale, r["h"] * scale)
            oc = dg.crop(orig, box)
            cc = dg.crop(clean, box)
            wc = dg.crop(warped, box)
            cb = dg.edge_chamfer(oc, cc)
            ca = dg.edge_chamfer(oc, wc)
            rec["trusted_crop_chamfer_before"][r["name"]] = cb
            rec["trusted_crop_chamfer_after"][r["name"]] = ca
            before_list.append(cb); after_list.append(ca)
            panels = [dg.label(dg.edge_overlay(oc, cc), f"{r['name']} BEFORE chamfer={cb:.2f}"),
                      dg.label(dg.edge_overlay(oc, wc), f"RoMa AFTER chamfer={ca:.2f}")]
            if ct is not None:
                ctc = dg.crop(ct, box)
                cct = dg.edge_chamfer(oc, ctc)
                rec["cotracker_crop_chamfer"][r["name"]] = cct
                panels.append(dg.label(dg.edge_overlay(oc, ctc), f"CoTracker chamfer={cct:.2f}"))
            rows.append(dg.hstack_pad(panels, pad=8))
        cv2.imwrite(f"{args.out}/f{fr:03d}_crop_edges_before_after.jpg", dg.vstack_pad(rows, pad=8))

        rec["visual_verdict"] = verdict(before_list, after_list)
        stats.append(rec)
        print(f"frame {fr}: matches {len(kA)}->{int(keep.sum())} roi | "
              f"inliers sim/aff {int(inl_s.sum())}/{int(inl_a.sum())} | "
              f"reproj {rec['median_reprojection_error']:.2f}px | "
              f"chamfer before {np.nanmean(before_list):.2f} -> after {np.nanmean(after_list):.2f} | "
              f"{rec['visual_verdict']}")

    with open(f"{args.out}/roma_match_stats.json", "w") as fp:
        json.dump(stats, fp, indent=2, default=lambda o: None if o == float("inf") else o)
    print(f"\nwrote {args.out}/roma_match_stats.json and per-frame jpgs")


if __name__ == "__main__":
    sys.exit(main())

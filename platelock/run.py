"""
platelock.run — config-driven CLI for the tracking-based plate-lock pipeline
(CoTracker3 multi-keyframe same-domain tracking + sparse cross-domain
bootstrap, validated end-to-end on the ExampleShot shot during development
under the working name "multikey13").

    python -m platelock.run --config platelock/configs/tracking_multikey13.yaml \
        --out output/my_run [--export-5k] [--force]

Stages checkpoint to `{out}/pairs.npz`, `{out}/params_raw.npz`,
`{out}/params.npz`, `{out}/microlock.npz` -- each is skipped and reloaded if
its file already exists (post-hoc parameter tuning on saved decomposed
params is ~free; re-tracking/re-reconstructing is the expensive part). Pass
--force to recompute everything from scratch.
"""
import argparse
import os

import cv2
import numpy as np
from scipy.signal import savgol_filter

from platelock import dense_field as df, geometry as g, masks as pm
from platelock.run_dense_cluster import load_config, grab
from platelock.tracking import track as tk, reconstruct as rc, microlock as ml, hole_fill as hf


def _native_working(cfg):
    probe = cv2.VideoCapture(cfg["inputs"]["original"])
    native_W = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_H = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    working_W = cfg["working"]["width"]
    working_H = int(round(native_H * (working_W / native_W)))
    return native_W, native_H, working_W, working_H, T


def _mask_path(cfg, fr, side):
    return f"{cfg['masks']['dir']}/{side}_mask_f{fr:04d}.png"


def stage_a(cfg, out, working_W, working_H, T, force):
    path = f"{out}/pairs.npz"
    if os.path.exists(path) and not force:
        print(f"[stage A] reusing {path}", flush=True)
        return dict(np.load(path))
    pairs = tk.track_keyframes(cfg, working_W, working_H, T)
    np.savez_compressed(path, **pairs)
    return pairs


def stage_b(cfg, out, pairs, working_W, working_H, force):
    raw_path, params_path = f"{out}/params_raw.npz", f"{out}/params.npz"
    rcfg = cfg["reconstruct"]
    if os.path.exists(raw_path) and not force:
        print(f"[stage B] reusing {raw_path}", flush=True)
        d = np.load(raw_path)
        Ms, creses = d["M"], d["cres"]
    else:
        Ms, creses = rc.reconstruct_sequence(pairs, working_W, working_H, rcfg)
        np.savez_compressed(raw_path, M=Ms, cres=creses)

    if os.path.exists(params_path) and not force:
        print(f"[stage B] reusing {params_path}", flush=True)
        d = np.load(params_path)
        return d["M"], d["cres"]

    def grab_orig(fr):
        return grab(cfg["inputs"]["original"], fr, working_W)

    def grab_clean(fr):
        return grab(cfg["inputs"]["clean"], fr, working_W)

    def mask_loader(fr):
        return pm.load_mask(_mask_path(cfg, fr, "orig"))

    Mt, ct = rc.smooth_temporal(Ms, creses, working_W, working_H, rcfg,
                                grab_orig=grab_orig, grab_clean=grab_clean, mask_loader=mask_loader)
    np.savez_compressed(params_path, M=Mt, cres=ct)
    return Mt, ct


def stage_c(cfg, out, Ms, creses, working_W, working_H, T, force):
    micro_path = f"{out}/microlock.npz"
    mcfg = cfg["microlock"]
    coarse = tuple(mcfg.get("coarse_grid", [48, 26]))
    pyramid = [tuple(p) for p in mcfg["pyramid"]]

    if os.path.exists(micro_path) and not force:
        print(f"[stage C micro-lock] reusing {micro_path}", flush=True)
        return np.load(micro_path)["ML"]

    ML = np.zeros((T, coarse[1], coarse[0], 2), np.float32)
    for fr in range(T):
        O = grab(cfg["inputs"]["original"], fr, working_W)
        C = grab(cfg["inputs"]["clean"], fr, working_W)
        mask0 = pm.load_mask(_mask_path(cfg, fr, "orig"))
        cmask = pm.load_mask(_mask_path(cfg, fr, "clean")).astype(np.float32)
        bx, by = df.assemble_map(Ms[fr], creses[fr], working_H, working_W)
        hole = cv2.dilate(mask0.astype(np.uint8),
                          np.ones((mcfg["hole_dilate_px"], mcfg["hole_dilate_px"]), np.uint8)).astype(bool)
        ML[fr] = ml.microlock_frame(O, C, bx, by, mask0, cmask, working_W, working_H, pyramid,
                                    clamp=mcfg["clamp_px"], hole=hole, coarse=coarse,
                                    damping=mcfg["damping"], min_score=mcfg["min_match_score"],
                                    min_std=mcfg["min_tile_std"], curv_gate=mcfg["curvature_gate"])
        if fr % 25 == 0:
            print(f"micro-lock f{fr}", flush=True)
    ML = savgol_filter(ML, mcfg["temporal_window"], 2, axis=0)
    np.savez_compressed(micro_path, ML=ML)
    return ML


def build_final_map_fn(cfg, Ms, creses, ML, working_W, working_H, T, pairs=None):
    """Compose base field - microlock, then the configured hole fill
    (`hole_fill.method`: harmonic [default] | ring_plane), returning a
    callable final_map(fr) -> (mx, my) at working resolution."""
    hcfg = cfg["hole_fill"]
    clamp = cfg["microlock"]["clamp_px"]

    # Opt-in perimeter clamp (the project lead 2026-07-17): beyond the per-frame track
    # coverage the cres residual AND the microlock correction fall back to the
    # affine prior / zero -- full perimeter (corners + all four edges), border-
    # restricted so hole-fill interior is untouched. Supersedes edge_clamp.
    pcfg = cfg.get("perimeter_clamp", {})
    ecfg = cfg.get("edge_clamp", {})
    if pcfg.get("enabled", False):
        if pairs is None:
            raise ValueError("perimeter_clamp.enabled requires pairs (track data)")
        if ecfg.get("enabled", False):
            print("[perimeter clamp] supersedes edge_clamp", flush=True)
        grids = [(creses.shape[2], creses.shape[1]), (ML.shape[2], ML.shape[1])]
        wc, wm = df.coverage_weights_perimeter(
            pairs["trO"], pairs["vi"], T, grids, working_W, working_H,
            track_res=(pairs["track_res"] if "track_res" in pairs else None),
            dist_px=pcfg.get("dist_px", 90.0),
            band_px=pcfg.get("band_px", 120.0),
            margin_px=pcfg.get("margin_px", 240.0),
            temporal_window=pcfg.get("temporal_window", 11),
            cov_k=pcfg.get("cov_k", 1))
        creses = df.damp_lattice(creses, wc)
        ml_damped = pcfg.get("damp_microlock", True)
        if ml_damped:
            # ML is otherwise applied unclamped to the very frame edge, matching
            # against clean content that is itself extrapolation there.
            ML = df.damp_lattice(ML, wm)
        print(f"[perimeter clamp] cres damped f0={np.mean(wc[0] < 0.99):.1%} "
              f"f{T-1}={np.mean(wc[-1] < 0.99):.1%} min_w={wc.min():.3f} "
              f"ml_damped={ml_damped}", flush=True)

    # Opt-in edge clamp (the project lead 2026-07-16): beyond the per-frame right-hand
    # track-coverage boundary the field falls back to the affine prior.
    elif ecfg.get("enabled", False):
        if pairs is None:
            raise ValueError("edge_clamp.enabled requires pairs (track data)")
        xcov = df.coverage_x_right(pairs["trO"], pairs["vi"], T,
                                   percentile=ecfg.get("cov_percentile", 99.5),
                                   temporal_window=ecfg.get("temporal_window", 11))
        band = ecfg.get("band_px", 120)
        creses = df.damp_cres_beyond_coverage(creses, xcov, working_W, band)
        print(f"[edge clamp] xcov f0={xcov[0]:.0f} f{T-1}={xcov[-1]:.0f} "
              f"band={band}px W={working_W}", flush=True)

    def base_map(fr):
        mx, my = df.assemble_map(Ms[fr], creses[fr], working_H, working_W)
        full = cv2.resize(np.clip(ML[fr], -clamp, clamp), (working_W, working_H),
                          interpolation=cv2.INTER_CUBIC)
        return mx - full[..., 0], my - full[..., 1]

    if hcfg.get("method", "harmonic") == "harmonic":
        # v3 hybrid: exact per-frame harmonic seam + depth-ramped TEMPORALLY
        # smoothed interior over a savgol-smoothed ring-plane prior. Pre-pass
        # solves the /4 residuals for the whole shot, then savgol over time.
        ds = hcfg.get("solve_downsample", 4)
        seam = hcfg.get("seam_px", 16)
        r0 = hcfg.get("interior_ramp_start_px", 24)
        rl = hcfg.get("interior_ramp_len_px", 90)
        twin = hcfg.get("interior_temporal_window", 11)
        h4, w4 = working_H // ds, working_W // ds
        raw = []
        for fr in range(T):
            mask = pm.load_mask(_mask_path(cfg, fr, "orig"))
            _, ring, _ = hf.ring_regions(mask, hcfg["inner_dilate_px"])
            mx, my = base_map(fr)
            c = hf.fit_ring_plane(mx, my, ring, hcfg.get("ring_samples", 4000))
            raw.append(c if c is not None else np.zeros((2, 3)))
        priors = hf.smooth_coefficients(np.stack(raw), hcfg.get("coeff_temporal_window", 9))
        yy, xx = np.mgrid[0:working_H, 0:working_W].astype(np.float32)
        sols = np.zeros((T, h4, w4, 2), np.float32)
        for fr in range(T):
            mask = pm.load_mask(_mask_path(cfg, fr, "orig"))
            inner = cv2.dilate(mask.astype(np.uint8),
                               np.ones((hcfg["inner_dilate_px"],) * 2, np.uint8)).astype(bool)
            mx, my = base_map(fr)
            c = priors[fr]
            px = xx * c[0, 0] + yy * c[0, 1] + c[0, 2]
            py = xx * c[1, 0] + yy * c[1, 1] + c[1, 2]
            r4 = [cv2.resize(r, (w4, h4), interpolation=cv2.INTER_AREA)
                  for r in (mx - px, my - py)]
            reg4 = cv2.resize(inner.astype(np.uint8), (w4, h4),
                              interpolation=cv2.INTER_NEAREST) > 0
            s4 = hf.harmonic_extend(r4, reg4)
            sols[fr, ..., 0], sols[fr, ..., 1] = s4[0], s4[1]
            if fr % 25 == 0:
                print(f"[hole fill] harmonic solve f{fr}", flush=True)
        sols_sm = savgol_filter(sols, twin, 2, axis=0)

        def final_map(fr):
            mask = pm.load_mask(_mask_path(cfg, fr, "orig"))
            inner = cv2.dilate(mask.astype(np.uint8),
                               np.ones((hcfg["inner_dilate_px"],) * 2, np.uint8)).astype(bool)
            mx, my = base_map(fr)
            c = priors[fr]
            px = xx * c[0, 0] + yy * c[0, 1] + c[0, 2]
            py = xx * c[1, 0] + yy * c[1, 1] + c[1, 2]
            dt = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
            wdeep = np.clip((dt - r0) / rl, 0, 1)[::ds, ::ds, None]
            sel = (1 - wdeep) * sols[fr] + wdeep * sols_sm[fr]
            su = [cv2.resize(sel[..., ch], (working_W, working_H),
                             interpolation=cv2.INTER_CUBIC) for ch in (0, 1)]
            wseam = np.clip(dt / seam, 0, 1)
            return ((wseam * (px + su[0]) + (1 - wseam) * mx).astype(np.float32),
                    (wseam * (py + su[1]) + (1 - wseam) * my).astype(np.float32))

        return final_map

    # ring_plane family. ring_model ("affine" default | "homography") and
    # per_component (False default | True) are opt-in upgrades -- with BOTH
    # at their defaults this branch must reproduce the original ring_plane
    # fill bit-for-bit (see the hole_fill.py module docstring / the
    # BIT-IDENTITY GATE that guarded this refactor). feather_px is genuinely
    # read from config (not hardcoded) for every combination below.
    ring_model = hcfg.get("ring_model", "affine")
    per_component = hcfg.get("per_component", False)
    feather = hcfg.get("feather_px", 180)

    if not per_component:
        coefs, rings = [], []
        for fr in range(T):
            mask = pm.load_mask(_mask_path(cfg, fr, "orig"))
            inner, ring, outer = hf.ring_regions(mask, hcfg["inner_dilate_px"], hcfg["outer_dilate_px"])
            mx, my = base_map(fr)
            if ring_model == "affine":
                coef = hf.fit_ring_plane(mx, my, ring, hcfg["ring_samples"])
                coefs.append(coef if coef is not None else np.zeros((2, 3)))
            else:
                coef = hf.fit_ring_homography(mx, my, ring, hcfg["ring_samples"])
                coefs.append(coef if coef is not None else np.eye(3))
            rings.append((inner, outer))
        coefs = np.stack(coefs)
        if ring_model == "affine":
            coefs = hf.smooth_coefficients(coefs, hcfg["coeff_temporal_window"])
        else:
            coefs = hf.smooth_homographies(coefs, hcfg["coeff_temporal_window"])

        def final_map(fr):
            mx, my = base_map(fr)
            inner, outer = rings[fr]
            if ring_model == "affine":
                return hf.apply_plane(mx, my, inner, outer, coefs[fr], feather)
            return hf.apply_homography(mx, my, inner, outer, coefs[fr], feather)

        return final_map

    # per_component=True: each disconnected mask component gets its own fit,
    # its own feathered application, and its own temporal smoothing TRACK
    # (component identity is not stable frame-to-frame -- see
    # hf.build_component_tracks docstring).
    labels_seq, regions_seq = [], []
    for fr in range(T):
        mask = pm.load_mask(_mask_path(cfg, fr, "orig"))
        labels, _, regions = hf.component_regions(mask, hcfg["inner_dilate_px"], hcfg["outer_dilate_px"])
        labels_seq.append(labels)
        regions_seq.append(regions)

    tracks = hf.build_component_tracks(labels_seq)
    print(f"[hole fill] {len(tracks)} component tracks found", flush=True)

    def per_frame_fit_fn(fr, lid):
        mx, my = base_map(fr)
        _, ring, _ = regions_seq[fr][lid]
        if ring_model == "affine":
            c = hf.fit_ring_plane(mx, my, ring, hcfg["ring_samples"])
            return c if c is not None else np.zeros((2, 3))
        c = hf.fit_ring_homography(mx, my, ring, hcfg["ring_samples"])
        return c if c is not None else np.eye(3)

    frame_components = {fr: [] for fr in range(T)}
    for track in tracks:
        smoothed = hf.fit_and_smooth_tracks(track, per_frame_fit_fn, hcfg["coeff_temporal_window"])
        for i, (fr, lid) in enumerate(zip(track["frames"], track["labels"])):
            frame_components[fr].append((regions_seq[fr][lid], smoothed[i]))

    def final_map(fr):
        mx, my = base_map(fr)
        comps = frame_components[fr]
        if not comps:
            return mx, my
        regions_list = [r for r, c in comps]
        coefs_list = [c for r, c in comps]
        return hf.apply_components(mx, my, regions_list, coefs_list, model=ring_model, feather_px=feather)

    return final_map


def render_overlay(cfg, out, final_map, working_W, working_H, T):
    path = f"{out}/overlay50.mp4"
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (working_W, working_H))
    for fr in range(T):
        O = grab(cfg["inputs"]["original"], fr, working_W)
        C = grab(cfg["inputs"]["clean"], fr, working_W)
        mx, my = final_map(fr)
        wa = g.apply_backward_map(C, mx, my)
        vw.write(cv2.addWeighted(O, 0.5, wa, 0.5, 0))
    vw.release()
    print(f"wrote {path}", flush=True)


def export_5k(cfg, out, final_map, T):
    native_W, native_H, _, _, _ = _native_working(cfg)
    for sub in ("warped", "stmaps"):
        os.makedirs(f"{out}/5k/{sub}", exist_ok=True)
    for fr in range(T):
        mx_w, my_w = final_map(fr)
        mx_5k, my_5k = g.upscale_backward_map(mx_w, my_w, native_H, native_W)
        C_native = grab(cfg["inputs"]["clean"], fr, native_W)
        warped = g.apply_backward_map(C_native, mx_5k, my_5k)
        cv2.imwrite(f"{out}/5k/warped/warped_f{fr:04d}.png", warped)
        stmap = g.backward_map_to_stmap(mx_5k, my_5k, native_W, native_H)
        g.write_stmap_exr(stmap, f"{out}/5k/stmaps/stmap_f{fr:04d}.exr")
        if fr % 25 == 0:
            print(f"5k export f{fr}", flush=True)
    print(f"wrote {out}/5k/", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Config-driven tracking-based plate-lock pipeline.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-5k", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    native_W, native_H, working_W, working_H, T = _native_working(cfg)
    print(f"native {native_W}x{native_H}  working {working_W}x{working_H}  frames {T}", flush=True)

    pairs = stage_a(cfg, args.out, working_W, working_H, T, args.force)
    Ms, creses = stage_b(cfg, args.out, pairs, working_W, working_H, args.force)
    ML = stage_c(cfg, args.out, Ms, creses, working_W, working_H, T, args.force)
    final_map = build_final_map_fn(cfg, Ms, creses, ML, working_W, working_H, T, pairs=pairs)
    render_overlay(cfg, args.out, final_map, working_W, working_H, T)
    if args.export_5k:
        export_5k(cfg, args.out, final_map, T)


if __name__ == "__main__":
    main()

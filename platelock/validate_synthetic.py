"""
platelock.validate_synthetic — synthetic ground-truth proof of the warp/STMap
machinery in platelock.geometry.

The whole pipeline's trust rests on the BACKWARD-map direction being correct
(see geometry.py). Every past "good" plate-lock result was judged on lying
full-frame metrics; this suite instead proves, on known transforms with known
answers, that:

    apply_backward_map(clean, M_bwd)  reconstructs  orig
    stmap round-trips exactly
    EXR write/read survives

Run:  python3 platelock/validate_synthetic.py   (exit 0 = all pass, 1 = any fail)

Construction identity (holds for ANY affine M_bwd):
    clean = cv2.warpAffine(orig, M_bwd, (W,H))     # default => clean(q)=orig(M_bwd^-1 q)
    apply_backward_map(clean, map_from(M_bwd))(p) = clean(M_bwd p) = orig(p)   OK
"""

import os
import sys
import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator

from platelock import geometry as g

H, W = 540, 960
BORDER = 40          # ignore this margin (extrapolation/edge artefacts)
TOL_UINT8 = 3.0      # mean-abs-err tolerance on inner crop
TOL_MAP = 1e-3


def make_test_image(H, W):
    """Textured pattern with sharp, recoverable edges."""
    img = np.full((H, W), 30, np.uint8)
    img[::32, :] = 200
    img[:, ::32] = 200
    rng = np.random.default_rng(0)
    for _ in range(12):
        x, y = int(rng.integers(60, W - 120)), int(rng.integers(60, H - 120))
        w, h = int(rng.integers(40, 110)), int(rng.integers(40, 110))
        cv2.rectangle(img, (x, y), (x + w, y + h), int(rng.integers(90, 255)), -1)
    for _ in range(10):
        x, y = int(rng.integers(60, W - 60)), int(rng.integers(60, H - 60))
        cv2.circle(img, (x, y), int(rng.integers(15, 45)), int(rng.integers(90, 255)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def inner_err(a, b):
    aa = a[BORDER:-BORDER, BORDER:-BORDER].astype(np.int32)
    bb = b[BORDER:-BORDER, BORDER:-BORDER].astype(np.int32)
    return float(np.abs(aa - bb).mean())


def stmap_roundtrip(mx, my):
    worst = 0.0
    for origin in ("bottom-left", "top-left"):
        st = g.backward_map_to_stmap(mx, my, W, H, origin=origin)
        rx, ry = g.stmap_to_backward_map(st, W, H, origin=origin)
        worst = max(worst, float(np.abs(rx - mx).max()), float(np.abs(ry - my).max()))
    return worst


def affine_case(M_bwd, orig):
    M_bwd = np.asarray(M_bwd, np.float32)
    clean = cv2.warpAffine(orig, M_bwd, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    mx, my = g.affine_to_backward_map(M_bwd, H, W)
    recovered = g.apply_backward_map(clean, mx, my)
    return inner_err(recovered, orig), stmap_roundtrip(mx, my)


def tps_case(orig):
    """Smooth non-affine residual (<=5px). Inverse built from negated control
    displacements: clean = orig warped by -residual, recovered by +residual."""
    xs = np.linspace(80, W - 80, 4)
    ys = np.linspace(80, H - 80, 3)
    ctrl = np.array([[x, y] for y in ys for x in xs], np.float32)
    rng = np.random.default_rng(1)
    disp = rng.uniform(-5, 5, ctrl.shape).astype(np.float32)

    gy, gx = np.mgrid[0:H, 0:W]
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)

    def build_map(sign):
        rbf = RBFInterpolator(ctrl, sign * disp, kernel="thin_plate_spline", smoothing=0.0)
        d = rbf(grid).astype(np.float32)
        mx = (grid[:, 0] + d[:, 0]).reshape(H, W).astype(np.float32)
        my = (grid[:, 1] + d[:, 1]).reshape(H, W).astype(np.float32)
        return mx, my

    bwd_x, bwd_y = build_map(+1.0)
    inv_x, inv_y = build_map(-1.0)
    clean = g.apply_backward_map(orig, inv_x, inv_y)
    recovered = g.apply_backward_map(clean, bwd_x, bwd_y)
    return inner_err(recovered, orig), stmap_roundtrip(bwd_x, bwd_y)


def exr_roundtrip():
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "platelock_synth_stmap.exr")
    mx, my = g.identity_map(H, W)
    st = g.backward_map_to_stmap(mx, my, W, H, "bottom-left")
    g.write_stmap_exr(st, tmp)
    back = cv2.imread(tmp, cv2.IMREAD_UNCHANGED)
    if back is None:
        return False, "read None"
    ok = back.shape[:2] == (H, W) and back.dtype == np.float32
    return ok, f"{back.shape}/{back.dtype}"


def main():
    orig = make_test_image(H, W)
    cx, cy = W / 2.0, H / 2.0

    def scale_c(s):
        return np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])

    ang = np.deg2rad(2.0); s = 1.03
    ca, sa = np.cos(ang) * s, np.sin(ang) * s
    rot = np.float32([[ca, -sa, cx - (ca * cx - sa * cy) + 8],
                      [sa,  ca, cy - (sa * cx + ca * cy) - 6]])

    cases = [
        ("identity",        lambda: affine_case([[1, 0, 0], [0, 1, 0]], orig)),
        ("translate+20x",   lambda: affine_case([[1, 0, 20], [0, 1, 0]], orig)),
        ("translate-15y",   lambda: affine_case([[1, 0, 0], [0, 1, -15]], orig)),
        ("translate+20+10", lambda: affine_case([[1, 0, 20], [0, 1, 10]], orig)),
        ("scale1.05",       lambda: affine_case(scale_c(1.05), orig)),
        ("scale0.95",       lambda: affine_case(scale_c(0.95), orig)),
        ("affine_rot2deg",  lambda: affine_case(rot, orig)),
        ("tps_mesh",        lambda: tps_case(orig)),
    ]

    print(f"{'transform':18} {'mean_abs_err':>12} {'map_roundtrip':>14}  result")
    print("-" * 56)
    all_pass = True
    for name, fn in cases:
        err, rt = fn()
        ok = (err < TOL_UINT8) and (rt < TOL_MAP)
        all_pass &= ok
        print(f"{name:18} {err:12.4f} {rt:14.2e}  {'PASS' if ok else 'FAIL'}")

    exr_ok, exr_info = exr_roundtrip()
    all_pass &= exr_ok
    print(f"{'exr_write_read':18} {'':>12} {exr_info:>14}  {'PASS' if exr_ok else 'FAIL'}")
    print("-" * 56)

    if all_pass:
        print("ALL SYNTHETIC TESTS PASSED")
        return 0
    print("SYNTHETIC TESTS FAILED — do not trust downstream warp/STMap")
    return 1


if __name__ == "__main__":
    sys.exit(main())

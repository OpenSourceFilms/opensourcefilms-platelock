"""
platelock.geometry — canonical BACKWARD-warp / STMap conventions.

DIRECTION CONVENTION (single source of truth for the whole pipeline)
-------------------------------------------------------------------
A per-frame warp is a BACKWARD map. For each OUTPUT pixel (x, y) in
ORIGINAL-plate space, the map gives the SOURCE coordinate in the CLEAN
plate that should be sampled:

    map_x[y, x] = source column in clean plate
    map_y[y, x] = source row    in clean plate

Rendering the locked clean plate is therefore:

    warped_clean = cv2.remap(clean, map_x, map_y)   # lives in original space

STMap convention
----------------
    R = map_x / (W - 1)
    G = map_y / (H - 1)                     # origin='top-left'
    G = 1 - map_y / (H - 1)                 # origin='bottom-left' (Nuke)

All map arrays are float32. STMaps are float32 HxWx2 (or HxWx3 for EXR,
B channel = 0).
"""

import os
import numpy as np
import cv2

try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except Exception:
    _HAS_OPENEXR = False


# ---------------------------------------------------------------------------
# Backward maps
# ---------------------------------------------------------------------------

def identity_map(H, W):
    """Identity BACKWARD map: map_x[y,x]=x, map_y[y,x]=y. Returns float32 HxW."""
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def apply_backward_map(src_img, map_x, map_y):
    """Sample src_img with a BACKWARD map.

    For output pixel (x,y), samples src_img at (map_x[y,x], map_y[y,x]).
    Preserves dtype (uint8 or float32). BORDER_REPLICATE avoids black edges.
    """
    return cv2.remap(
        src_img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def affine_to_backward_map(M, H, W):
    """2x3 affine -> BACKWARD map.

    M maps OUTPUT(original) coords -> SOURCE(clean) coords:
        [sx, sy]^T = M @ [x, y, 1]^T
    so the backward map at (x,y) is simply M applied to the output grid.
    """
    M = np.asarray(M, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    map_x = M[0, 0] * gx + M[0, 1] * gy + M[0, 2]
    map_y = M[1, 0] * gx + M[1, 1] * gy + M[1, 2]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def upscale_backward_map(map_x, map_y, new_H, new_W):
    """Upscale a BACKWARD map to a higher output resolution (e.g. working-res
    solve -> 5K delivery). Resizing the grid alone is NOT enough: map_x/map_y
    store absolute pixel coordinates into the SOURCE (clean) frame, so the
    coordinate VALUES must also be rescaled by the same factor the source
    frame itself is scaled by, or the map points at the wrong source pixels.

    cv2.resize samples under a half-pixel-center convention
    (src = (dst+0.5)*scale - 0.5), but this codebase's maps are built via
    plain np.arange (index == coordinate, no half-pixel offset). The two
    conventions differ by a constant bias of 0.5*(1-scale) in the resized+
    scaled result — exact for the affine part of the map, negligible
    (sub-pixel) for the smooth TPS residual part. Corrected below; verified
    against both identity and a nonlinear synthetic field."""
    old_H, old_W = map_x.shape
    sx = new_W / old_W
    sy = new_H / old_H
    mx = cv2.resize(map_x.astype(np.float32), (new_W, new_H), interpolation=cv2.INTER_CUBIC) * sx
    my = cv2.resize(map_y.astype(np.float32), (new_W, new_H), interpolation=cv2.INTER_CUBIC) * sy
    mx += 0.5 * (sx - 1)
    my += 0.5 * (sy - 1)
    return mx.astype(np.float32), my.astype(np.float32)


def compose_backward_maps(outer_x, outer_y, inner_x, inner_y):
    """Compose two BACKWARD maps: result(x,y) = inner( outer(x,y) ).

    `outer` maps output-space -> intermediate-space; `inner` maps
    intermediate-space -> source-space. Used to chain a global affine
    (outer) with a residual field (inner). Implemented by sampling the
    inner map arrays at the coordinates given by the outer map.
    """
    comp_x = cv2.remap(inner_x.astype(np.float32),
                       outer_x.astype(np.float32), outer_y.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    comp_y = cv2.remap(inner_y.astype(np.float32),
                       outer_x.astype(np.float32), outer_y.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return comp_x.astype(np.float32), comp_y.astype(np.float32)


# ---------------------------------------------------------------------------
# STMap <-> backward map
# ---------------------------------------------------------------------------

def backward_map_to_stmap(map_x, map_y, W, H, origin="bottom-left"):
    """BACKWARD map -> normalized STMap (HxWx2 float32).

    R = map_x/(W-1); G = map_y/(H-1) [top-left] or 1-map_y/(H-1) [bottom-left].
    """
    r = map_x.astype(np.float32) / float(W - 1)
    g = map_y.astype(np.float32) / float(H - 1)
    if origin == "bottom-left":
        g = 1.0 - g
    elif origin != "top-left":
        raise ValueError(f"origin must be 'top-left' or 'bottom-left', got {origin!r}")
    return np.stack([r, g], axis=-1).astype(np.float32)


def stmap_to_backward_map(stmap, W, H, origin="bottom-left"):
    """Normalized STMap -> BACKWARD map. Exact inverse of backward_map_to_stmap."""
    r = stmap[..., 0].astype(np.float32)
    g = stmap[..., 1].astype(np.float32)
    map_x = r * float(W - 1)
    if origin == "bottom-left":
        map_y = (1.0 - g) * float(H - 1)
    elif origin == "top-left":
        map_y = g * float(H - 1)
    else:
        raise ValueError(f"origin must be 'top-left' or 'bottom-left', got {origin!r}")
    return map_x.astype(np.float32), map_y.astype(np.float32)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _to_exr_rgb(stmap):
    """Return contiguous float32 R,G,B channels (B=0 if stmap is 2-channel)."""
    R = np.ascontiguousarray(stmap[..., 0].astype(np.float32))
    G = np.ascontiguousarray(stmap[..., 1].astype(np.float32))
    if stmap.shape[-1] >= 3:
        B = np.ascontiguousarray(stmap[..., 2].astype(np.float32))
    else:
        B = np.zeros_like(R)
    return R, G, B


def write_stmap_exr(stmap, path):
    """Write a 32-bit float EXR STMap (R=U, G=V, B=0).

    Tries OpenEXR+Imath, falls back to cv2.imwrite. Raises if neither works.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    R, G, B = _to_exr_rgb(stmap)
    H, W = stmap.shape[:2]

    if _HAS_OPENEXR:
        header = OpenEXR.Header(W, H)
        fchan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header["channels"] = {"R": fchan, "G": fchan, "B": fchan}
        out = OpenEXR.OutputFile(path, header)
        try:
            out.writePixels({"R": R.tobytes(), "G": G.tobytes(), "B": B.tobytes()})
        finally:
            out.close()
        return

    # Fallback: OpenCV EXR (needs OPENCV_IO_ENABLE_OPENEXR=1 in some builds).
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    img_bgr = np.stack([B, G, R], axis=-1).astype(np.float32)
    if not cv2.imwrite(path, img_bgr):
        raise IOError(
            f"Failed to write EXR to {path}: no OpenEXR module and cv2 EXR "
            f"support unavailable (set OPENCV_IO_ENABLE_OPENEXR=1 or install OpenEXR)."
        )


def write_stmap_npy(stmap, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.save(path, stmap.astype(np.float32))


def read_stmap_npy(path):
    return np.load(path).astype(np.float32)


if __name__ == "__main__":
    # In-memory sanity: identity + stmap round-trips (both origins).
    H, W = 8, 8
    mx, my = identity_map(H, W)
    img = np.random.randint(0, 255, (H, W), np.uint8)
    assert np.array_equal(apply_backward_map(img, mx, my), img)
    for origin in ("bottom-left", "top-left"):
        st = backward_map_to_stmap(mx, my, W, H, origin=origin)
        rx, ry = stmap_to_backward_map(st, W, H, origin=origin)
        assert np.allclose(rx, mx, atol=1e-4) and np.allclose(ry, my, atol=1e-4), origin
    print("geometry.py self-test passed.")

"""
platelock.correspondence.roma_provider — RoMa dense matcher (Provider B).

Wraps romatch.roma_outdoor (the real RoMa; the PyPI `romav2` package is an
unrelated namesake). Returns sparse high-certainty correspondences between the
ORIGINAL and CLEAN plates in working-resolution pixel coords:

    kA = points in ORIGINAL,  kB = matched points in CLEAN,  cert = certainty

For the BACKWARD-map fit (geometry.py), pass from=kA (orig) -> to=kB (clean).

NOTE: use_custom_corr=False is required — the CUDA local_corr extension is not
compiled here, and the True default silently produces identity warps (past bug).
"""

import numpy as np
import cv2
from PIL import Image


class RomaProvider:
    name = "roma"

    def __init__(self, device=None, coarse_res=560, upsample_res=864):
        import torch
        from romatch import roma_outdoor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = roma_outdoor(
            device=self.device, use_custom_corr=False,
            coarse_res=coarse_res, upsample_res=upsample_res)

    def match(self, orig_bgr, clean_bgr, num=10000):
        """Return (kA_orig Nx2, kB_clean Nx2, cert N) in working pixel coords."""
        H, W = orig_bgr.shape[:2]
        oi = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        ci = Image.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
        warp, cert = self.model.match(oi, ci, device=self.device)
        matches, mcert = self.model.sample(warp, cert, num=num)
        kA, kB = self.model.to_pixel_coordinates(matches, H, W, H, W)
        return (kA.cpu().numpy().astype(np.float32),
                kB.cpu().numpy().astype(np.float32),
                mcert.cpu().numpy().astype(np.float32))

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from utils.flow_utils import compute_bg_mask

logger = logging.getLogger(__name__)

MOTION_MAP = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


class ECCMethod:
    def __init__(self, mode: str = "homography", num_iterations: int = 200, termination_eps: float = 1e-5):
        if mode not in MOTION_MAP:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(MOTION_MAP.keys())}.")
        self.mode = mode
        self.motion = MOTION_MAP[mode]
        self.num_iterations = num_iterations
        self.termination_eps = termination_eps
        self._criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(num_iterations), float(termination_eps))

    def align_frame(self, original: np.ndarray, clean: np.ndarray,
                    init_warp: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Align clean plate to original using ECC. Returns (warped_clean, transform_matrix, info).

        init_warp: optional warm-start matrix from the previous frame. Critical for sequences
        with large inter-frame drift — ECC's search radius is limited and it will snap to wrong
        local correlation maxima if started from identity when the true displacement is large.
        """
        try:
            gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            gray_clean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        except cv2.error as e:
            logger.warning("Color conversion failed: %s", e)
            return self._fallback(clean)

        # templateMask: identify reliable background in the TEMPLATE (original) domain —
        # pixels where |orig-clean| is small, i.e. not AI-inpainted. ECC only uses
        # gradient within this mask to drive the alignment.
        # inputMask: must be full-white. The same |orig-clean| mask is NOT valid in
        # the clean-frame domain — it is derived at identity alignment, not at the
        # current warp estimate. OpenCV warps inputMask into template frame each
        # iteration and intersects with templateMask, so passing the same mask for
        # both compounds the exclusion incorrectly.
        template_mask = compute_bg_mask(original, clean, threshold=30, dilation_px=20)
        input_mask = np.full_like(template_mask, 255, dtype=np.uint8)
        reliable_frac = template_mask.mean() / 255.0
        logger.debug("ECC bg_mask reliable fraction: %.1f%%", reliable_frac * 100)
        if reliable_frac < 0.2:
            logger.warning("Only %.0f%% reliable pixels for ECC; falling back to identity.", reliable_frac * 100)
            return self._fallback(clean)

        if self.motion == cv2.MOTION_HOMOGRAPHY:
            warp = np.eye(3, dtype=np.float32)
        else:
            warp = np.eye(2, 3, dtype=np.float32)

        try:
            correlation, warp = cv2.findTransformECCWithMask(
                gray_orig, gray_clean, template_mask, input_mask,
                warp, self.motion, self._criteria, 5)
        except cv2.error as e:
            logger.warning("ECC failed: %s", e)
            return self._fallback(clean)

        if correlation is None or np.isnan(correlation):
            logger.warning("ECC did not converge (correlation=%s)", correlation)
            return self._fallback(clean)

        h, w = original.shape[:2]
        # ECC finds warp W s.t. orig ≈ warpAffine(clean, W, WARP_INVERSE_MAP).
        # WARP_INVERSE_MAP treats W as a forward map (dst→src), sampling clean at W*(x,y).
        # Without this flag, W is inverted first, which reverses the correction.
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        if self.motion == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(clean, warp, (w, h), flags=flags)
        else:
            aligned = cv2.warpAffine(clean, warp, (w, h), flags=flags)

        return aligned, warp, {"mode": self.mode, "correlation": float(correlation), "converged": True}

    def align_sequence(self, originals: List[np.ndarray], cleans: List[np.ndarray],
                       show_progress: bool = True,
                       temporal_smooth_kernel: int = 5,
                       max_jump_px: float = 0.5) -> Tuple[List, List, List]:
        if len(originals) != len(cleans):
            raise ValueError("originals and cleans must have the same length")

        # Pass 1: collect raw per-frame transforms
        raw_warps, infos = [], []
        it = range(len(originals))
        if show_progress:
            it = tqdm(it, total=len(originals), desc="ECC alignment")
        for i in it:
            _, t, info = self.align_frame(originals[i], cleans[i])
            raw_warps.append(t)
            infos.append(info)

        # Pass 2: temporally smooth the translation components to suppress per-frame jitter.
        # Raw ECC tx/ty can vary ~1px between adjacent frames (0.63→1.63→0.11) causing
        # visible temporal swimming even when every individual frame has improved PSNR.
        from scipy.signal import medfilt
        import numpy as _np
        n = len(raw_warps)
        if self.motion == cv2.MOTION_TRANSLATION:
            tx = _np.array([w[0, 2] for w in raw_warps], dtype=np.float32)
            ty = _np.array([w[1, 2] for w in raw_warps], dtype=np.float32)
            k = min(temporal_smooth_kernel, n if n % 2 == 1 else n - 1)
            if k >= 3:
                tx = medfilt(tx, kernel_size=k).astype(np.float32)
                ty = medfilt(ty, kernel_size=k).astype(np.float32)
            # clamp frame-to-frame jumps
            for i in range(1, n):
                tx[i] = float(_np.clip(tx[i], tx[i-1] - max_jump_px, tx[i-1] + max_jump_px))
                ty[i] = float(_np.clip(ty[i], ty[i-1] - max_jump_px, ty[i-1] + max_jump_px))
            smoothed_warps = []
            for i, w in enumerate(raw_warps):
                sw = w.copy()
                sw[0, 2] = tx[i]; sw[1, 2] = ty[i]
                smoothed_warps.append(sw)
        else:
            smoothed_warps = raw_warps

        # Pass 3: apply smoothed transforms
        h, w = originals[0].shape[:2]
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        warped = []
        for i, sw in enumerate(smoothed_warps):
            if self.motion == cv2.MOTION_HOMOGRAPHY:
                aligned = cv2.warpPerspective(cleans[i], sw, (w, h), flags=flags)
            else:
                aligned = cv2.warpAffine(cleans[i], sw, (w, h), flags=flags,
                                          borderMode=cv2.BORDER_REPLICATE)
            warped.append(aligned)

        return warped, smoothed_warps, infos

    @staticmethod
    def save_transforms(transforms: List[np.ndarray], path: Path) -> None:
        np.save(str(path), np.array(transforms, dtype=object), allow_pickle=True)

    @staticmethod
    def load_transforms(path: Path) -> List[np.ndarray]:
        return list(np.load(str(path), allow_pickle=True))

    def _fallback(self, clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        identity = np.eye(3, dtype=np.float32) if self.motion == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
        return clean, identity, {"mode": self.mode, "correlation": 0.0, "converged": False}

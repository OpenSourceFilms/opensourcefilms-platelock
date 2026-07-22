"""Masking utilities for plate-lock registration.

Returns uint8 masks: 255 = safe background, 0 = unsafe / AI-changed.
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def diff_mask(orig: np.ndarray, clean: np.ndarray,
              threshold: int = 25, dilation_px: int = 25) -> np.ndarray:
    """diff_mask: 255=safe_bg, 0=AI-changed/dynamic. Uses max-channel diff."""
    diff = np.abs(orig.astype(np.float32) - clean.astype(np.float32)).max(axis=-1)
    changed = (diff > threshold).astype(np.uint8)
    if dilation_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        changed = cv2.dilate(changed, k)
    return np.where(changed, np.uint8(0), np.uint8(255))


def compute_unsafe_mask(
    orig: np.ndarray,
    clean: np.ndarray,
    diff_threshold: int = 25,
    dilation_px: int = 25,
    sam2_cfg: str | None = None,
    sam2_checkpoint: str | None = None,
) -> np.ndarray:
    """Unsafe mask combining diff detection with optional SAM2 actor mask.

    Returns uint8: 255=safe background, 0=unsafe (AI-changed or actor).
    SAM2 is silently skipped if not installed or if cfg/checkpoint are not provided.
    """
    mask = diff_mask(orig, clean, threshold=diff_threshold, dilation_px=dilation_px)

    if not (sam2_cfg and sam2_checkpoint):
        return mask

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2)
        predictor.set_image(orig)

        H, W = orig.shape[:2]
        masks_out, scores, _ = predictor.predict(
            point_coords=np.array([[W // 2, H // 2]], np.float32),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        person = masks_out[np.argmax(scores)].astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (63, 63))
        person = cv2.dilate(person, k)
        # Union of unsafe regions: pixel is unsafe if either source marks it unsafe
        mask = np.minimum(mask, np.where(person, np.uint8(0), np.uint8(255)))
        logger.debug("SAM2 person mask applied")
    except ImportError as e:
        logger.warning("SAM2 not available, using diff_mask only: %s", e)
    except Exception as e:
        logger.warning("SAM2 failed, using diff_mask only: %s", e)

    return mask

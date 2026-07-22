"""
platelock.segmenters.sam3_segmenter — SAM3-based instance segmentation wrapper.
Segmenter interface: BGR image in, bool mask out.
"""

import os

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Default checkpoint location. Override with the PLATELOCK_SAM3_CHECKPOINT
# environment variable, or pass checkpoint_path explicitly.
DEFAULT_SAM3_CHECKPOINT = os.environ.get(
    "PLATELOCK_SAM3_CHECKPOINT", "models/sam3/sam3.pt"
)


class Sam3Segmenter:
    """Segmenter using Meta's SAM3 image model."""

    def __init__(self, checkpoint_path=None, device="cuda", confidence_threshold=0.5):
        checkpoint_path = checkpoint_path or DEFAULT_SAM3_CHECKPOINT
        self.device = device
        self.default_confidence_threshold = confidence_threshold
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False, device=device)
        self.proc = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)

    def segment(self, image_bgr: np.ndarray, prompts: list, max_area_frac: float = 0.15,
                confidence_threshold: float = None):
        """
        Run SAM3 on the given BGR image for each prompt string, returning a bool mask.
        Masks from individual instances are unioned; any instance whose bounding box
        exceeds max_area_frac of the frame area is discarded. confidence_threshold
        overrides the segmenter's default for this call only (e.g. small-fixture
        concept prompts want a higher bar than the actor prompt).
        """
        if image_bgr.size == 0 or not prompts:
            return np.zeros(image_bgr.shape[:2], dtype=bool)
        thresh = confidence_threshold if confidence_threshold is not None else self.default_confidence_threshold
        self.proc.set_confidence_threshold(thresh)
        pil_image = Image.fromarray(image_bgr[..., ::-1])
        h, w = image_bgr.shape[:2]
        area = h * w
        max_area = area * max_area_frac
        combined_mask = np.zeros((h, w), dtype=bool)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.proc.set_image(pil_image)
            for prompt in prompts:
                st = self.proc.set_text_prompt(prompt, dict(state))
                masks = st["masks"]          # [N,1,H,W]
                boxes = st["boxes"]          # [N,4]
                if masks.numel() == 0:
                    continue
                for i in range(masks.shape[0]):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    box_area = max(0, (x2 - x1) * (y2 - y1))
                    if box_area > max_area:
                        continue
                    mask_i = masks[i, 0].cpu().numpy()
                    combined_mask |= mask_i
        return combined_mask

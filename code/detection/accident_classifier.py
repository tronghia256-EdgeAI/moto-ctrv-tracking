"""
code/detection/accident_classifier.py
────────────────────────────────────────
EfficientNet-B0 severity classifier.

3-class output: non-accident / minor / severe

Input: cropped ROI from a track flagged as accident by YOLO11.
Output: severity label + confidence.

Blocker isolation: if the fine-tuned weights are missing, the classifier
constructs with enabled=False and the pipeline skips the accident branch
(logged once). The interface and tests still run against a random-weight
model when enabled explicitly.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Severity(Enum):
    NON_ACCIDENT = 0
    MINOR        = 1
    SEVERE       = 2


# Labels for relabeling reference (from the project scope notes):
#   SEVERE : xe lật, người bị văng, >2 xe liên quan, xe không di chuyển được
#   MINOR  : va chạm nhẹ, xe tự di chuyển được sau va chạm


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AccidentClassifier:
    """
    Wraps EfficientNet-B0 fine-tuned on the VN accident dataset.

    Usage:
        classifier = AccidentClassifier('weights/efficientnet_severity.pt')
        if classifier.enabled:
            severity, conf = classifier.classify(roi_frame)
    """

    def __init__(self, weights_path: str, input_size: int = 224,
                 allow_random_weights: bool = False) -> None:
        self.input_size = input_size
        self.enabled = False
        self.model = None

        has_weights = Path(weights_path).exists()
        if not has_weights and not allow_random_weights:
            logger.warning(
                "Severity weights %s not found — accident classification "
                "disabled (pipeline skips this branch).", weights_path,
            )
            return

        import timm
        import torch

        self.torch = torch
        self.model = timm.create_model("efficientnet_b0", pretrained=False,
                                       num_classes=3)
        if has_weights:
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state)
        self.model.eval()
        self.enabled = True

    def classify(self, roi: np.ndarray) -> tuple[Severity, float]:
        """
        Args:
            roi: cropped accident region, BGR (H, W, 3)

        Returns:
            (Severity enum, confidence float 0.0–1.0)
        """
        assert self.enabled, "classifier is disabled (no weights)"
        import cv2

        img = cv2.resize(roi, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = self.torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

        with self.torch.no_grad():
            logits = self.model(tensor)
            probs = self.torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        return Severity(idx), float(probs[idx])

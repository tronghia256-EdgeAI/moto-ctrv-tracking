"""
code/detection/detector.py
───────────────────────────
YOLO11 (Ultralytics) detector wrapper.

Provides a thin interface so the rest of the pipeline does not
import ultralytics directly everywhere (lazy import — tracker tests
never load torch).

IMPORTANT (D3): the detector always runs at a LOW fixed confidence
(DETECTOR_CONF = 0.1). The adaptive confidence from AdaptiveController
is applied as the tracker's track_thresh — BYTE's second association
stage *needs* the low-score detections, so filtering here would
silently disable half the algorithm.

Fallback (D6): if the VN-trained weights are missing, fall back to
COCO-pretrained yolo11n and remap COCO ids → project ids
(car 2→0, motorcycle 3→1, bus 5→2, truck 7→3). COCO has no accident
class, so the accident branch is disabled upstream in that case.

Project class ids: 0 car, 1 motorcycle, 2 bus, 3 truck, 4 accident.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DETECTOR_CONF = 0.1   # fixed low conf — see module docstring (D3)

# Prefer the already-downloaded copy in weights/ (gitignored); a bare
# model name makes ultralytics auto-download to the CWD.
COCO_FALLBACK_WEIGHTS = (
    "weights/yolo11n.pt" if Path("weights/yolo11n.pt").exists() else "yolo11n.pt"
)

# Canonical project class ids. The id→id remap is built from the loaded
# model's class NAMES, so any model whose labels use these names works
# regardless of its internal ordering (COCO's car=2/motorcycle=3, or a
# custom-trained model with e.g. Accident=0/bus=1/car=2/...).
PROJECT_CLASSES = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3,
                   "accident": 4}


class Detector:
    """
    Wraps YOLO11 inference.

    Usage:
        detector = Detector(weights='weights/yolo11_vn.pt')
        detections = detector.detect(frame)     # N×6 numpy
    """

    def __init__(self, weights_path: str, img_size: int = 640,
                 task: str | None = None, roi=None) -> None:
        from ultralytics import YOLO  # lazy: only pipeline runs need torch

        self.img_size = img_size
        self.roi = roi
        # `task` matters only for exported weights (.onnx, openvino dirs): a
        # .pt carries its task, an export does not always. Guessing wrong is
        # silent and total — a segment model read as `detect` parses its 32
        # mask coefficients as class scores, yielding confidences above 1.0
        # and boxes that the class remap then discards, so the pipeline sees
        # zero detections while every layer still reports success.
        kwargs = {"task": task} if task else {}
        if Path(weights_path).exists():
            if not task and Path(weights_path).suffix != ".pt":
                logger.warning(
                    "Loading exported weights %s without an explicit task; "
                    "ultralytics will guess, and a wrong guess fails silently "
                    "(zero detections, no error). Pass task= to be sure.",
                    weights_path)
            self.model = YOLO(weights_path, **kwargs)
        else:
            logger.warning(
                "Weights %s not found — falling back to COCO-pretrained %s. "
                "Only car/motorcycle/bus/truck are detected; the accident "
                "class is unavailable in this mode.",
                weights_path, COCO_FALLBACK_WEIGHTS,
            )
            self.model = YOLO(COCO_FALLBACK_WEIGHTS)
        # COCO-ness is a property of the loaded model, not of which path
        # it came from (an 80-class model has no accident class).
        self.using_coco_fallback = len(self.model.names) == 80
        # Name-based id remap (see PROJECT_CLASSES). Classes the project
        # does not know (person, bicycle, ...) are dropped in detect().
        self.remap = {
            mid: PROJECT_CLASSES[name.lower()]
            for mid, name in self.model.names.items()
            if name.lower() in PROJECT_CLASSES
        }
        if not self.remap:
            raise ValueError(
                f"model classes {self.model.names} share no names with "
                f"project classes {list(PROJECT_CLASSES)}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR frame (H, W, 3)

        Returns:
            N×6 numpy [x1, y1, x2, y2, score, class_id] in frame-native
            pixels, project class ids.
        """
        from code.scripts.gt_roi import pixel_mask,filter_detections
        mask=pixel_mask(frame,self.roi)
        if mask is not None:
            import cv2
            frame=cv2.bitwise_and(frame,frame,mask=mask)
        results = self.model(frame, conf=DETECTOR_CONF, imgsz=self.img_size,
                             verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        keep = np.array([c in self.remap for c in cls])
        if not keep.any():
            return np.empty((0, 6))
        xyxy, conf = xyxy[keep], conf[keep]
        cls = np.array([self.remap[c] for c in cls[keep]])

        return filter_detections(np.column_stack([xyxy, conf, cls]).astype(float),self.roi)

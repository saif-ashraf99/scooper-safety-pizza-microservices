import numpy as np
import logging
import os
from typing import List, Dict, Any, Optional

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLO detector using Ultralytics YOLOv12."""

    def __init__(self, model_path):
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("Model path must be provided and point to a valid .pt file")

        original_load = torch.load

        def load_with_weights_only_false(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        try:
            torch.load = load_with_weights_only_false
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from: {model_path}")

            # Auto-select CUDA/GPU if available (handled by Ultralytics)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            logger.info(f"YOLO model moved to device: {device}")

            # Passes a blank image through the model to reduce latency on first real use.
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False)
            logger.info("YOLO model warmed up on a dummy frame.")

        except Exception as e:
            logger.error(f"Failed to load or warm up YOLO model: {e}")
            raise
        finally:
            torch.load = original_load

    def detect(self, frame: np.ndarray, conf_thres: float = 0.25, allowed_classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            # If allowed_classes are specified, map them to class IDs
            class_ids = None
            if allowed_classes:
                # Get class IDs corresponding to the allowed class names
                class_ids = [k for k, v in self.model.names.items() if v in allowed_classes]
                if not class_ids:
                    logger.warning(f"No valid class IDs found for allowed_classes: {allowed_classes}")
                    return []

            results = self.model(frame, verbose=False, conf=conf_thres, classes=class_ids)[0]
            detections = []

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []



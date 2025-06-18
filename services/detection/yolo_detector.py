import numpy as np
import logging
import os
from typing import List, Dict, Any, Optional

import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO detector using Ultralytics API with weights_only=False fix"""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("Model path must be provided and point to a valid .pt file")
        
        # Save original torch.load
        original_load = torch.load
        
        def load_with_weights_only_false(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        try:
            # Temporarily replace torch.load
            torch.load = load_with_weights_only_false
            
            # Load the model
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        finally:
            # Restore original torch.load
            torch.load = original_load

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform object detection on frame"""
        try:
            results = self.model(frame)[0] 
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

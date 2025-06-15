import cv2
import numpy as np
import base64
import time
import logging
import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import argparse

import torch
from ultralytics import YOLO


# Import shared modules
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import (
    RabbitMQConsumer, RabbitMQPublisher, Database, 
    Detection, Violation, BoundingBox, ViolationType, 
    DetectionClass, ViolationRecord, DetectionResult
)

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

class ViolationDetector:
    """Violation detection logic"""
    
    def __init__(self, rois: List[Dict[str, Any]]):
        self.rois = rois
        self.hand_history = {}  # Track hand movements
        self.scooper_threshold = 0.7
        
    def is_point_in_roi(self, point: Tuple[float, float], roi: Dict[str, Any]) -> bool:
        """Check if point is inside ROI"""
        x, y = point
        x1, y1, x2, y2 = roi['coordinates']
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def detect_violations(self, detections: List[Detection], frame_id: str) -> List[Violation]:
        """Detect violations based on detections and ROIs"""
        violations = []
        
        # Extract hands and scoopers
        hands = [d for d in detections if d.class_name == DetectionClass.HAND]
        scoopers = [d for d in detections if d.class_name == DetectionClass.SCOOPER]
        
        # Check each hand
        for hand in hands:
            hand_center = self.get_bbox_center([hand.bbox.x1, hand.bbox.y1, hand.bbox.x2, hand.bbox.y2])
            
            # Check if hand is in any ROI
            for roi in self.rois:
                if not roi.get('active', True):
                    continue
                    
                if self.is_point_in_roi(hand_center, roi):
                    # Hand is in ROI, check if scooper is present
                    scooper_present = False
                    
                    for scooper in scoopers:
                        scooper_center = self.get_bbox_center([scooper.bbox.x1, scooper.bbox.y1, scooper.bbox.x2, scooper.bbox.y2])
                        
                        # Check if scooper is close to hand
                        distance = np.sqrt((hand_center[0] - scooper_center[0])**2 + 
                                         (hand_center[1] - scooper_center[1])**2)
                        
                        if distance < 100 and scooper.confidence > self.scooper_threshold:  # Within 100 pixels
                            scooper_present = True
                            break
                    
                    # If no scooper present, it's a violation
                    if not scooper_present:
                        violation = Violation(
                            type=ViolationType.NO_SCOOPER,
                            roi_id=roi['id'],
                            confidence=hand.confidence,
                            bbox=hand.bbox,
                            description=f"Hand detected in {roi['name']} without scooper"
                        )
                        violations.append(violation)
                        logger.warning(f"Violation detected in {roi['id']}: {violation.description}")
        
        return violations

class DetectionService:
    def __init__(self, rabbitmq_host: str = 'localhost', rabbitmq_port: int = 5672,
                 rabbitmq_user: str = 'guest', rabbitmq_password: str = 'guest',
                 model_path: Optional[str] = None, db_path: str = 'violations.db'):
        
        # Initialize RabbitMQ
        self.consumer = RabbitMQConsumer(
            host=rabbitmq_host,
            port=rabbitmq_port,
            username=rabbitmq_user,
            password=rabbitmq_password
        )
        
        self.publisher = RabbitMQPublisher(
            host=rabbitmq_host,
            port=rabbitmq_port,
            username=rabbitmq_user,
            password=rabbitmq_password
        )
        
        # Initialize database
        self.db = Database(db_path)
        
        # Initialize detector
        self.detector = YOLODetector(model_path)
        
        # Initialize violation detector with ROIs from database
        rois = self.db.get_rois()
        self.violation_detector = ViolationDetector([roi.dict() for roi in rois])
        
        # Metrics
        self.frames_processed = 0
        self.violations_detected = 0
        self.start_time = time.time()
        
        logger.info("Detection service initialized")
    
    def decode_frame(self, frame_data: str) -> np.ndarray:
        """Decode base64 frame to numpy array"""
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                       violations: List[Violation]) -> np.ndarray:
        """Draw detections and violations on frame"""
        frame_copy = frame.copy()
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = int(detection.bbox.x1), int(detection.bbox.y1), int(detection.bbox.x2), int(detection.bbox.y2)
            
            # Color based on class
            color_map = {
                DetectionClass.HAND: (0, 255, 0),      # Green
                DetectionClass.PERSON: (255, 0, 0),    # Blue
                DetectionClass.PIZZA: (0, 255, 255),   # Yellow
                DetectionClass.SCOOPER: (255, 0, 255)  # Magenta
            }
            color = color_map.get(detection.class_name, (128, 128, 128))
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, f"{detection.class_name.value}: {detection.confidence:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ROIs
        for roi in self.violation_detector.rois:
            if roi.get('active', True):
                x1, y1, x2, y2 = [int(coord) for coord in roi['coordinates']]
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red ROI
                cv2.putText(frame_copy, roi['name'], (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw violations
        for violation in violations:
            x1, y1, x2, y2 = int(violation.bbox.x1), int(violation.bbox.y1), int(violation.bbox.x2), int(violation.bbox.y2)
            
            # Red violation box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame_copy, "VIOLATION!", (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame_copy, violation.description or "No scooper", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame_copy
    
    def save_violation_frame(self, frame: np.ndarray, frame_id: str) -> str:
        """Save violation frame to disk"""
        os.makedirs('violation_frames', exist_ok=True)
        frame_path = f"violation_frames/{frame_id}.jpg"
        cv2.imwrite(frame_path, frame)
        return frame_path
    
    def process_frame(self, message: Dict[str, Any]):
        """Process a single frame message"""
        try:
            start_time = time.time()
            
            # Extract frame data
            frame_id = message['frame_id']
            timestamp = datetime.fromisoformat(message['timestamp'])
            frame_data = message['frame_data']
            
            # Decode frame
            frame = self.decode_frame(frame_data)
            
            # Perform detection
            raw_detections = self.detector.detect(frame)
            
            # Convert to Detection objects
            detections = []
            for det in raw_detections:
                detection = Detection(
                    class_name=DetectionClass(det['class']),
                    confidence=det['confidence'],
                    bbox=BoundingBox(
                        x1=det['bbox'][0],
                        y1=det['bbox'][1],
                        x2=det['bbox'][2],
                        y2=det['bbox'][3]
                    )
                )
                detections.append(detection)
            
            # Detect violations
            violations = self.violation_detector.detect_violations(detections, frame_id)
            
            # Draw detections and violations on frame
            annotated_frame = self.draw_detections(frame, detections, violations)
            annotated_frame_data = self.encode_frame(annotated_frame)
            
            # Save violations to database
            for violation in violations:
                frame_path = self.save_violation_frame(annotated_frame, frame_id)
                
                violation_record = ViolationRecord(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    violation_type=violation.type,
                    roi_id=violation.roi_id,
                    confidence=violation.confidence,
                    frame_path=frame_path,
                    bounding_boxes=detections,
                    metadata={'description': violation.description}
                )
                
                self.db.insert_violation(violation_record)
                self.violations_detected += 1
            
            # Create detection result
            processing_time = time.time() - start_time
            result = DetectionResult(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=detections,
                violations=violations,
                frame_data=annotated_frame_data,
                processing_time=processing_time
            )
            
            # Publish result
            self.publisher.publish_detection_result(result.dict())
            
            self.frames_processed += 1
            
            if self.frames_processed % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frames_processed / elapsed
                logger.info(f"Processed {self.frames_processed} frames, "
                          f"FPS: {fps:.2f}, Violations: {self.violations_detected}")
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
    
    def start(self):
        """Start the detection service"""
        logger.info("Starting detection service...")
        try:
            self.consumer.consume_frames(self.process_frame)
        except KeyboardInterrupt:
            logger.info("Detection service stopped by user")
        except Exception as e:
            logger.error(f"Detection service error: {e}")
            raise
        finally:
            self.consumer.disconnect()
            self.publisher.disconnect()

def main():
    parser = argparse.ArgumentParser(description='Detection Service')
    parser.add_argument('--model-path', 
                       help='Path to YOLO model file')
    parser.add_argument('--db-path', 
                       help='Database path')
    parser.add_argument('--rabbitmq-host', 
                       help='RabbitMQ host')
    parser.add_argument('--rabbitmq-port', type=int, 
                       help='RabbitMQ port')
    parser.add_argument('--rabbitmq-user', 
                       help='RabbitMQ username')
    parser.add_argument('--rabbitmq-password', 
                       help='RabbitMQ password')
    
    args = parser.parse_args()
    
    # Get configuration from environment variables or command line args
    rabbitmq_host = args.rabbitmq_host or os.getenv('RABBITMQ_HOST', 'localhost')
    rabbitmq_port = args.rabbitmq_port or int(os.getenv('RABBITMQ_PORT', '5672'))
    rabbitmq_user = args.rabbitmq_user or os.getenv('RABBITMQ_USER', 'guest')
    rabbitmq_password = args.rabbitmq_password or os.getenv('RABBITMQ_PASSWORD', 'guest')
    model_path = args.model_path or os.getenv('MODEL_PATH')
    db_path = args.db_path or os.getenv('DATABASE_PATH', 'violations.db')
    
    # Print configuration for debugging
    print(f"RabbitMQ Configuration:")
    print(f"  Host: {rabbitmq_host}")
    print(f"  Port: {rabbitmq_port}")
    print(f"  User: {rabbitmq_user}")
    print(f"  Database: {db_path}")
    print(f"  Model: {model_path}")
    
    # Create detection service
    service = DetectionService(
        rabbitmq_host=rabbitmq_host,
        rabbitmq_port=rabbitmq_port,
        rabbitmq_user=rabbitmq_user,
        rabbitmq_password=rabbitmq_password,
        model_path=model_path,
        db_path=db_path
    )
    
    try:
        service.start()
    except Exception as e:
        logger.error(f"Failed to start detection service: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


import os
import cv2
import numpy as np
import base64
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from services.shared import (
    RabbitMQConsumer, RabbitMQPublisher, Database, 
    DetectionResult
)

from services.detection.video_processor import VideoProcessor
from services.detection.yolo_detector import YOLODetector
from services.detection.violation_detector import ViolationDetector

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionService:
    def __init__(self, camera_id: str, model_path: Optional[str] = None, db_path: str = 'violations.db',
                 rabbitmq_host: str = 'localhost', rabbitmq_port: int = 5672,
                 rabbitmq_user: str = 'admin', rabbitmq_password: str = 'admin123', enable_rabbit: bool = False):
        
        self.camera_id = camera_id
        self.consumer = None
        self.publisher = None

        if enable_rabbit:
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
        
        self.db = Database(db_path)
        self.detector = YOLODetector(model_path)
        
        # Initialize violation detector without ROIs from database; ROIs are now managed by ContainerFinder
        self.violation_detector = ViolationDetector([]) # Initialize with empty list, ROIs will be set per frame
        
        # Internally instantiate VideoProcessor (no standalone ContainerFinder)
        self.video_processor = VideoProcessor(
            camera_id=self.camera_id,
            detector=self.detector,
            violation_detector=self.violation_detector,
            db=self.db
        )
        
        # Metrics
        self.frames_processed = 0
        self.violations_detected = 0
        self.start_time = time.time()
        
        logger.info(f"Detection service initialized for camera_id: {self.camera_id}")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """Process video file"""
        return self.video_processor.process_video(video_path, output_path, **kwargs)
    
    def decode_frame(self, frame_data: str) -> np.ndarray:
        """Decode base64 frame to numpy array"""
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 string"""
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def process_frame(self, message: Dict[str, Any]):
        """Process a single frame message (for RabbitMQ mode)"""
        try:
            start_time = time.time()
            
            # Extract frame data
            frame_id = message['frame_id']
            timestamp = datetime.fromisoformat(message['timestamp'])
            frame_data = message['frame_data']
            
            # Decode frame
            frame = self.decode_frame(frame_data)
            
            # Process using video processor
            result = self.video_processor.process_single_frame(
                frame, frame_id, timestamp, save_violations=True
            )
            
            # Encode annotated frame
            annotated_frame_data = self.encode_frame(result['annotated_frame'])
            
            # Create detection result
            processing_time = time.time() - start_time
            detection_result = DetectionResult(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=result['detections'],
                violations=result['violations'],
                frame_data=annotated_frame_data,
                processing_time=processing_time
            )
            
            # Publish result if publisher available
            if self.publisher:
                self.publisher.publish_detection_result(detection_result.model_dump())
            
            self.frames_processed += 1
            self.violations_detected += len(result['violations'])
            
            if self.frames_processed % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frames_processed / elapsed
                logger.info(f"Processed {self.frames_processed} frames, "
                          f"FPS: {fps:.2f}, Violations: {self.violations_detected}")
            
        except Exception as e:
            logger.error(f"Error processing frame {message.get('frame_id', 'UNKNOWN')}: {e}")
    
    def start(self):
        """Start the detection service (RabbitMQ mode)"""
        if not self.consumer:
            logger.error("RabbitMQ consumer not initialized. Set enable_rabbit=True for RabbitMQ mode.")
            return
            
        logger.info("Starting detection service...")
        try:
            self.consumer.consume_frames(self.process_frame)
        except KeyboardInterrupt:
            logger.info("Detection service stopped by user")
        except Exception as e:
            logger.error(f"Detection service error: {e}")
            raise
        finally:
            if self.consumer:
                self.consumer.disconnect()
            if self.publisher:
                self.publisher.disconnect()





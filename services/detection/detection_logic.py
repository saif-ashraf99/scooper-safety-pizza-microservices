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
from services.detection.container_finder import ContainerFinder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionService:
    def __init__(self, rabbitmq_host: str , rabbitmq_port: int,
                 rabbitmq_user: str, rabbitmq_password: str,
                 model_path: Optional[str], db_path: str):
        
        # Initialize RabbitMQ (optional for video processing)
        self.consumer = None
        self.publisher = None
        
        if rabbitmq_host != 'none':  # Allow skipping RabbitMQ for video processing
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
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            self.detector, self.violation_detector, self.db
        )
        self.container_finder = ContainerFinder(
            match_threshold=0.8,
            use_edges=False         # try True if lighting varies wildly
        )
        
        # Metrics
        self.frames_processed = 0
        self.violations_detected = 0
        self.start_time = time.time()
        
        logger.info("Detection service initialized")
    
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
        _, buffer = cv2.imencode('.jpg', frame)
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
            
            if self.frames_processed % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frames_processed / elapsed
                logger.info(f"Processed {self.frames_processed} frames, "
                          f"FPS: {fps:.2f}, Violations: {self.violations_detected}")
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
    
    def start(self):
        """Start the detection service (RabbitMQ mode)"""
        if not self.consumer:
            logger.error("RabbitMQ consumer not initialized. Use process_video() for video files.")
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

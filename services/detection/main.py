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
sys.path.append('/home/ubuntu/scooper-safety-pizza-microservices')
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

class VideoProcessor:
    """Video processing class for handling video file input"""
    
    def __init__(self, detector: YOLODetector, violation_detector: ViolationDetector, 
                 db: Database):
        self.detector = detector
        self.violation_detector = violation_detector
        self.db = db
        self.frame_count = 0
        self.violations_detected = 0
        
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     save_violations: bool = True, display: bool = True,
                     skip_frames: int = 0) -> Dict[str, Any]:
        """
        Process video file and detect objects/violations
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            save_violations: Whether to save violations to database
            display: Whether to display video while processing
            skip_frames: Process every Nth frame (0 = process all frames)
        
        Returns:
            Dictionary with processing statistics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup video writer if output path provided
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Output will be saved to: {output_path}")
        
        # Processing statistics
        start_time = time.time()
        processed_frames = 0
        total_detections = 0
        frame_violations = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if skip_frames > 0 and self.frame_count % (skip_frames + 1) != 0:
                    self.frame_count += 1
                    continue
                
                # Generate frame ID and timestamp
                frame_id = f"video_{uuid.uuid4().hex[:8]}_{self.frame_count:06d}"
                timestamp = datetime.now()
                
                # Process frame
                frame_result = self.process_single_frame(
                    frame, frame_id, timestamp, save_violations
                )
                
                # Update statistics
                processed_frames += 1
                total_detections += len(frame_result['detections'])
                if frame_result['violations']:
                    frame_violations.append({
                        'frame_id': frame_id,
                        'frame_number': self.frame_count,
                        'timestamp': timestamp.isoformat(),
                        'violations': len(frame_result['violations'])
                    })
                
                # Get annotated frame
                annotated_frame = frame_result['annotated_frame']
                
                # Write frame to output video
                if out_writer:
                    out_writer.write(annotated_frame)
                
                # Display frame
                if display:
                    # Add processing info to frame
                    info_text = f"Frame: {self.frame_count}/{total_frames} | " \
                               f"Detections: {len(frame_result['detections'])} | " \
                               f"Violations: {len(frame_result['violations'])}"
                    cv2.putText(annotated_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Video Processing', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processing stopped by user")
                        break
                
                # Progress update
                if processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = processed_frames / elapsed
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% | "
                              f"Processing FPS: {fps_current:.1f} | "
                              f"Violations detected: {self.violations_detected}")
                
                self.frame_count += 1
        
        finally:
            # Cleanup
            cap.release()
            if out_writer:
                out_writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        stats = {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'processing_time': processing_time,
            'avg_fps': processed_frames / processing_time if processing_time > 0 else 0,
            'total_detections': total_detections,
            'total_violations': self.violations_detected,
            'violation_frames': frame_violations,
            'video_duration': total_frames / fps if fps > 0 else 0
        }
        
        logger.info("Video processing completed:")
        logger.info(f"  Processed {processed_frames}/{total_frames} frames")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        logger.info(f"  Average FPS: {stats['avg_fps']:.2f}")
        logger.info(f"  Total violations: {self.violations_detected}")
        
        return stats
    
    def process_single_frame(self, frame: np.ndarray, frame_id: str, 
                           timestamp: datetime, save_violations: bool = True) -> Dict[str, Any]:
        """Process a single frame and return results"""
        try:
            # Perform detection
            raw_detections = self.detector.detect(frame)
            
            # Convert to Detection objects
            detections = []
            for det in raw_detections:
                # Map class names to DetectionClass enum
                class_name = self._map_class_name(det['class'])
                if class_name:  # Only include relevant classes
                    detection = Detection(
                        class_name=class_name,
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
            
            # Save violations to database if requested
            if save_violations and violations:
                self._save_violations(violations, frame_id, timestamp, 
                                    annotated_frame, detections)
            
            return {
                'detections': detections,
                'violations': violations,
                'annotated_frame': annotated_frame
            }
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return {
                'detections': [],
                'violations': [],
                'annotated_frame': frame
            }
    
    def _map_class_name(self, class_name: str) -> Optional[DetectionClass]:
        """Map YOLO class names to DetectionClass enum"""
        class_mapping = {
            'person': DetectionClass.PERSON,
            'hand': DetectionClass.HAND,
            'pizza': DetectionClass.PIZZA,
            'scooper': DetectionClass.SCOOPER,
            # Add more mappings as needed based on your model
        }
        return class_mapping.get(class_name.lower())
    
    def _save_violations(self, violations: List[Violation], frame_id: str, 
                        timestamp: datetime, frame: np.ndarray, 
                        detections: List[Detection]):
        """Save violations to database"""
        for violation in violations:
            # Save violation frame
            frame_path = self._save_violation_frame(frame, frame_id)
            
            # Create violation record
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
            
            # Insert to database
            self.db.insert_violation(violation_record)
            self.violations_detected += 1
    
    def _save_violation_frame(self, frame: np.ndarray, frame_id: str) -> str:
        """Save violation frame to disk"""
        os.makedirs('violation_frames', exist_ok=True)
        frame_path = f"violation_frames/{frame_id}.jpg"
        cv2.imwrite(frame_path, frame)
        return frame_path
    
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

class DetectionService:
    def __init__(self, rabbitmq_host: str = 'rabbitmq', rabbitmq_port: int = 5672,
                 rabbitmq_user: str = 'admin', rabbitmq_password: str = 'admin123',
                 model_path: Optional[str] = None, db_path: str = 'violations.db'):
        
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

def main():
    parser = argparse.ArgumentParser(description='Detection Service')
    parser.add_argument('--mode', choices=['service', 'video'], default='service',
                       help='Operating mode: service (RabbitMQ) or video processing')
    parser.add_argument('--video-path', 
                       help='Path to video file (for video mode)')
    parser.add_argument('--output-path', 
                       help='Path to output video file')
    parser.add_argument('--model-path', 
                       help='Path to YOLO model file')
    parser.add_argument('--db-path', 
                       help='Database path')
    parser.add_argument('--rabbitmq-host', 
                       help='RabbitMQ host (use "none" to skip RabbitMQ)')
    parser.add_argument('--rabbitmq-port', type=int, 
                       help='RabbitMQ port')
    parser.add_argument('--rabbitmq-user', 
                       help='RabbitMQ username')
    parser.add_argument('--rabbitmq-password', 
                       help='RabbitMQ password')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Process every Nth frame (0 = all frames)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display during processing')
    parser.add_argument('--no-save-violations', action='store_true',
                       help='Disable saving violations to database')
    
    args = parser.parse_args()
    
    # Get configuration from environment variables or command line args
    rabbitmq_host = args.rabbitmq_host or os.getenv('RABBITMQ_HOST', 'localhost')
    rabbitmq_port = args.rabbitmq_port or int(os.getenv('RABBITMQ_PORT', '5672'))
    rabbitmq_user = args.rabbitmq_user or os.getenv('RABBITMQ_USER', 'guest')
    rabbitmq_password = args.rabbitmq_password or os.getenv('RABBITMQ_PASSWORD', 'guest')
    model_path = args.model_path or os.getenv('MODEL_PATH')
    db_path = args.db_path or os.getenv('DATABASE_PATH', 'violations.db')
    
    # Print configuration for debugging
    print(f"Mode: {args.mode}")
    if args.mode == 'video':
        print(f"Video path: {args.video_path}")
        print(f"Output path: {args.output_path}")
        print(f"Skip frames: {args.skip_frames}")
        print(f"Display: {not args.no_display}")
    else:
        print(f"RabbitMQ Configuration:")
        print(f"  Host: {rabbitmq_host}")
        print(f"  Port: {rabbitmq_port}")
        print(f"  User: {rabbitmq_user}")
    print(f"Database: {db_path}")
    print(f"Model: {model_path}")
    
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
        if args.mode == 'video':
            if not args.video_path:
                logger.error("Video path required for video mode")
                return 1
            
            # Process video
            stats = service.process_video(
                video_path=args.video_path,
                output_path=args.output_path,
                save_violations=not args.no_save_violations,
                display=not args.no_display,
                skip_frames=args.skip_frames
            )
            
            # Print final statistics
            print("\n" + "="*50)
            print("VIDEO PROCESSING COMPLETE")
            print("="*50)
            print(f"Input: {stats['video_path']}")
            if stats['output_path']:
                print(f"Output: {stats['output_path']}")
            print(f"Frames processed: {stats['processed_frames']}/{stats['total_frames']}")
            print(f"Processing time: {stats['processing_time']:.2f}s")
            print(f"Average FPS: {stats['avg_fps']:.2f}")
            print(f"Total violations: {stats['total_violations']}")
            print(f"Violation frames: {len(stats['violation_frames'])}")
            
        else:
            # Start RabbitMQ service mode
            service.start()
            
    except Exception as e:
        logger.error(f"Failed to run detection service: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
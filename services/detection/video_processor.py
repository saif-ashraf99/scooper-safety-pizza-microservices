import cv2
import numpy as np
import time
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from services.shared import (
    Database,  Detection, Violation, BoundingBox,
    DetectionClass, ViolationRecord,
)

from services.detection.yolo_detector import YOLODetector
from services.detection.violation_detector import ViolationDetector
from services.detection.container_finder import ContainerFinder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video processing class for handling video file input"""
    
    def __init__(self, detector: YOLODetector, violation_detector: ViolationDetector, 
                 db: Database):
        self.detector = detector
        self.violation_detector = violation_detector
        self.db = db

        # Build your finder here, too:
        self.container_finder = ContainerFinder(
            match_threshold=0.8,
            use_edges=False         # try True if lighting varies wildly
        )

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
    

    def process_single_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        timestamp: datetime,
        save_violations: bool = True
    ) -> Dict[str, Any]:
        """Process a single frame and return results"""
        try:
            # 1) YOLO → Detection objects
            raw = self.detector.detect(frame)
            detections: List[Detection] = []
            for det in raw:
                cls = self._map_class_name(det['class'])
                if not cls:
                    continue
                detections.append(
                    Detection(
                        class_name=cls,
                        confidence=det['confidence'],
                        bbox=BoundingBox(
                            x1=det['bbox'][0],
                            y1=det['bbox'][1],
                            x2=det['bbox'][2],
                            y2=det['bbox'][3]
                        )
                    )
                )

            # 2) Dynamic ROI via template-matching
            dyn = self.container_finder.find(frame)
            if dyn:
                x1, y1, x2, y2 = dyn
                rois = [{
                    "id":          "protein_container",
                    "name":        "Protein Container",
                    "coordinates": [x1, y1, x2, y2],
                    "active":      True
                }]
                logger.debug(f"[{frame_id}] using dynamic ROI {x1,y1,x2,y2}")
            else:
                # fallback to static from DB
                rois = [r.dict() for r in self.db.get_rois()]
                logger.debug(f"[{frame_id}] no dynamic ROI; using static {rois[0]['coordinates']}")

            # 3) Inject into violation detector
            self.violation_detector.rois = rois

            # 4) Detect violations
            violations = self.violation_detector.detect_violations(detections, frame_id)

            # 5) Draw & save
            annotated = self.draw_detections(frame, detections, violations)
            if save_violations and violations:
                self._save_violations(violations, frame_id, timestamp, annotated, detections)

            return {"detections": detections,
                    "violations": violations,
                    "annotated_frame": annotated}

        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return {"detections": [], "violations": [], "annotated_frame": frame}


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

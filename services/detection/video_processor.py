import cv2
import numpy as np
import time
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from services.shared import (
    Database, Detection, Violation, BoundingBox,
    DetectionClass, ViolationRecord,
)

from services.detection.yolo_detector import YOLODetector
from services.detection.violation_detector import ViolationDetector
from services.detection.container_finder import ContainerFinder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Processes video frames for object detection and violation detection."""
    
    # Class constants
    CLASS_MAPPING = {
        'person': DetectionClass.PERSON,
        'hand': DetectionClass.HAND,
        'pizza': DetectionClass.PIZZA,
        'scooper': DetectionClass.SCOOPER,
    }
    
    COLOR_MAP = {
        DetectionClass.HAND: (0, 255, 0),
        DetectionClass.PERSON: (255, 0, 0),
        DetectionClass.PIZZA: (0, 255, 255),
        DetectionClass.SCOOPER: (255, 0, 255)
    }
    
    DEFAULT_COLOR = (128, 128, 128)
    VIOLATION_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    
    def __init__(
        self, 
        camera_id: str, 
        detector: YOLODetector, 
        violation_detector: ViolationDetector, 
        db: Database
    ):
        self.camera_id = camera_id
        self.detector = detector
        self.violation_detector = violation_detector
        self.db = db    
        self.container_finder = ContainerFinder(camera_id=self.camera_id)
        self.frame_count = 0
        self.violations_detected = 0

    def process_single_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        timestamp: datetime,
        save_violations: bool = True
    ) -> Dict[str, Any]:
        """Process a single frame and return results."""
        try:
            detections = self._run_object_detection(frame)
            rois = self._get_calibrated_rois(frame, frame_id)
            
            if not rois:
                logger.error(f"[{frame_id}] No calibrated ROIs found. Cannot proceed with violation detection.")
                return self._create_empty_result(frame)
            
            violations = self._detect_violations(detections, rois, frame, frame_id)
            annotated_frame = self.draw_detections(frame, detections, violations, rois)
            
            if save_violations and violations:
                self._save_violations(violations, frame_id, timestamp, annotated_frame, detections)

            return {
                "detections": detections,
                "violations": violations,
                "annotated_frame": annotated_frame
            }

        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return self._create_empty_result(frame)

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        save_violations: bool = True,
        display: bool = True,
        skip_frames: int = 0
    ) -> Dict[str, Any]:
        """Process entire video file."""
        self._validate_video_path(video_path)
        
        cap = self._initialize_video_capture(video_path)
        video_info = self._get_video_info(cap)
        
        first_frame = self._read_first_frame(cap)
        self._perform_calibration_if_needed(first_frame, cap)
        
        out_writer = self._initialize_video_writer(output_path, video_info)
        
        processing_stats = self._process_video_frames(
            cap, out_writer, video_info, save_violations, display, skip_frames
        )
        
        self._cleanup_resources(cap, out_writer, display)
        
        return self._compile_final_stats(
            video_path, output_path, video_info, processing_stats
        )

    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Detection], 
        violations: List[Violation], 
        rois: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draw detections, violations, and ROIs on frame."""
        frame_copy = frame.copy()
        
        self._draw_detection_boxes(frame_copy, detections)
        self._draw_roi_boxes(frame_copy, rois)
        self._draw_violation_boxes(frame_copy, violations)
        
        return frame_copy

    # Private methods for object detection
    def _run_object_detection(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection and convert to Detection objects."""
        raw_detections = self.detector.detect(frame)
        detections = []
        
        for det in raw_detections:
            detection_class = self._map_class_name(det['class'])
            if not detection_class:
                continue
                
            detection = Detection(
                class_name=detection_class,
                confidence=det['confidence'],
                bbox=BoundingBox(
                    x1=det['bbox'][0],
                    y1=det['bbox'][1],
                    x2=det['bbox'][2],
                    y2=det['bbox'][3]
                ),
                track_id=det.get('track_id'),
                hand_id=det.get('hand_id')
            )
            detections.append(detection)
            
        return detections

    def _map_class_name(self, class_name: str) -> Optional[DetectionClass]:
        """Map YOLO class names to DetectionClass enum."""
        return self.CLASS_MAPPING.get(class_name.lower())

    # Private methods for ROI handling
    def _get_calibrated_rois(self, frame: np.ndarray, frame_id: str) -> List[Dict[str, Any]]:
        """Get calibrated ROIs from container finder."""
        calibrated_rois = self.container_finder.find(frame)
        if not calibrated_rois:
            return []
            
        rois = []
        for i, roi_coords in enumerate(calibrated_rois):
            x1, y1, x2, y2 = roi_coords
            rois.append({
                "id": f"protein_container_{i}",
                "name": f"Protein Container {i+1}",
                "coordinates": [x1, y1, x2, y2],
                "active": True
            })
            
        logger.debug(f"[{frame_id}] using calibrated ROIs: {calibrated_rois}")
        return rois

    # Private methods for violation detection
    def _detect_violations(
        self, 
        detections: List[Detection], 
        rois: List[Dict[str, Any]], 
        frame: np.ndarray, 
        frame_id: str
    ) -> List[Violation]:
        """Detect violations using the violation detector."""
        self.violation_detector.rois = rois
        return self.violation_detector.detect_violations(detections, frame, frame_id)

    # Private methods for video processing
    def _validate_video_path(self, video_path: str) -> None:
        """Validate that video file exists."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def _initialize_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """Initialize and validate video capture."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        return cap

    def _get_video_info(self, cap: cv2.VideoCapture) -> Dict[str, int]:
        """Extract video information."""
        info = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        logger.info(f"Resolution: {info['width']}x{info['height']}, "
                   f"FPS: {info['fps']}, Total frames: {info['total_frames']}")
        return info

    def _read_first_frame(self, cap: cv2.VideoCapture) -> np.ndarray:
        """Read the first frame for calibration."""
        ret, first_frame = cap.read()
        if not ret:
            logger.error("Could not read the first frame for calibration.")
            raise ValueError("Could not read the first frame for calibration.")
        return first_frame

    def _perform_calibration_if_needed(self, first_frame: np.ndarray, cap: cv2.VideoCapture) -> None:
        """Perform calibration if no existing ROIs for this camera."""
        if not self.container_finder.static_rois:
            logger.info(f"[VideoProcessor] Calibrating container for camera {self.camera_id}...")
            self.container_finder.calibrate(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _initialize_video_writer(
        self, 
        output_path: Optional[str], 
        video_info: Dict[str, int]
    ) -> Optional[cv2.VideoWriter]:
        """Initialize video writer if output path is specified."""
        if not output_path:
            return None
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            output_path, fourcc, video_info['fps'], 
            (video_info['width'], video_info['height'])
        )
        logger.info(f"Output will be saved to: {output_path}")
        return out_writer

    def _process_video_frames(
        self,
        cap: cv2.VideoCapture,
        out_writer: Optional[cv2.VideoWriter],
        video_info: Dict[str, int],
        save_violations: bool,
        display: bool,
        skip_frames: int
    ) -> Dict[str, Any]:
        """Process all video frames."""
        start_time = time.time()
        processed_frames = 0
        total_detections = 0
        frame_violations = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if self._should_skip_frame(skip_frames):
                    self.frame_count += 1
                    continue

                frame_id = self._generate_frame_id()
                timestamp = datetime.now()

                frame_result = self.process_single_frame(
                    frame, frame_id, timestamp, save_violations
                )

                processed_frames += 1
                total_detections += len(frame_result['detections'])

                if frame_result['violations']:
                    frame_violations.append({
                        'frame_id': frame_id,
                        'frame_number': self.frame_count,
                        'timestamp': timestamp.isoformat(),
                        'violations': len(frame_result['violations'])
                    })

                annotated_frame = self._add_stats_overlay(
                    frame_result['annotated_frame'], processed_frames, 
                    video_info['total_frames'], len(frame_result['detections'])
                )

                if out_writer:
                    out_writer.write(annotated_frame)

                if display and self._should_stop_display(annotated_frame):
                    break

                self._log_progress_if_needed(processed_frames, start_time, video_info['total_frames'])
                self.frame_count += 1

        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            raise

        processing_time = time.time() - start_time
        return {
            'processed_frames': processed_frames,
            'total_detections': total_detections,
            'frame_violations': frame_violations,
            'processing_time': processing_time
        }

    def _should_skip_frame(self, skip_frames: int) -> bool:
        """Determine if current frame should be skipped."""
        return skip_frames > 0 and self.frame_count % (skip_frames + 1) != 0

    def _generate_frame_id(self) -> str:
        """Generate unique frame ID."""
        return f"video_{uuid.uuid4().hex[:8]}_{self.frame_count:06d}"

    def _add_stats_overlay(
        self, 
        frame: np.ndarray, 
        processed_frames: int, 
        total_frames: int, 
        current_detections: int
    ) -> np.ndarray:
        """Add running statistics overlay to frame."""
        stats_lines = [
            f"Frames: {processed_frames}/{total_frames}",
            f"Detected Objects: {current_detections}",
            f"Total Violations: {self.violations_detected}"
        ]
        
        for i, line in enumerate(stats_lines):
            y = 30 + i * 30
            color = self.TEXT_COLOR if i < 2 else self.VIOLATION_COLOR
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        
        return frame

    def _should_stop_display(self, frame: np.ndarray) -> bool:
        """Check if user wants to stop display."""
        cv2.imshow('Video Processing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Processing stopped by user")
            return True
        return False

    def _log_progress_if_needed(self, processed_frames: int, start_time: float, total_frames: int) -> None:
        """Log progress every 30 frames."""
        if processed_frames % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = processed_frames / elapsed if elapsed > 0 else 0
            progress = (self.frame_count / total_frames) * 100
            logger.info(
                f"Progress: {progress:.1f}% | FPS: {fps_current:.1f} | "
                f"Violations detected: {self.violations_detected}"
            )

    def _cleanup_resources(
        self, 
        cap: cv2.VideoCapture, 
        out_writer: Optional[cv2.VideoWriter], 
        display: bool
    ) -> None:
        """Clean up video processing resources."""
        cap.release()
        if out_writer:
            out_writer.release()
        if display:
            cv2.destroyAllWindows()

    def _compile_final_stats(
        self,
        video_path: str,
        output_path: Optional[str],
        video_info: Dict[str, int],
        processing_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile and log final processing statistics."""
        stats = {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': video_info['total_frames'],
            'processed_frames': processing_stats['processed_frames'],
            'processing_time': processing_stats['processing_time'],
            'avg_fps': (processing_stats['processed_frames'] / processing_stats['processing_time'] 
                       if processing_stats['processing_time'] > 0 else 0),
            'total_detections': processing_stats['total_detections'],
            'total_violations': self.violations_detected,
            'violation_frames': processing_stats['frame_violations'],
            'video_duration': video_info['total_frames'] / video_info['fps'] if video_info['fps'] > 0 else 0
        }

        self._log_final_stats(stats)
        return stats

    def _log_final_stats(self, stats: Dict[str, Any]) -> None:
        """Log final processing statistics."""
        logger.info("Video processing completed:")
        logger.info(f"  Processed {stats['processed_frames']}/{stats['total_frames']} frames")
        logger.info(f"  Processing time: {stats['processing_time']:.2f}s")
        logger.info(f"  Average FPS: {stats['avg_fps']:.2f}")
        logger.info(f"  Total violations: {stats['total_violations']}")

    # Private methods for drawing
    def _draw_detection_boxes(self, frame: np.ndarray, detections: List[Detection]) -> None:
        """Draw detection bounding boxes and labels."""
        for detection in detections:
            bbox_coords = self._get_bbox_coordinates(detection.bbox)
            color = self.COLOR_MAP.get(detection.class_name, self.DEFAULT_COLOR)
            
            cv2.rectangle(frame, bbox_coords[:2], bbox_coords[2:], color, 2)
            
            label = self._create_detection_label(detection)
            cv2.putText(
                frame, label, (bbox_coords[0], bbox_coords[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    def _draw_roi_boxes(self, frame: np.ndarray, rois: List[Dict[str, Any]]) -> None:
        """Draw ROI bounding boxes."""
        for roi in rois:
            if roi.get('active', True):
                coords = [int(coord) for coord in roi['coordinates']]
                cv2.rectangle(frame, coords[:2], coords[2:], self.VIOLATION_COLOR, 2)
                cv2.putText(
                    frame, roi['name'], (coords[0], coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.VIOLATION_COLOR, 2
                )

    def _draw_violation_boxes(self, frame: np.ndarray, violations: List[Violation]) -> None:
        """Draw violation bounding boxes and labels."""
        for violation in violations:
            bbox_coords = self._get_bbox_coordinates(violation.bbox)
            
            cv2.rectangle(frame, bbox_coords[:2], bbox_coords[2:], self.VIOLATION_COLOR, 4)
            cv2.putText(
                frame, "VIOLATION!", (bbox_coords[0], bbox_coords[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.VIOLATION_COLOR, 3
            )
            
            description = self._create_violation_description(violation)
            cv2.putText(
                frame, description, (bbox_coords[0], bbox_coords[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.VIOLATION_COLOR, 2
            )

    def _get_bbox_coordinates(self, bbox: BoundingBox) -> tuple:
        """Convert bounding box to integer coordinates."""
        return (int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2))

    def _create_detection_label(self, detection: Detection) -> str:
        """Create label for detection box."""
        label = f"{detection.class_name.value}: {detection.confidence:.2f}"
        
        if detection.class_name == DetectionClass.HAND and detection.hand_id is not None:
            label += f" (ID {detection.hand_id})"
        elif detection.track_id is not None:
            label += f" (ID {detection.track_id})"
            
        return label

    def _create_violation_description(self, violation: Violation) -> str:
        """Create description for violation."""
        description = violation.description or "No scooper"
        track_id = getattr(violation, 'track_id', None)
        
        if track_id is not None:
            description = f"[ID {track_id}] " + description
            
        return description

    # Private methods for violation saving
    def _save_violations(
        self, 
        violations: List[Violation], 
        frame_id: str, 
        timestamp: datetime, 
        frame: np.ndarray, 
        detections: List[Detection]
    ) -> None:
        """Save violations to database."""
        for violation in violations:
            record_metadata = self._create_violation_metadata(violation)
            frame_path = self._save_violation_frame(frame, frame_id)
            
            violation_record = ViolationRecord(
                frame_id=frame_id,
                camera_id=self.camera_id,
                timestamp=timestamp,
                violation_type=violation.type,
                roi_id=violation.roi_id,
                confidence=violation.confidence,
                frame_path=frame_path,
                bounding_boxes=detections,
                metadata=record_metadata
            )
            
            self.db.insert_violation(violation_record)
            self.violations_detected += 1

    def _create_violation_metadata(self, violation: Violation) -> Dict[str, Any]:
        """Create metadata dictionary for violation record."""
        metadata = {"description": violation.description}
        
        if violation.hand_id is not None:
            metadata["hand_id"] = violation.hand_id
        elif violation.track_id is not None:
            metadata["track_id"] = violation.track_id
            
        return metadata

    def _save_violation_frame(self, frame: np.ndarray, frame_id: str) -> str:
        """Save violation frame to disk."""
        os.makedirs('violation_frames', exist_ok=True)
        frame_path = f"violation_frames/{frame_id}.jpg"
        cv2.imwrite(frame_path, frame)
        return frame_path

    # Utility methods
    def _create_empty_result(self, frame: np.ndarray) -> Dict[str, Any]:
        """Create empty result dictionary."""
        return {
            "detections": [], 
            "violations": [], 
            "annotated_frame": frame
        }


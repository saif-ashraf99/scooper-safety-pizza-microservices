import cv2
import time
import logging
from typing import Optional
from base_reader import BaseReader

logger = logging.getLogger(__name__)

class CameraReader(BaseReader):
    """Reads from camera and publishes frames"""
    
    def read_frames(self, camera_id: int = 0, fps_limit: Optional[float] = 30):
        """Read from camera and publish frames"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, fps_limit)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_delay = self._calculate_frame_delay(fps_limit)
        
        logger.info(f"Reading from camera {camera_id}: {width}x{height}, {fps_limit} FPS")
        
        frame_count = 0
        start_time = time.time()
        self.is_running = True
        source = f"camera_{camera_id}"
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    continue
                
                # Publish frame
                self._publish_frame(frame, width, height, fps_limit, source)
                
                frame_count += 1
                self._log_progress(frame_count, start_time)
                
                # Control frame rate
                if frame_delay > 0:
                    time.sleep(frame_delay)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error reading camera: {e}")
            raise
        finally:
            self._cleanup(cap, frame_count)
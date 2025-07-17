import cv2
import time
import logging
from typing import Optional
from base_reader import BaseReader

logger = logging.getLogger(__name__)

class RTSPReader(BaseReader):
    """Reads from RTSP stream and publishes frames"""
    
    def read_frames(self, rtsp_url: str, fps_limit: Optional[float] = 30):
        """Read from RTSP stream and publish frames"""
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise ValueError(f"Cannot open RTSP stream: {rtsp_url}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_delay = self._calculate_frame_delay(fps_limit)
        
        logger.info(f"Reading from RTSP stream: {rtsp_url}")
        logger.info(f"Properties: {width}x{height}, {fps_limit} FPS")
        
        frame_count = 0
        start_time = time.time()
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from RTSP stream, retrying...")
                    time.sleep(1)
                    continue
                
                # Publish frame
                self._publish_frame(frame, width, height, fps_limit, rtsp_url)
                
                frame_count += 1
                self._log_progress(frame_count, start_time)
                
                # Control frame rate
                if frame_delay > 0:
                    time.sleep(frame_delay)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error reading RTSP stream: {e}")
            raise
        finally:
            self._cleanup(cap, frame_count)
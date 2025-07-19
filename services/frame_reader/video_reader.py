import cv2
import os
import time
import logging
from typing import Optional
from base_reader import BaseReader

logger = logging.getLogger(__name__)

class VideoReader(BaseReader):
    """Reads video files and publishes frames"""
    
    def read_frames(self, video_path: str, fps_limit: Optional[float] = None):
        """Read video file and publish frames"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame delay
        target_fps = fps_limit if fps_limit else original_fps
        frame_delay = self._calculate_frame_delay(target_fps)
        
        logger.info(f"Reading video: {video_path}")
        logger.info(f"Properties: {width}x{height}, {original_fps} FPS, {total_frames} frames")
        logger.info(f"Target FPS: {target_fps}")
        
        frame_count = 0
        start_time = time.time()
        self.is_running = True
        source = os.path.basename(video_path)
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                # Publish frame
                self._publish_frame(frame, width, height, target_fps, source)
                
                frame_count += 1
                self._log_progress(frame_count, start_time, total_frames)
                
                # Control frame rate
                if frame_delay > 0:
                    time.sleep(frame_delay)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error reading video: {e}")
            raise
        finally:
            self._cleanup(cap, frame_count)
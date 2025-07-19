import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
from frame_factory import FrameFactory

# Import shared modules
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import RabbitMQPublisher

logger = logging.getLogger(__name__)

class BaseReader(ABC):
    """Base class for all frame readers"""
    
    def __init__(self, rabbitmq_host: str = 'rabbitmq', rabbitmq_port: int = 5672,
                 rabbitmq_user: str = 'admin', rabbitmq_password: str = 'admin123'):
        self.publisher = RabbitMQPublisher(
            host=rabbitmq_host,
            port=rabbitmq_port,
            username=rabbitmq_user,
            password=rabbitmq_password
        )
        self.is_running = False
        self.frame_factory = FrameFactory()
    
    def stop(self):
        """Stop the frame reader service"""
        self.is_running = False
    
    def _publish_frame(self, frame, width: int, height: int, fps: float, source: str):
        """Publish a frame using the frame factory"""
        video_frame = self.frame_factory.create_frame(frame, width, height, fps, source)
        self.publisher.publish_frame(video_frame.dict())
    
    def _log_progress(self, frame_count: int, start_time: float, total_frames: Optional[int] = None):
        """Log processing progress"""
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            if total_frames:
                logger.info(f"Processed {frame_count}/{total_frames} frames, "
                          f"Current FPS: {current_fps:.2f}")
            else:
                logger.info(f"Processed {frame_count} frames, Current FPS: {current_fps:.2f}")
    
    def _calculate_frame_delay(self, fps_limit: Optional[float]) -> float:
        """Calculate frame delay based on FPS limit"""
        return 1.0 / fps_limit if fps_limit and fps_limit > 0 else 0
    
    def _cleanup(self, cap, frame_count: int):
        """Clean up resources"""
        cap.release()
        self.publisher.disconnect()
        logger.info(f"Finished processing {frame_count} frames")
    
    @abstractmethod
    def read_frames(self, **kwargs):
        """Abstract method to be implemented by subclasses"""
        pass
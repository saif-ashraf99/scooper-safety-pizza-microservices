import logging
from typing import Optional
from video_reader import VideoReader
from camera_reader import CameraReader
from rtsp_reader import RTSPReader

logger = logging.getLogger(__name__)

class FrameReaderService:
    """Main service that coordinates different frame readers"""
    
    def __init__(self, rabbitmq_host: str = 'rabbitmq', rabbitmq_port: int = 5672,
                 rabbitmq_user: str = 'admin', rabbitmq_password: str = 'admin123'):
        self.rabbitmq_config = {
            'rabbitmq_host': rabbitmq_host,
            'rabbitmq_port': rabbitmq_port,
            'rabbitmq_user': rabbitmq_user,
            'rabbitmq_password': rabbitmq_password
        }
        self.current_reader = None
        
    def read_video_file(self, video_path: str, fps_limit: Optional[float] = None):
        """Read video file and publish frames"""
        self.current_reader = VideoReader(**self.rabbitmq_config)
        self.current_reader.read_frames(video_path, fps_limit)
    
    def read_camera_stream(self, camera_id: int = 0, fps_limit: Optional[float] = 30):
        """Read from camera and publish frames"""
        self.current_reader = CameraReader(**self.rabbitmq_config)
        self.current_reader.read_frames(camera_id, fps_limit)
    
    def read_rtsp_stream(self, rtsp_url: str, fps_limit: Optional[float] = 30):
        """Read from RTSP stream and publish frames"""
        self.current_reader = RTSPReader(**self.rabbitmq_config)
        self.current_reader.read_frames(rtsp_url, fps_limit)
    
    def stop(self):
        """Stop the current reader service"""
        if self.current_reader:
            self.current_reader.stop()
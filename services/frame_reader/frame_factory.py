import uuid
from datetime import datetime
from frame_encoder import FrameEncoder

# Import shared modules
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import VideoFrame, FrameMetadata

class FrameFactory:
    """Creates VideoFrame objects from raw frame data"""
    
    def __init__(self):
        self.encoder = FrameEncoder()
    
    def create_frame(self, frame, width: int, height: int, fps: float, source: str) -> VideoFrame:
        """Create a VideoFrame from raw frame data"""
        frame_id = str(uuid.uuid4())
        timestamp = datetime.now()
        frame_data = self.encoder.encode_frame(frame)
        
        metadata = FrameMetadata(
            width=width,
            height=height,
            fps=fps,
            source=source
        )
        
        return VideoFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_data=frame_data,
            metadata=metadata
        )
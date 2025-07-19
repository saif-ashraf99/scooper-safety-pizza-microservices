import cv2
import base64
import logging

logger = logging.getLogger(__name__)

class FrameEncoder:
    """Handles frame encoding operations"""
    
    @staticmethod
    def encode_frame(frame) -> str:
        """Encode frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
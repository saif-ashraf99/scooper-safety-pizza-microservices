import cv2
import base64
import time
import logging
import os
import uuid
from datetime import datetime
from typing import Optional
import argparse

# Import shared modules
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import RabbitMQPublisher, VideoFrame, FrameMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FrameReaderService:
    def __init__(self, rabbitmq_host: str = 'rabbitmq', rabbitmq_port: int = 5672,
                 rabbitmq_user: str = 'admin', rabbitmq_password: str = 'admin123'):
        self.publisher = RabbitMQPublisher(
            host=rabbitmq_host,
            port=rabbitmq_port,
            username=rabbitmq_user,
            password=rabbitmq_password
        )
        self.is_running = False
        
    def encode_frame(self, frame) -> str:
        """Encode frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def read_video_file(self, video_path: str, fps_limit: Optional[float] = None):
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
        frame_delay = 1.0 / target_fps if target_fps > 0 else 0
        
        logger.info(f"Reading video: {video_path}")
        logger.info(f"Properties: {width}x{height}, {original_fps} FPS, {total_frames} frames")
        logger.info(f"Target FPS: {target_fps}")
        
        frame_count = 0
        start_time = time.time()
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                # Create frame message
                frame_id = str(uuid.uuid4())
                timestamp = datetime.now()
                frame_data = self.encode_frame(frame)
                
                metadata = FrameMetadata(
                    width=width,
                    height=height,
                    fps=target_fps,
                    source=os.path.basename(video_path)
                )
                
                video_frame = VideoFrame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_data=frame_data,
                    metadata=metadata
                )
                
                # Publish frame
                self.publisher.publish_frame(video_frame.dict())
                
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    logger.info(f"Processed {frame_count}/{total_frames} frames, "
                              f"Current FPS: {current_fps:.2f}")
                
                # Control frame rate
                if frame_delay > 0:
                    time.sleep(frame_delay)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error reading video: {e}")
            raise
        finally:
            cap.release()
            self.publisher.disconnect()
            logger.info(f"Finished processing {frame_count} frames")
    
    def read_camera_stream(self, camera_id: int = 0, fps_limit: Optional[float] = 30):
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
        
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0
        
        logger.info(f"Reading from camera {camera_id}: {width}x{height}, {fps_limit} FPS")
        
        frame_count = 0
        start_time = time.time()
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    continue
                
                # Create frame message
                frame_id = str(uuid.uuid4())
                timestamp = datetime.now()
                frame_data = self.encode_frame(frame)
                
                metadata = FrameMetadata(
                    width=width,
                    height=height,
                    fps=fps_limit,
                    source=f"camera_{camera_id}"
                )
                
                video_frame = VideoFrame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_data=frame_data,
                    metadata=metadata
                )
                
                # Publish frame
                self.publisher.publish_frame(video_frame.dict())
                
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    logger.info(f"Processed {frame_count} frames, Current FPS: {current_fps:.2f}")
                
                # Control frame rate
                if frame_delay > 0:
                    time.sleep(frame_delay)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error reading camera: {e}")
            raise
        finally:
            cap.release()
            self.publisher.disconnect()
            logger.info(f"Finished processing {frame_count} frames")
    
    def read_rtsp_stream(self, rtsp_url: str, fps_limit: Optional[float] = 30):
        """Read from RTSP stream and publish frames"""
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise ValueError(f"Cannot open RTSP stream: {rtsp_url}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0
        
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
                
                # Create frame message
                frame_id = str(uuid.uuid4())
                timestamp = datetime.now()
                frame_data = self.encode_frame(frame)
                
                metadata = FrameMetadata(
                    width=width,
                    height=height,
                    fps=fps_limit,
                    source=rtsp_url
                )
                
                video_frame = VideoFrame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_data=frame_data,
                    metadata=metadata
                )
                
                # Publish frame
                self.publisher.publish_frame(video_frame.dict())
                
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    logger.info(f"Processed {frame_count} frames, Current FPS: {current_fps:.2f}")
                
                # Control frame rate
                if frame_delay > 0:
                    time.sleep(frame_delay)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error reading RTSP stream: {e}")
            raise
        finally:
            cap.release()
            self.publisher.disconnect()
            logger.info(f"Finished processing {frame_count} frames")
    
    def stop(self):
        """Stop the frame reader service"""
        self.is_running = False

def main():
    parser = argparse.ArgumentParser(description='Frame Reader Service')
    parser.add_argument('--source', required=True, 
                       help='Video source: file path, camera ID (0,1,2...), or RTSP URL')
    parser.add_argument('--fps', type=float, default=30, 
                       help='Target FPS (default: 30)')
    parser.add_argument('--rabbitmq-host', default='rabbitmq', 
                       help='RabbitMQ host (default: rabbitmq)')
    parser.add_argument('--rabbitmq-port', type=int, default=5672, 
                       help='RabbitMQ port (default: 5672)')
    parser.add_argument('--rabbitmq-user', default='admin', 
                       help='RabbitMQ username (default: admin)')
    parser.add_argument('--rabbitmq-password', default='admin123', 
                       help='RabbitMQ password (default: admin123)')
    
    args = parser.parse_args()
    
    # Create frame reader service
    service = FrameReaderService(
        rabbitmq_host=args.rabbitmq_host,
        rabbitmq_port=args.rabbitmq_port,
        rabbitmq_user=args.rabbitmq_user,
        rabbitmq_password=args.rabbitmq_password
    )
    
    try:
        # Determine source type and start reading
        if args.source.startswith('rtsp://'):
            service.read_rtsp_stream(args.source, args.fps)
        elif args.source.isdigit():
            service.read_camera_stream(int(args.source), args.fps)
        else:
            service.read_video_file(args.source, args.fps)
    except Exception as e:
        logger.error(f"Failed to start frame reader: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


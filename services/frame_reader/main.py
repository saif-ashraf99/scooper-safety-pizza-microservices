import logging
import argparse
from frame_reader_service import FrameReaderService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
import logging
import os
import argparse

from services.detection.detection_logic import DetectionService

import sys
sys.path.append('/home/ubuntu/scooper-safety-pizza-microservices')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Detection Service')
    parser.add_argument('--mode', choices=['service', 'video'], default='service',
                       help='Operating mode: service (RabbitMQ) or video processing')
    parser.add_argument('--video-path', 
                       help='Path to video file (for video mode)')
    parser.add_argument('--output-path', 
                       help='Path to output video file')
    parser.add_argument('--model-path', 
                       help='Path to YOLO model file')
    parser.add_argument('--db-path', 
                       help='Database path')
    parser.add_argument('--rabbitmq-host', 
                       help='RabbitMQ host (use "none" to skip RabbitMQ)')
    parser.add_argument('--rabbitmq-port', type=int, 
                       help='RabbitMQ port')
    parser.add_argument('--rabbitmq-user', 
                       help='RabbitMQ username')
    parser.add_argument('--rabbitmq-password', 
                       help='RabbitMQ password')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Process every Nth frame (0 = all frames)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display during processing')
    parser.add_argument('--no-save-violations', action='store_true',
                       help='Disable saving violations to database')
    
    args = parser.parse_args()
    
    # Get configuration from environment variables or command line args
    rabbitmq_host = args.rabbitmq_host or os.getenv('RABBITMQ_HOST', 'rabbitmq')
    rabbitmq_port = args.rabbitmq_port or int(os.getenv('RABBITMQ_PORT', '5672'))
    rabbitmq_user = args.rabbitmq_user or os.getenv('RABBITMQ_USER', 'admin')
    rabbitmq_password = args.rabbitmq_password or os.getenv('RABBITMQ_PASSWORD', 'admin123')
    model_path = args.model_path or os.getenv('MODEL_PATH')
    db_path = args.db_path or os.getenv('DATABASE_PATH', 'violations.db')
    
    # Print configuration for debugging
    print(f"Mode: {args.mode}")
    if args.mode == 'video':
        print(f"Video path: {args.video_path}")
        print(f"Output path: {args.output_path}")
        print(f"Skip frames: {args.skip_frames}")
        print(f"Display: {not args.no_display}")
    else:
        print(f"RabbitMQ Configuration:")
        print(f"  Host: {rabbitmq_host}")
        print(f"  Port: {rabbitmq_port}")
        print(f"  User: {rabbitmq_user}")
    print(f"Database: {db_path}")
    print(f"Model: {model_path}")
    
    # Create detection service
    service = DetectionService(
        rabbitmq_host=rabbitmq_host,
        rabbitmq_port=rabbitmq_port,
        rabbitmq_user=rabbitmq_user,
        rabbitmq_password=rabbitmq_password,
        model_path=model_path,
        db_path=db_path
    )
    
    try:
        if args.mode == 'video':
            if not args.video_path:
                logger.error("Video path required for video mode")
                return 1
            
            # Process video
            stats = service.process_video(
                video_path=args.video_path,
                output_path=args.output_path,
                save_violations=not args.no_save_violations,
                display=not args.no_display,
                skip_frames=args.skip_frames
            )
            
            # Print final statistics
            print("\n" + "="*50)
            print("VIDEO PROCESSING COMPLETE")
            print("="*50)
            print(f"Input: {stats['video_path']}")
            if stats['output_path']:
                print(f"Output: {stats['output_path']}")
            print(f"Frames processed: {stats['processed_frames']}/{stats['total_frames']}")
            print(f"Processing time: {stats['processing_time']:.2f}s")
            print(f"Average FPS: {stats['avg_fps']:.2f}")
            print(f"Total violations: {stats['total_violations']}")
            print(f"Violation frames: {len(stats['violation_frames'])}")
            
        else:
            # Start RabbitMQ service mode
            service.start()
            
    except Exception as e:
        logger.error(f"Failed to run detection service: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
import logging
import os
import argparse

from services.detection.detection import DetectionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Detection Service')
    parser.add_argument(
        '-c', '--camera-id', type=str, required=True,
        help='Unique identifier for the camera'
    )
    parser.add_argument('--mode', choices=['service', 'video'], default='service',
                        help='Operating mode: service (RabbitMQ) or video processing')
    parser.add_argument('--video-path',
                        help='Path to video file (for video mode)'
    )
    parser.add_argument('--output-path',
                        help='Path to output video file'
    )
    parser.add_argument('--model-path',
                        help='Path to YOLO model file'
    )
    parser.add_argument('--db-path',
                        help='Database path'
    )
    parser.add_argument('--rabbitmq-host',
                        help='RabbitMQ host'
    )
    parser.add_argument('--rabbitmq-port', type=int,
                        help='RabbitMQ port'
    )
    parser.add_argument('--rabbitmq-user',
                        help='RabbitMQ username'
    )
    parser.add_argument('--rabbitmq-password',
                        help='RabbitMQ password'
    )
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Process every Nth frame (0 = all frames)'
    )
    parser.add_argument('--no-display', action='store_true',
                        help='Disable video display during processing'
    )
    parser.add_argument('--no-save-violations', action='store_true',
                        help='Disable saving violations to database'
    )

    args = parser.parse_args()

    # Get configuration from environment variables or command line args
    camera_id = args.camera_id
    rabbitmq_host = args.rabbitmq_host or os.getenv('RABBITMQ_HOST', 'localhost')
    rabbitmq_port = args.rabbitmq_port or int(os.getenv('RABBITMQ_PORT', '5672'))
    rabbitmq_user = args.rabbitmq_user or os.getenv('RABBITMQ_USER', 'guest')
    rabbitmq_password = args.rabbitmq_password or os.getenv('RABBITMQ_PASSWORD', 'guest')
    model_path = args.model_path or os.getenv('MODEL_PATH', '/home/saif/det/models/fine-tuned-yolo12m-v2.pt')
    db_path = args.db_path or os.getenv('DATABASE_PATH', 'violations.db')

    # Print configuration for debugging
    logger.info(f"Starting Detection Service with Camera ID: {camera_id}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == 'video':
        logger.info(f"Video path: {args.video_path}")
        logger.info(f"Output path: {args.output_path}")
        logger.info(f"Skip frames: {args.skip_frames}")
        logger.info(f"Display: {not args.no_display}")
    else:
        logger.info(f"RabbitMQ Configuration:")
        logger.info(f"  Host: {rabbitmq_host}")
        logger.info(f"  Port: {rabbitmq_port}")
        logger.info(f"  User: {rabbitmq_user}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Model: {model_path}")

    # Validate required args
    if args.mode == 'video' and not args.video_path:
        logger.error("Error: --video-path is required for video mode.")
        return 1

    # Create detection service
    service = DetectionService(
        camera_id=camera_id,
        model_path=model_path,
        db_path=db_path,
        rabbitmq_host=rabbitmq_host,
        rabbitmq_port=rabbitmq_port,
        rabbitmq_user=rabbitmq_user,
        rabbitmq_password=rabbitmq_password,
        enable_rabbit=(args.mode == 'service')
    )

    try:
        if args.mode == 'video':
            # Process video
            stats = service.process_video(
                video_path=args.video_path,
                output_path=args.output_path,
                save_violations=not args.no_save_violations,
                display=not args.no_display,
                skip_frames=args.skip_frames
            )

            # Print final statistics
            logger.info("\n" + "=" * 50)
            logger.info("VIDEO PROCESSING COMPLETE")
            logger.info("=" * 50)
            logger.info(f"Input: {stats['video_path']}")
            if stats['output_path']:
                logger.info(f"Output: {stats['output_path']}")
            logger.info(f"Frames processed: {stats['processed_frames']}/{stats['total_frames']}")
            logger.info(f"Processing time: {stats['processing_time']:.2f}s")
            logger.info(f"Average FPS: {stats['avg_fps']:.2f}")
            logger.info(f"Total violations: {stats['total_violations']}")
            logger.info(f"Violation frames: {len(stats['violation_frames'])}")

        else:
            # Start RabbitMQ service mode
            service.start()

    except Exception as e:
        logger.error(f"Failed to run detection service: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())



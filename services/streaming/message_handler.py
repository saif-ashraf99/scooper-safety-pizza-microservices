import asyncio
import logging
from websocket_manager import manager
from metrics import system_metrics

logger = logging.getLogger(__name__)

def handle_detection_result(message: dict, loop: asyncio.AbstractEventLoop):
    try:
        logger.info("📨 Received detection message from RabbitMQ")
        frame_data = message.get('frame_data')
        logger.info(f"Frame data present: {bool(frame_data)}")

        system_metrics['frames_processed'] += 1
        if message.get('violations'):
            system_metrics['violations_detected'] += len(message['violations'])

        websocket_message = {
            "type": "frame",
            "frame_id": message.get('frame_id'),
            "timestamp": message.get('timestamp'),
            "image_data": frame_data,
            "detections": message.get('detections', []),
            "violations": message.get('violations', [])
        }

        asyncio.run_coroutine_threadsafe(manager.broadcast(websocket_message), loop)

    except Exception as e:
        logger.error(f"Error handling detection result: {e}")
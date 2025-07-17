import logging
import asyncio
import time
from connection_manager import manager

# Shared metrics dict
system_metrics = {
    'frames_processed': 0,
    'violations_detected': 0,
    'processing_fps': 0.0,
    'queue_size': 0,
    'start_time': time.time()
}

# Handler for messages from RabbitMQ
def handle_detection_result(message: dict, loop: asyncio.AbstractEventLoop):
    try:
        logging.info("📨 Received detection message from RabbitMQ")
        frame_data = message.get('frame_data')
        logging.info(f"Frame data present: {bool(frame_data)}")

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

        # Send to all connected websockets
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(websocket_message),
            loop
        )
    except Exception as e:
        logging.error(f"Error handling detection result: {e}")
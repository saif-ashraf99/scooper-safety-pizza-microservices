import asyncio
import logging
import threading
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import RabbitMQConsumer
from config import config
from message_handler import handle_detection_result

logger = logging.getLogger(__name__)

# Global variables
rabbitmq_consumer = None
fastapi_event_loop = None

async def startup_event():
    global rabbitmq_consumer, fastapi_event_loop
    try:
        fastapi_event_loop = asyncio.get_running_loop()  # <-- capture main loop here

        rabbitmq_consumer = RabbitMQConsumer(
            host=config.RABBITMQ_HOST,
            port=config.RABBITMQ_PORT,
            username=config.RABBITMQ_USER,
            password=config.RABBITMQ_PASSWORD
        )

        thread = threading.Thread(
            target=rabbitmq_consumer.consume_detection_results,
            args=(lambda msg: handle_detection_result(msg, fastapi_event_loop),),
            daemon=True
        )
        thread.start()

        logger.info("Streaming service started and consumer thread launched.")
    except Exception as e:
        logger.error(f"Failed to start streaming service: {e}")

async def shutdown_event():
    """Cleanup on shutdown"""
    global rabbitmq_consumer
    if rabbitmq_consumer:
        rabbitmq_consumer.disconnect()
    logger.info("Streaming service shut down")

def get_rabbitmq_consumer():
    return rabbitmq_consumer
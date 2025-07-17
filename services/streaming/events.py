import asyncio
import logging
import threading
from shared import RabbitMQConsumer
from config import config
from consumer import handle_detection_result

rabbitmq_consumer = None
fastapi_event_loop = None

async def startup_event():
    global rabbitmq_consumer, fastapi_event_loop
    try:
        fastapi_event_loop = asyncio.get_running_loop()
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
        logging.info("Streaming service started and consumer thread launched.")
    except Exception as e:
        logging.error(f"Failed to start streaming service: {e}")

async def shutdown_event():
    global rabbitmq_consumer
    if rabbitmq_consumer:
        rabbitmq_consumer.disconnect()
    logging.info("Streaming service shut down")
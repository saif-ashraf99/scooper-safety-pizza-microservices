import pika
import json
import logging
import time
import os
from typing import Callable, Optional, Dict, Any
from datetime import datetime
import uuid
from pika.exceptions import AMQPConnectionError, ConnectionClosed

logger = logging.getLogger(__name__)


class RabbitMQClient:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 username: Optional[str] = None, password: Optional[str] = None):
        self.host = host if host is not None else os.getenv('RABBITMQ_HOST', 'rabbitmq')
        self.port = port if port is not None else int(os.getenv('RABBITMQ_PORT', 5672))
        self.username = username if username is not None else os.getenv('RABBITMQ_USER', 'admin')
        self.password = password if password is not None else os.getenv('RABBITMQ_PASSWORD', 'admin123')
        
        self.connection = None
        self.channel = None
        self.max_retries = 10
        self.retry_delay = 5
        self.connect()
    
    def connect(self):
        """Establish connection to RabbitMQ with retry logic"""
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
            connection_attempts=3,
            retry_delay=2
        )
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting to connect to RabbitMQ at {self.host}:{self.port} "
                           f"(attempt {attempt + 1}/{self.max_retries})")
                
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                
                self.channel.queue_declare(queue='video_frames', durable=True)
                self.channel.queue_declare(queue='detection_results', durable=True)
                
                logger.info("Successfully connected to RabbitMQ")
                return
                
            except (AMQPConnectionError, ConnectionClosed, OSError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay = min(self.retry_delay * 1.5, 30)
                else:
                    logger.error("All connection attempts failed")
                    raise Exception(f"Failed to connect to RabbitMQ after {self.max_retries} attempts")
    
    def ensure_connection(self):
        try:
            if not self.connection or self.connection.is_closed:
                logger.info("Connection is closed, reconnecting...")
                self.connect()
            elif not self.channel or self.channel.is_closed:
                logger.info("Channel is closed, creating new channel...")
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue='video_frames', durable=True)
                self.channel.queue_declare(queue='detection_results', durable=True)
        except Exception as e:
            logger.error(f"Error ensuring connection: {e}")
            self.connect()
    
    def disconnect(self):
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
    
    def publish_message(self, queue: str, message: Dict[str, Any], 
                       exchange: str = '', routing_key: str = None):
        if routing_key is None:
            routing_key = queue
        
        max_publish_retries = 3
        for attempt in range(max_publish_retries):
            try:
                self.ensure_connection()
                message.setdefault('timestamp', datetime.now().isoformat())
                message.setdefault('message_id', str(uuid.uuid4()))
                
                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=json.dumps(message, default=str),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        timestamp=int(time.time())
                    )
                )
                logger.debug(f"Published message to {queue}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to publish message (attempt {attempt + 1}): {e}")
                if attempt < max_publish_retries - 1:
                    time.sleep(1)
                    try:
                        self.connect()
                    except:
                        pass
                else:
                    logger.error(f"Failed to publish message to {queue} after {max_publish_retries} attempts")
                    raise
    
    def consume_messages(self, queue: str, callback: Callable, 
                        auto_ack: bool = False, prefetch_count: int = 1):
        while True:
            try:
                self.ensure_connection()
                self.channel.basic_qos(prefetch_count=prefetch_count)
                
                def wrapper(ch, method, properties, body):
                    try:
                        message = json.loads(body)
                        callback(message)
                        if not auto_ack:
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        if not auto_ack:
                            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                
                self.channel.basic_consume(
                    queue=queue,
                    on_message_callback=wrapper,
                    auto_ack=auto_ack
                )
                
                logger.info(f"Started consuming from {queue}")
                self.channel.start_consuming()
                
            except KeyboardInterrupt:
                logger.info("Stopping consumer...")
                try:
                    self.channel.stop_consuming()
                except:
                    pass
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                logger.info("Attempting to reconnect in 5 seconds...")
                time.sleep(5)
                try:
                    if self.channel:
                        self.channel.stop_consuming()
                except:
                    pass
    
    def get_queue_info(self, queue: str) -> Dict[str, Any]:
        try:
            self.ensure_connection()
            method = self.channel.queue_declare(queue=queue, passive=True)
            return {
                'queue': queue,
                'message_count': method.method.message_count,
                'consumer_count': method.method.consumer_count
            }
        except Exception as e:
            logger.error(f"Failed to get queue info for {queue}: {e}")
            return {'queue': queue, 'message_count': 0, 'consumer_count': 0}
    
    def purge_queue(self, queue: str):
        try:
            self.ensure_connection()
            self.channel.queue_purge(queue=queue)
            logger.info(f"Purged queue {queue}")
        except Exception as e:
            logger.error(f"Failed to purge queue {queue}: {e}")
            raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class RabbitMQPublisher(RabbitMQClient):
    def publish_frame(self, frame_data: Dict[str, Any]):
        self.publish_message('video_frames', frame_data)
    
    def publish_detection_result(self, result_data: Dict[str, Any]):
        self.publish_message('detection_results', result_data)


class RabbitMQConsumer(RabbitMQClient):
    def consume_frames(self, callback: Callable):
        self.consume_messages('video_frames', callback)
    
    def consume_detection_results(self, callback: Callable):
        self.consume_messages('detection_results', callback)

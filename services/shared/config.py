import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class RabbitMQConfig:
    host: str = os.getenv('RABBITMQ_HOST', 'localhost')
    port: int = int(os.getenv('RABBITMQ_PORT', 5672))
    username: str = os.getenv('RABBITMQ_USER', 'guest')
    password: str = os.getenv('RABBITMQ_PASSWORD', 'guest')
    
    @property
    def connection_url(self) -> str:
        return f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"

@dataclass
class DatabaseConfig:
    path: str = os.getenv('DATABASE_PATH', 'violations.db')
    
@dataclass
class StreamingConfig:
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', 8000))
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    cors_origins: str = os.getenv('CORS_ORIGINS', '*')
    
@dataclass
class DetectionConfig:
    model_path: Optional[str] = os.getenv('MODEL_PATH')
    confidence_threshold: float = float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))
    scooper_distance_threshold: float = float(os.getenv('SCOOPER_DISTANCE_THRESHOLD', 100.0))
    
@dataclass
class FrameReaderConfig:
    fps_limit: float = float(os.getenv('FPS_LIMIT', 30.0))
    video_source: str = os.getenv('VIDEO_SOURCE', '0')
    
@dataclass
class LoggingConfig:
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_path: Optional[str] = os.getenv('LOG_FILE_PATH')

@dataclass
class Config:
    rabbitmq: RabbitMQConfig = RabbitMQConfig()
    database: DatabaseConfig = DatabaseConfig()
    streaming: StreamingConfig = StreamingConfig()
    detection: DetectionConfig = DetectionConfig()
    frame_reader: FrameReaderConfig = FrameReaderConfig()
    logging: LoggingConfig = LoggingConfig()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        return cls(
            rabbitmq=RabbitMQConfig(),
            database=DatabaseConfig(),
            streaming=StreamingConfig(),
            detection=DetectionConfig(),
            frame_reader=FrameReaderConfig(),
            logging=LoggingConfig()
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate RabbitMQ config
        if not self.rabbitmq.host:
            errors.append("RabbitMQ host is required")
        
        if not (1 <= self.rabbitmq.port <= 65535):
            errors.append("RabbitMQ port must be between 1 and 65535")
        
        # Validate streaming config
        if not (1 <= self.streaming.port <= 65535):
            errors.append("Streaming port must be between 1 and 65535")
        
        # Validate detection config
        if not (0.0 <= self.detection.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        # Validate frame reader config
        if self.frame_reader.fps_limit <= 0:
            errors.append("FPS limit must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        
        return True

# Global configuration instance
config = Config.from_env()


import os

class Config:
    RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq')
    RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
    RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'admin')
    RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'admin123')
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'violations.db')
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

config = Config()
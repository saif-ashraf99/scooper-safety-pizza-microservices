.PHONY: help build up down logs test clean install dev

# Default target
help:
	@echo "Pizza Violation Detection System"
	@echo "Available commands:"
	@echo "  build     - Build all Docker images"
	@echo "  up        - Start all services"
	@echo "  down      - Stop all services"
	@echo "  logs      - View logs from all services"
	@echo "  test      - Run all tests"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  install   - Install Python dependencies"
	@echo "  dev       - Start development environment"

# Build Docker images
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d
	@echo "Services started. Access the application at http://localhost:8000"

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Run tests
test:
	python -m pytest tests/ -v

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Install dependencies
install:
	pip install -r requirements.txt

# Development environment
dev:
	@echo "Starting development environment..."
	@echo "1. Start RabbitMQ:"
	@echo "   docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.12-management"
	@echo "2. Start services in separate terminals:"
	@echo "   cd services/streaming && python main.py"
	@echo "   cd services/detection && python main.py"
	@echo "   cd services/frame_reader && python main.py --source 0"

# Quick start with sample video
demo:
	@echo "Starting demo with sample video..."
	docker-compose up -d rabbitmq streaming_service detection_service
	@echo "Waiting for services to start..."
	sleep 10
	@echo "Demo ready at http://localhost:8000"


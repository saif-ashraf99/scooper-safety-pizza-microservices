# Pizza Violation Detection System

A microservices-based computer vision system for detecting scooper hygiene violations in pizza stores using FastAPI and RabbitMQ.

## 🎯 Overview

This system monitors pizza store workers to ensure they use scoopers when handling ingredients in designated ROI (Region of Interest) areas. It uses computer vision to detect hands, scoopers, and violations in real-time, providing alerts and maintaining violation records.

## 🏗️ Architecture

The system consists of five main microservices:

1. **Frame Reader Service** - Reads video frames from files, cameras, or RTSP streams
2. **Detection Service** - Performs object detection and violation logic using YOLO
3. **Streaming Service** - FastAPI-based service providing REST API and WebSocket streaming
4. **Streamlit Frontend** - Interactive web dashboard for monitoring and management
5. **RabbitMQ** - Message broker for inter-service communication

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- RabbitMQ (included in Docker Compose)

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project:**

   ```bash
   cd pizza_violation_detection
   ```
2. **Start all services:**

   ```bash
   docker-compose up --build
   ```
3. **Access the application:**

   - **Streamlit Dashboard**: http://localhost:8501 (Main Interface)
   - FastAPI Backend: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - RabbitMQ Management: http://localhost:15672 (admin/admin123)
4. **Add a video source:**

   ```bash
   # Copy your video file to the videos directory
   cp /path/to/your/video.mp4 ./videos/

   # Update docker-compose.yml to specify the video source
   command: ["python", "main.py", "--source", "/app/videos/your_video.mp4", "--fps", "10"]

   # Restart the frame reader service
   docker-compose restart frame_reader
   ```

### Local Development Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
2. **Start RabbitMQ:**

   ```bash
   docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.12-management
   ```
3. **Start services individually:**

   ```bash
   # Terminal 1 - Streaming Service
   cd services/streaming
   python main.py

   # Terminal 2 - Detection Service
   cd services/detection
   python main.py

   # Terminal 3 - Frame Reader Service
   cd services/frame_reader
   python main.py --source /path/to/video.mp4 --fps 10

   # Terminal 4 - Streamlit Frontend
   cd frontend
   streamlit run app.py
   ```

## 📁 Project Structure

```
pizza_violation_detection/
├── services/
│   ├── streaming/          # FastAPI streaming service
│   ├── detection/          # Object detection service
│   └── frame_reader/       # Video frame reader service
├── frontend/               # Streamlit web dashboard
├── shared/                 # Shared modules and utilities
├── tests/                  # Unit and integration tests
├── data/                   # Database and persistent data
├── videos/                 # Video files for processing
├── models/                 # YOLO model files
├── docs/         	    # Project docs
├── violation_frames/       # Saved violation frames
├── docker-compose.yml      # Docker orchestration
└── README.md
```

## 🔧 Configuration

The system uses environment variables for configuration:

### RabbitMQ Configuration

- `RABBITMQ_HOST` - RabbitMQ host (default: localhost)
- `RABBITMQ_PORT` - RabbitMQ port (default: 5672)
- `RABBITMQ_USER` - RabbitMQ username (default: guest)
- `RABBITMQ_PASSWORD` - RabbitMQ password (default: guest)

### Streaming Service Configuration

- `HOST` - Service host (default: 0.0.0.0)
- `PORT` - Service port (default: 8000)
- `DATABASE_PATH` - SQLite database path (default: violations.db)
- `DEBUG` - Debug mode (default: false)

### Detection Service Configuration

- `MODEL_PATH` - Path to YOLO model file
- `CONFIDENCE_THRESHOLD` - Detection confidence threshold (default: 0.7)
- `SCOOPER_DISTANCE_THRESHOLD` - Distance threshold for scooper detection (default: 100.0)

### Frame Reader Configuration

- `VIDEO_SOURCE` - Video source (file path, camera ID, or RTSP URL)
- `FPS_LIMIT` - Target FPS for processing (default: 30.0)

## 📊 API Endpoints

### REST API

- `GET /health` - Health check
- `GET /api/violations/summary` - Violation statistics
- `GET /api/violations` - Detailed violation records (with pagination)
- `GET /api/rois` - ROI configurations
- `POST /api/rois` - Create/update ROI configuration
- `GET /api/status` - System status and metrics

### WebSocket

- `WS /ws/video` - Real-time video stream with detections

### Example API Usage

```bash
# Get violation summary
curl http://localhost:8000/api/violations/summary

# Get recent violations
curl "http://localhost:8000/api/violations?limit=10&offset=0"

# Create a new ROI
curl -X POST http://localhost:8000/api/rois \
  -H "Content-Type: application/json" \
  -d '{
    "id": "sauce_container",
    "name": "Sauce Container",
    "coordinates": [200, 200, 400, 400],
    "active": true,
    "violation_type": "no_scooper"
  }'
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/test_shared.py -v

# Integration tests
python -m pytest tests/test_integration.py -v

# All tests
python -m pytest tests/ -v
```

## 🔍 Monitoring and Logging

### Logging

- Structured logging with configurable levels
- Automatic log rotation
- Performance monitoring decorators

### Metrics

- Frames processed count
- Violations detected count
- Processing FPS
- Queue sizes
- Service uptime

### Health Checks

- Service health endpoints
- Docker health checks
- RabbitMQ connection monitoring

## 🎥 Video Sources

The system supports multiple video sources:

1. **Video Files**: MP4, AVI, MOV, etc.

   ```bash
   python main.py --source /path/to/video.mp4
   ```
2. **Camera**: USB cameras or built-in webcams

   ```bash
   python main.py --source 0  # Camera ID
   ```
3. **RTSP Streams**: IP cameras or streaming sources

   ```bash
   python main.py --source rtsp://camera_ip:port/stream
   ```

## 🤖 YOLO Model Integration

The system is designed to work with YOLO models trained to detect:

- Hand
- Person
- Pizza
- Scooper

### Using a Custom Model

1. Place your YOLO model file in the `models/` directory
2. Set the `MODEL_PATH` environment variable:
   ```bash
   export MODEL_PATH=/app/models/your_model.pt
   ```
3. Restart the detection service

## 🔒 Production Considerations

### Security

- Use strong RabbitMQ credentials
- Enable HTTPS for production deployment
- Implement authentication for API endpoints
- Secure database access

### Performance

- Adjust FPS limits based on hardware capabilities
- Monitor memory usage with high-resolution videos
- Use GPU acceleration for YOLO inference
- Implement horizontal scaling for multiple cameras

### Reliability

- Set up service monitoring and alerting
- Implement automatic service restart policies
- Use persistent volumes for data storage
- Regular database backups

## 🐛 Troubleshooting

### Common Issues

1. **RabbitMQ Connection Failed**

   - Check if RabbitMQ is running: `docker ps | grep rabbitmq`
   - Verify connection parameters in environment variables
   - Check firewall settings
2. **Video Source Not Found**

   - Verify video file path exists
   - Check camera permissions for camera sources
   - Test RTSP URL with VLC or similar player
3. **High Memory Usage**

   - Reduce FPS limit
   - Lower video resolution
   - Adjust RabbitMQ queue sizes
4. **Slow Processing**

   - Check CPU/GPU utilization
   - Reduce detection confidence threshold
   - Optimize YOLO model size

### Logs and Debugging

```bash
# View service logs
docker-compose logs -f streaming_service
docker-compose logs -f detection_service
docker-compose logs -f frame_reader

# Check RabbitMQ queues
# Access http://localhost:15672 and login with admin/admin123
```

## 📈 Performance Benchmarks

Typical performance on modern hardware:

- **Processing Speed**: 15-30 FPS (1080p video)
- **Detection Latency**: 50-150ms per frame
- **Memory Usage**: 2-4GB (depending on video resolution)
- **Storage**: ~1MB per violation frame

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 🆘 Support

For support and questions:

- Check the troubleshooting section above
- Review the API documentation at `/docs`
- Check service logs for error messages
- Verify configuration parameters

## 🔄 Version History

- **v1.0.0** - Initial production release
  - FastAPI streaming service
  - RabbitMQ message broker integration
  - YOLO-based object detection
  - Real-time violation detection
  - Docker containerization
  - Comprehensive testing suite

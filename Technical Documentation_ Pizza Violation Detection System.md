# Technical Documentation: Pizza Violation Detection System

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Service-Specific Documentation](#3-service-specific-documentation)
   * [3.1 Frame Reader Service](#31-frame-reader-service)
   * [3.2 Detection Service](#32-detection-service)
   * [3.3 Streaming Service (FastAPI)](#33-streaming-service-fastapi)
   * [3.4 Streamlit Frontend](#34-streamlit-frontend)
4. [Shared Modules](#4-shared-modules)
   * [4.1 Data Models (`models.py`)](#41-data-models-modelspy)
   * [4.2 Database Operations (`database.py`)](#42-database-operations-databasepy)
   * [4.3 RabbitMQ Client (`rabbitmq_client.py`)](#43-rabbitmq-client-rabbitmqclientpy)
   * [4.4 Configuration (`config.py`)](#44-configuration-configpy)
   * [4.5 Logging (`logging_config.py`)](#45-logging-logging_configpy)
5. [Testing Strategy](#5-testing-strategy)
   * [5.1 Unit Tests](#51-unit-tests)
   * [5.2 Integration Tests](#52-integration-tests)
6. [Deployment](#6-deployment)
   * [6.1 Docker Compose](#61-docker-compose)
   * [6.2 Environment Variables](#62-environment-variables)
7. [API Reference](#7-api-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [Future Enhancements](#9-future-enhancements)

## 1. Introduction

The Pizza Violation Detection System is a microservices-based computer vision solution designed to monitor and ensure hygiene compliance in pizza stores. Specifically, it detects whether workers are using a scooper when handling ingredients in designated Regions of Interest (ROIs). The system leverages real-time video processing, object detection, and a message-driven architecture to identify and log violations.

This document provides a comprehensive technical overview of the system, detailing its architecture, individual components, shared utilities, testing methodologies, and deployment procedures. It serves as a guide for developers, and anyone seeking an in-depth understanding of the system's inner workings.

## 2. System Architecture

The system is built upon a modular microservices architecture, promoting scalability, maintainability, and fault isolation. Communication between services is primarily facilitated by a central RabbitMQ message broker. The core components are:

- **Frame Reader Service**: Ingests video streams and publishes individual frames to the message broker.
- **Detection Service**: Subscribes to video frames, performs object detection (YOLO), applies violation logic, and publishes detection results.
- **Streaming Service (FastAPI)**: Provides RESTful APIs for system management and a WebSocket endpoint for real-time streaming of annotated video frames and violation data to the frontend.
- **Streamlit Frontend**: An interactive web dashboard for visualizing live video, system metrics, violation history, and managing ROIs.
- **RabbitMQ**: Acts as the central message queue, ensuring reliable and asynchronous communication between services.
- **Database (SQLite/PostgreSQL)**: Stores violation records and ROI configurations.

**Key Architectural Principles:**

- **Loose Coupling**: Services operate independently, communicating via messages rather than direct calls.
- **Asynchronous Communication**: RabbitMQ ensures that services can process data at their own pace without blocking.
- **Scalability**: Individual services can be scaled horizontally based on load.
- **Modularity**: Clear separation of concerns simplifies development, testing, and maintenance.
- **Real-time Processing**: Designed to handle continuous video streams and provide immediate feedback.

## 3. Service-Specific Documentation

### 3.1 Frame Reader Service

**Location**: `services/frame_reader/`

**Purpose**: The Frame Reader Service is responsible for ingesting video streams from various sources (video files, live camera feeds, or RTSP streams) and publishing individual frames to the RabbitMQ `video_frames` queue. This service acts as the entry point for visual data into the system.

**Key Components**:

- `main.py`: Contains the core logic for video capture, frame processing, and publishing to RabbitMQ.
- `requirements.txt`: Lists Python dependencies, primarily `opencv-python` for video processing and `pika` for RabbitMQ communication.
- `Dockerfile`: Defines the containerization environment for the service.

**Functionality**:

- **Video Ingestion**: Supports reading from:
  - Local video files (e.g., MP4, AVI)
  - Connected webcams (by camera ID)
  - RTSP (Real-Time Streaming Protocol) URLs for IP cameras.
- **Frame Encoding**: Converts raw video frames into base64-encoded JPEG strings for efficient transmission over RabbitMQ.
- **Frame Metadata**: Attaches metadata such as `frame_id`, `timestamp`, `width`, `height`, `fps`, and `source` to each published frame.
- **Rate Limiting**: Can be configured to publish frames at a specified FPS to control the processing load on downstream services.
- **Error Handling**: Includes basic error handling for video capture failures and RabbitMQ connection issues.

**Configuration (Environment Variables)**:

- `RABBITMQ_HOST`: Hostname or IP address of the RabbitMQ server (default: `localhost`)
- `RABBITMQ_PORT`: Port of the RabbitMQ server (default: `5672`)
- `RABBITMQ_USER`: Username for RabbitMQ authentication (default: `guest`)
- `RABBITMQ_PASSWORD`: Password for RabbitMQ authentication (default: `guest`)
- `VIDEO_SOURCE`: Specifies the video input source (file path, camera ID, or RTSP URL).
- `FPS_LIMIT`: Target frames per second for publishing (default: `30.0`).

**Example Usage (from `main.py`)**:

```python
# Example for reading from a video file
service.read_video_file("/app/videos/sample.mp4", args.fps)

# Example for reading from a camera
service.read_camera_stream(0, args.fps)

# Example for reading from an RTSP stream
service.read_rtsp_stream("rtsp://your_camera_ip/stream", args.fps)
```

**Message Format (Published to `video_frames` queue)**:

```json
{
  "frame_id": "uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "frame_data": "base64_encoded_image",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "source": "camera_1"
  }
}
```

**Dependencies**: `opencv-python`, `pika`, `numpy`

**Docker Image**: Built from `services/frame_reader/Dockerfile`

### 3.2 Detection Service

**Location**: `services/detection/`

**Purpose**: The Detection Service is the core computer vision component of the system. It subscribes to video frames from the RabbitMQ `video_frames` queue, performs object detection using a YOLO model, applies the violation detection logic, and then publishes the results (including annotated frames and violation statuses) to the RabbitMQ `detection_results` queue. It also persists violation records to the database.

**Key Components**:

- `main.py`: Contains the logic for consuming frames, running the detection model, applying violation rules, drawing annotations, and publishing results.
- `requirements.txt`: Lists Python dependencies, including `opencv-python`, `pika`, `numpy`, and `ultralytics` (for YOLO).
- `Dockerfile`: Defines the containerization environment for the service.

**Functionality**:

- **Frame Consumption**: Continuously consumes `VideoFrame` messages from the `video_frames` queue.
- **Object Detection**: Utilizes a pre-trained YOLO model to identify objects such as `hand`, `person`, `pizza`, and `scooper` within each frame.
- **Violation Logic**: Implements the business rules for detecting scooper hygiene violations:
  - Identifies if a `hand` enters a predefined Region of Interest (ROI).
  - Checks for the presence and proximity of a `scooper` when a hand is within an ROI.
  - Flags a violation if a hand is in an ROI without a scooper being used.
- **ROI Management**: Retrieves ROI configurations from the shared database to define critical zones.
- **Annotation**: Draws bounding boxes for detected objects, highlights ROIs, and visually indicates violations directly on the video frames.
- **Database Persistence**: Stores detailed `ViolationRecord` data (frame ID, timestamp, violation type, ROI, bounding boxes, metadata) in the central database.
- **Result Publishing**: Publishes `DetectionResult` messages containing processed frames (with annotations), detected objects, and violation information to the `detection_results` queue.

**Configuration (Environment Variables)**:

- `RABBITMQ_HOST`: Hostname or IP address of the RabbitMQ server (default: `localhost`)
- `RABBITMQ_PORT`: Port of the RabbitMQ server (default: `5672`)
- `RABBITMQ_USER`: Username for RabbitMQ authentication (default: `guest`)
- `RABBITMQ_PASSWORD`: Password for RabbitMQ authentication (default: `guest`)
- `MODEL_PATH`: Path to the YOLO model file (e.g., `/app/models/yolo_model.pt`).
- `DATABASE_PATH`: Path to the SQLite database file (default: `violations.db`).
- `CONFIDENCE_THRESHOLD`: Minimum confidence score for object detections to be considered valid (default: `0.7`).
- `SCOOPER_DISTANCE_THRESHOLD`: Maximum pixel distance between a hand and a scooper for the scooper to be considered

used (default: `100.0`).

**Message Format (Consumed from `video_frames` queue)**:

(See Frame Reader Service documentation for `VideoFrame` format)

**Message Format (Published to `detection_results` queue)**:

```json
{
  "frame_id": "uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "detections": [
    {
      "class": "hand",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "violations": [
    {
      "type": "no_scooper",
      "roi_id": "protein_container",
      "confidence": 0.9,
      "bbox": [x1, y1, x2, y2],
      "description": "Hand detected in ROI without scooper"
    }
  ],
  "frame_data": "base64_encoded_image_with_detections",
  "processing_time": 0.15
}
```

**Dependencies**: `opencv-python`, `pika`, `numpy`, `ultralytics`

**Docker Image**: Built from `services/detection/Dockerfile`

### 3.3 Streaming Service (FastAPI)

**Location**: `services/streaming/`

**Purpose**: The Streaming Service, built with FastAPI, serves as the primary interface for the system. It provides RESTful APIs for querying violation data and system status, and a WebSocket endpoint for real-time streaming of annotated video frames to connected clients (e.g., the Streamlit frontend).

**Key Components**:

- `main.py`: Contains the FastAPI application, defining API routes, WebSocket endpoint, and logic for consuming detection results from RabbitMQ and broadcasting them.
- `requirements.txt`: Lists Python dependencies, including `fastapi`, `uvicorn`, `pydantic`, and `websockets`.
- `Dockerfile`: Defines the containerization environment for the service.

**Functionality**:

- **REST API**: Exposes several endpoints for:
  - **Health Check** (`GET /health`): Basic service health status.
  - **Violation Summary** (`GET /api/violations/summary`): Aggregated statistics on detected violations.
  - **Violation Details** (`GET /api/violations`): Paginated access to detailed violation records stored in the database.
  - **ROI Configuration** (`GET /api/rois`, `POST /api/rois`): Retrieve and update Region of Interest definitions.
  - **System Status** (`GET /api/status`): Provides real-time metrics on frames processed, violations detected, processing FPS, and service uptime.
- **WebSocket Streaming** (`/ws/video`):
  - Accepts WebSocket connections from clients.
  - Consumes `DetectionResult` messages from the RabbitMQ `detection_results` queue.
  - Broadcasts annotated video frames and associated detection/violation data to all connected WebSocket clients in real-time.
- **CORS Support**: Configured with Cross-Origin Resource Sharing (CORS) to allow frontend applications from different origins to connect.
- **Database Integration**: Interacts with the shared database module to retrieve and store data.
- **Metrics Collection**: Tracks internal metrics like frames processed and violations detected.

**Configuration (Environment Variables)**:

- `RABBITMQ_HOST`: Hostname or IP address of the RabbitMQ server (default: `localhost`)
- `RABBITMQ_PORT`: Port of the RabbitMQ server (default: `5672`)
- `RABBITMQ_USER`: Username for RabbitMQ authentication (default: `guest`)
- `RABBITMQ_PASSWORD`: Password for RabbitMQ authentication (default: `guest`)
- `DATABASE_PATH`: Path to the SQLite database file (default: `violations.db`).
- `HOST`: Host address for the FastAPI application (default: `0.0.0.0`)
- `PORT`: Port for the FastAPI application (default: `8000`)
- `DEBUG`: Boolean flag for debug mode (default: `false`)

**Dependencies**: `fastapi`, `uvicorn`, `pydantic`, `websockets`, `pika`

**Docker Image**: Built from `services/streaming/Dockerfile`

### 3.4 Streamlit Frontend

**Location**: `frontend/`

**Purpose**: The Streamlit Frontend provides an intuitive and interactive web-based dashboard for users to monitor the Pizza Violation Detection System in real-time. It visualizes the live video feed with detections, displays key system metrics, shows a history of violations, and allows for basic configuration of ROIs.

**Key Components**:

- `app.py`: The main Streamlit application script, defining the UI layout, data fetching logic, and real-time updates.
- `requirements.txt`: Lists Python dependencies, including `streamlit`, `requests`, `pandas`, `plotly`, and `websocket-client`.
- `Dockerfile`: Defines the containerization environment for the Streamlit application.

**Functionality**:

- **Live Video Display**: Connects to the FastAPI Streaming Service's WebSocket endpoint (`/ws/video`) to receive and display annotated video frames in real-time. Uses `st.image` to update the video feed continuously.
- **Violation Alerts**: Prominently displays alerts when violations are detected in the live stream.
- **System Metrics Dashboard**: Fetches data from FastAPI REST APIs (`/api/violations/summary`, `/api/status`) to show:
  - Total violations and violations by type.
  - Processing status, frames processed, and processing FPS.
  - System uptime and service health indicators.
- **Recent Violations Table**: Displays a paginated table of the most recent violation records fetched from `/api/violations`.
- **ROI Configuration**: Provides a simple form to add new Regions of Interest by sending `POST` requests to the FastAPI `/api/rois` endpoint.
- **API Configuration**: Allows users to configure the FastAPI backend URL from the sidebar.
- **Auto-Refresh**: Option to automatically refresh static data at configurable intervals.

**Configuration (Environment Variables)**:

- `STREAMLIT_SERVER_PORT`: Port for the Streamlit application (default: `8501`)
- `STREAMLIT_SERVER_ADDRESS`: Address for the Streamlit application (default: `0.0.0.0`)
- `STREAMLIT_SERVER_HEADLESS`: Run Streamlit in headless mode (default: `true`)

**Dependencies**: `streamlit`, `requests`, `pandas`, `plotly`, `websocket-client`

**Docker Image**: Built from `frontend/Dockerfile`

## 4. Shared Modules

The `shared/` directory contains common Python modules and utilities used across multiple microservices. This promotes code reusability, consistency, and simplifies maintenance.

### 4.1 Data Models (`models.py`)

**Purpose**: Defines Pydantic models for all data structures exchanged between services (via RabbitMQ or API endpoints) and stored in the database. This ensures data validation, serialization, and clear data contracts.

**Key Models**:

- `BoundingBox`: Represents the coordinates of a bounding box (`x1`, `y1`, `x2`, `y2`).
- `Detection`: Describes a detected object, including its `class_name` (e.g., `hand`, `scooper`), `confidence`, and `bbox`.
- `Violation`: Represents a detected violation, including `type` (e.g., `no_scooper`), `roi_id`, `confidence`, `bbox`, and an optional `description`.
- `FrameMetadata`: Contains metadata about a video frame (e.g., `width`, `height`, `fps`, `source`).
- `VideoFrame`: The complete data structure for a video frame, including `frame_id`, `timestamp`, `frame_data` (base64 encoded image), and `metadata`.
- `DetectionResult`: Encapsulates the results of the detection service, including `frame_id`, `timestamp`, lists of `detections` and `violations`, `frame_data` (annotated image), and `processing_time`.
- `ROI`: Defines a Region of Interest, including `id`, `name`, `coordinates`, `active` status, and `violation_type`.
- `ViolationRecord`: The data structure for storing violation events in the database.
- `ViolationSummary`, `SystemStatus`, `HealthCheck`, `WebSocketMessage`: Models for API responses and WebSocket communication.

**Usage**: These models are imported and used by FastAPI for request/response validation, by RabbitMQ clients for message serialization, and by the database module for data storage.

### 4.2 Database Operations (`database.py`)

**Purpose**: Provides a centralized interface for interacting with the SQLite database. It handles database initialization, connection management, and CRUD (Create, Read, Update, Delete) operations for `violations` and `rois` tables.

**Key Features**:

- **`Database` Class**: Manages the database connection and operations.
- **`init_database()`**: Creates `violations` and `rois` tables if they don't exist, and inserts a default ROI (`protein_container`).
- **`get_connection()`**: A context manager for safely acquiring and releasing database connections.
- **`insert_violation()`**: Stores a new `ViolationRecord` in the `violations` table.
- **`get_violations()`**: Retrieves violation records with support for pagination and time-based filtering.
- **`get_violation_count()`**: Returns the total number of violations.
- **`get_violation_summary()`**: Provides aggregated statistics on violations.
- **`get_rois()`**: Retrieves all configured ROIs.
- **`upsert_roi()`**: Inserts a new ROI or updates an existing one.

**Database Schema (SQLite)**:

```sql
CREATE TABLE violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id VARCHAR(36) NOT NULL,
    timestamp DATETIME NOT NULL,
    violation_type VARCHAR(50) NOT NULL,
    roi_id VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    frame_path VARCHAR(255),
    bounding_boxes TEXT, -- JSON string
    metadata TEXT, -- JSON string
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rois (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    coordinates TEXT NOT NULL, -- JSON array [x1, y1, x2, y2]
    active BOOLEAN DEFAULT TRUE,
    violation_type VARCHAR(50) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 4.3 RabbitMQ Client (`rabbitmq_client.py`)

**Purpose**: Provides a robust and reusable client for interacting with the RabbitMQ message broker. It abstracts away the complexities of `pika` (the Python AMQP client library) and offers simplified methods for connecting, publishing, and consuming messages.

**Key Features**:

- **`RabbitMQClient` Class**: Base class for RabbitMQ interactions.
- **`connect()` / `disconnect()`**: Manages connection lifecycle with retry mechanisms.
- **`publish_message()`**: Generic method to publish messages to a specified queue, ensuring message persistence.
- **`consume_messages()`**: Generic method to consume messages from a queue, with support for callbacks and acknowledgment.
- **`RabbitMQPublisher`**: A specialized class for publishing `VideoFrame` and `DetectionResult` messages to their respective queues.
- **`RabbitMQConsumer`**: A specialized class for consuming `VideoFrame` and `DetectionResult` messages.
- **Queue Declaration**: Automatically declares necessary queues (`video_frames`, `detection_results`) upon connection.

**Message Queues**:

- `video_frames`: Used by the Frame Reader Service to send raw video frames to the Detection Service.
- `detection_results`: Used by the Detection Service to send processed frames, detections, and violation data to the Streaming Service.

### 4.4 Configuration (`config.py`)

**Purpose**: Centralizes application configuration using Python dataclasses and environment variables. This allows for easy management of settings across different deployment environments (development, testing, production).

**Key Features**:

- **Dataclasses**: Defines structured configuration objects for RabbitMQ, Database, Streaming, Detection, Frame Reader, and Logging.
- **Environment Variable Loading**: Automatically loads configuration values from environment variables using `os.getenv()`.
- **`Config.from_env()`**: A class method to create a `Config` instance populated from environment variables.
- **`validate()` Method**: Provides basic validation for configuration parameters (e.g., port ranges, confidence thresholds).

**Benefits**: Enables flexible deployment without modifying code, promotes consistency, and makes it easy to understand available configuration options.

### 4.5 Logging (`logging_config.py`)

**Purpose**: Provides a standardized and centralized logging setup for all services. It configures Python's `logging` module to ensure consistent log formatting, output destinations (console and file), and log rotation.

**Key Features**:

- **`setup_logging()`**: Initializes the root logger with specified level, format, and optional file output with rotation.
- **`get_logger()`**: Returns a logger instance for a specific module, allowing for granular logging control.
- **`log_performance` Decorator**: A decorator to measure and log the execution time of functions, useful for performance monitoring.
- **`LogContext` Context Manager**: Provides a convenient way to log the start and end of operations, including their duration and any exceptions.
- **Third-Party Library Control**: Sets logging levels for verbose libraries like `pika` and `uvicorn` to reduce noise.

**Benefits**: Facilitates debugging, monitoring, and auditing of the system's behavior in various environments.

## 5. Testing Strategy

To ensure the reliability, correctness, and performance of the Pizza Violation Detection System, a multi-faceted testing strategy has been adopted, encompassing both unit and integration tests.

### 5.1 Unit Tests

**Location**: `tests/test_shared.py`

**Purpose**: Unit tests focus on individual components or functions in isolation, verifying that each part of the codebase works as expected. For this project, unit tests primarily cover the shared utility modules.

**Key Areas Tested**:

- **Database Module (`database.py`)**:
  - Database initialization and table creation.
  - Insertion and retrieval of `ViolationRecord` and `ROI` objects.
  - Correctness of query methods (e.g., `get_violations`, `get_violation_summary`).
  - `upsert_roi` functionality.
- **Data Models (`models.py`)**:
  - Correct instantiation and validation of Pydantic models (e.g., `Detection`, `ROI`).
  - Ensuring data types and constraints are enforced.
- **RabbitMQ Client (`rabbitmq_client.py`)**:
  - Mocking RabbitMQ connections to test `connect`, `publish_message`, and `consume_messages` logic without a live broker.
  - Verification of message formatting and publishing behavior.

**Framework**: `unittest` (Python's built-in testing framework)

**How to Run**:

```bash
python -m unittest tests/test_shared.py
```

### 5.2 Integration Tests

**Location**: `tests/test_integration.py`

**Purpose**: Integration tests verify the interactions between different components and services, ensuring that they work together seamlessly as a complete system. These tests simulate real-world scenarios to validate end-to-end data flow and functionality.

**Key Areas Tested**:

- **End-to-End Data Flow**:
  - Simulating frame publishing to RabbitMQ.
  - Verifying that the Detection Service processes frames and publishes results.
  - Confirming that the Streaming Service receives and handles detection results.
- **Violation Detection Logic**:
  - Testing various scenarios to ensure violations are correctly identified and recorded.
  - Verifying that ROIs are correctly applied in the detection process.
- **Database Integration**:
  - Ensuring that violation records are correctly persisted and retrieved from the database.
  - Testing the consistency of data across services.
- **Performance Benchmarks**:
  - Includes basic performance tests for database operations (e.g., insertion and retrieval of a large number of records) to identify potential bottlenecks.

**Framework**: `unittest` (with `patch` from `unittest.mock` for external dependencies like RabbitMQ)

**How to Run**:

```bash
python -m unittest tests/test_integration.py
```

**Running All Tests**:

To run all unit and integration tests:

```bash
python -m unittest discover tests/
# Or using pytest (if installed and configured)
pytest tests/
```

## 6. Deployment

The Pizza Violation Detection System is designed for easy deployment using Docker and Docker Compose, ensuring a consistent and isolated environment across different machines.

### 6.1 Docker Compose

**File**: `docker-compose.yml`

**Purpose**: Docker Compose is used to define and run multi-container Docker applications. It orchestrates all the microservices (RabbitMQ, Frame Reader, Detection, Streaming, Streamlit) and their dependencies, networks, and volumes.

**Key Services Defined**:

- **`rabbitmq`**: The RabbitMQ message broker, exposed on ports `5672` (AMQP) and `15672` (management UI). Includes a health check to ensure it's ready before other services start.
- **`streaming_service`**: The FastAPI application. Built from `services/streaming/Dockerfile`, exposed on port `8000`. Depends on `rabbitmq`.
- **`detection_service`**: The object detection and violation logic service. Built from `services/detection/Dockerfile`. Depends on `rabbitmq`.
- **`frame_reader`**: The video ingestion service. Built from `services/frame_reader/Dockerfile`. Depends on `rabbitmq`. Its `command` can be overridden to specify the video source.
- **`streamlit_frontend`**: The Streamlit web dashboard. Built from `frontend/Dockerfile`, exposed on port `8501`. Depends on `streaming_service`.

**Volumes**:

- `rabbitmq_data`: Persistent volume for RabbitMQ data.
- `./data`: Mounts the host's `./data` directory to `/app/data` in containers for database persistence.
- `./violation_frames`: Mounts the host's `./violation_frames` to store captured violation frames.
- `./models`: Mounts the host's `./models` to `/app/models` for YOLO model files.
- `./videos`: Mounts the host's `./videos` to `/app/videos` for video input files for the `frame_reader`.

**Networks**: A `pizza_network` bridge network is defined to enable communication between services.

**Deployment Steps**:

1. **Build Images**:
   ```bash
   docker-compose build
   ```
2. **Start Services**:
   ```bash
   docker-compose up -d
   ```
3. **Stop Services**:
   ```bash
   docker-compose down
   ```
4. **View Logs**:
   ```bash
   docker-compose logs -f
   ```

### 6.2 Environment Variables

Configuration for each service is managed through environment variables, as defined in `shared/config.py` and used in the respective `Dockerfile`s and `docker-compose.yml`. A `.env.example` file is provided as a template.

**Common Environment Variables**:

- `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, `RABBITMQ_PASSWORD`: For RabbitMQ connection.
- `DATABASE_PATH`: Path to the SQLite database file.

**Service-Specific Environment Variables**:

- **Streaming Service**:
  - `HOST`, `PORT`: For FastAPI server binding.
  - `DEBUG`: Enables/disables FastAPI debug mode.
  - `CORS_ORIGINS`: CORS allowed origins.
- **Detection Service**:
  - `MODEL_PATH`: Path to the YOLO model file.
  - `CONFIDENCE_THRESHOLD`: Minimum confidence for detections.
  - `SCOOPER_DISTANCE_THRESHOLD`: Distance threshold for scooper detection.
- **Frame Reader Service**:
  - `VIDEO_SOURCE`: Specifies the video input (file, camera ID, RTSP URL).
  - `FPS_LIMIT`: Target frames per second.
- **Streamlit Frontend**:
  - `STREAMLIT_SERVER_PORT`, `STREAMLIT_SERVER_ADDRESS`, `STREAMLIT_SERVER_HEADLESS`: Streamlit server configuration.

**Usage**: Create a `.env` file in the root directory of the project (next to `docker-compose.yml`) and populate it with your desired values. Docker Compose will automatically load these variables.

## 7. API Reference

The FastAPI Streaming Service exposes a comprehensive set of RESTful API endpoints and a WebSocket endpoint for real-time data streaming. Detailed interactive documentation is available via Swagger UI at `http://localhost:8000/docs` when the `streaming_service` is running.

### REST API Endpoints

- **`GET /health`**

  - **Purpose**: Checks the health and status of the Streaming Service.
  - **Response**: `HealthCheck` model (e.g., `{"status": "healthy", "timestamp": "...", "version": "1.0.0"}`)
- **`GET /api/violations/summary`**

  - **Purpose**: Retrieves aggregated statistics and metadata about detected violations.
  - **Response**: `ViolationSummary` model (e.g., `{"total_violations": 5, "violations_by_type": {"no_scooper": 5}, "last_violation": "...", "active_rois": [...], "processing_status": "active"}`)
- **`GET /api/violations`**

  - **Purpose**: Fetches detailed records of violations with pagination and time-based filtering.
  - **Query Parameters**:
    - `limit` (int, default: 50): Maximum number of records to return.
    - `offset` (int, default: 0): Number of records to skip.
    - `start_time` (str, optional): ISO format datetime to filter violations from.
    - `end_time` (str, optional): ISO format datetime to filter violations until.
  - **Response**: JSON object containing a list of `ViolationRecord` objects and total count.
- **`GET /api/rois`**

  - **Purpose**: Retrieves all currently configured Regions of Interest.
  - **Response**: List of `ROI` models.
- **`POST /api/rois`**

  - **Purpose**: Creates a new ROI or updates an existing one.
  - **Request Body**: `ROI` model (JSON object).
  - **Response**: Confirmation message.
- **`GET /api/status`**

  - **Purpose**: Provides comprehensive system status and performance metrics.
  - **Response**: `SystemStatus` model (e.g., `{"services": {...}, "metrics": {...}, "uptime": "..."}`)

### WebSocket Endpoint

- **`WS /ws/video`**
  - **Purpose**: Establishes a real-time bidirectional communication channel for streaming annotated video frames and associated detection/violation data.
  - **Message Format (Sent to Client)**: `WebSocketMessage` model (e.g., `{"type": "frame", "frame_id": "...", "timestamp": "...", "image_data": "base64_encoded_image", "detections": [...], "violations": [...]}`)
  - **Usage**: Clients (like the Streamlit frontend) connect to this endpoint to receive a continuous stream of processed video frames.

## 8. Troubleshooting

This section provides guidance on common issues and how to resolve them.

### 8.1 General Debugging Steps

1. **Check Docker Container Status**: Ensure all services are running:
   ```bash
   docker-compose ps
   ```
2. **Review Logs**: Examine the logs of individual services for error messages:
   ```bash
   docker-compose logs -f <service_name>
   # e.g., docker-compose logs -f streaming_service
   ```
3. **Verify Network Connectivity**: Ensure containers can communicate with each other (e.g., `streaming_service` can reach `rabbitmq`).
4. **Check Environment Variables**: Confirm that all necessary environment variables are correctly set in your `.env` file or `docker-compose.yml`.

### 8.2 Common Issues and Solutions

- **RabbitMQ Connection Failed**:

  - **Symptom**: Services fail to start or continuously try to reconnect to RabbitMQ.
  - **Possible Causes**: RabbitMQ container not running, incorrect hostname/port, wrong credentials, firewall issues.
  - **Solution**:
    - Verify `rabbitmq` container is healthy: `docker-compose ps` and `docker-compose logs rabbitmq`.
    - Check `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, `RABBITMQ_PASSWORD` in your `.env` file.
    - Access RabbitMQ Management UI at `http://localhost:15672` (default user: `guest`, pass: `guest` or `admin`/`admin123` if changed in `docker-compose.yml`).
- **Video Feed Not Appearing / Frame Reader Not Publishing**:

  - **Symptom**: No video frames or detections in the Streamlit dashboard.
  - **Possible Causes**: Incorrect `VIDEO_SOURCE` path, video file not accessible within the container, `frame_reader` service not running or crashing.
  - **Solution**:
    - Ensure your video file is in the `pizza_violation_detection/videos` directory on your host and the `docker-compose.yml` `command` for `frame_reader` points to the correct path inside the container (e.g., `/app/videos/your_video.mp4`).
    - Check `docker-compose logs -f frame_reader` for errors related to video capture.
    - If using a camera, ensure it's accessible by the Docker container (may require specific Docker configurations for device access).
- **No Detections / Violations**:

  - **Symptom**: Video feed is present, but no bounding boxes or violation alerts.
  - **Possible Causes**: Detection model not loaded, incorrect `MODEL_PATH`, low `CONFIDENCE_THRESHOLD`, `detection_service` crashing.
  - **Solution**:
    - Check `docker-compose logs -f detection_service` for model loading errors or processing issues.
    - Ensure your YOLO model file is correctly placed in `pizza_violation_detection/models` and `MODEL_PATH` is set.
    - Adjust `CONFIDENCE_THRESHOLD` in `.env` (e.g., lower it to `0.5` for initial testing).
    - Verify ROIs are correctly configured via the Streamlit UI or by checking the `rois` table in the database.
- **Streamlit Frontend Not Accessible**:

  - **Symptom**: Cannot access `http://localhost:8501`.
  - **Possible Causes**: `streamlit_frontend` container not running, port conflict, incorrect `STREAMLIT_SERVER_PORT`.
  - **Solution**:
    - Check `docker-compose ps` to ensure `streamlit_frontend` is running.
    - Verify port mapping in `docker-compose.yml` (`8501:8501`).
    - Check `docker-compose logs -f streamlit_frontend` for startup errors.
- **High CPU/Memory Usage**:

  - **Symptom**: System becomes unresponsive, containers restart frequently.
  - **Possible Causes**: High resolution video, high `FPS_LIMIT`, complex detection model.
  - **Solution**:
    - Reduce `FPS_LIMIT` in `frame_reader` service in `docker-compose.yml`.
    - Use lower resolution video files.
    - Consider optimizing the YOLO model or using a smaller one.

## 9. Future Enhancements

This system provides a solid foundation for real-time computer vision applications. Several areas can be enhanced to improve its functionality, robustness, and user experience:

- **Advanced Violation Logic**:
  - Implement more sophisticated rules for violation detection (e.g., tracking objects over time, gesture recognition).
  - Support for multiple types of violations beyond just

scooper usage.

- **Scalability Improvements**:
  - Implement a distributed task queue for the Detection Service to process frames in parallel across multiple machines or GPUs.
  - Explore Kubernetes for more advanced container orchestration and auto-scaling.
- **Model Management**:
  - Integrate with a model serving framework (e.g., MLflow, BentoML) for easier model deployment and versioning.
  - Enable dynamic model updates without service restarts.
- **User Management and Authentication**:
  - Add user authentication and authorization to the FastAPI and Streamlit interfaces.
  - Implement role-based access control for different user types.
- **Enhanced Frontend Features**:
  - Allow users to draw and define ROIs directly on the live video feed in the Streamlit UI.
  - Implement historical data visualization (e.g., trends of violations over time).
  - Add real-time alerts and notifications (e.g., email, SMS integration).
- **Database Scalability**:
  - Migrate from SQLite to a more robust database like PostgreSQL or MongoDB for production deployments with high data volumes.
  - Implement database sharding or replication for improved performance and availability.
- **Video Storage and Retrieval**:
  - Implement a dedicated service for storing and retrieving raw video footage, possibly integrating with cloud storage solutions.
  - Develop a video playback feature in the frontend to review violation events with context.
- **Performance Optimization**:
  - Explore using GPU acceleration for video decoding and encoding in the Frame Reader Service.
  - Optimize the YOLO inference pipeline for faster processing.
- **Alerting and Reporting**:
  - Integrate with external alerting systems (e.g., PagerDuty, Slack) for critical violations.
  - Generate daily/weekly reports on compliance and violation trends.
- **Edge Deployment**:
  - Consider deploying parts of the system (Frame Reader, Detection) on edge devices for reduced latency and bandwidth usage.

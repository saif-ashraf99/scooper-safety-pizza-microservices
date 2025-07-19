# Implementation Details: Pizza Store Scooper Violation Detection System

This document provides a detailed overview of the implementation of each microservice and shared components within the Pizza Store Scooper Violation Detection System.

## 1. Frame Reader Service

**Location:** `/scooper-safety-pizza-microservices/services/frame_reader/`

**Purpose:** Responsible for ingesting video frames from various sources (video files, camera feeds, RTSP streams) and publishing them to the RabbitMQ message broker.

**Key Files:**

* `main.py`: Entry point for the service. Parses command-line arguments for video source, FPS, and RabbitMQ connection details. It then initializes and calls `FrameReaderService` based on the source type.
* `frame_reader_service.py`: Orchestrates the different frame reading mechanisms. It instantiates `VideoReader`, `CameraReader`, or `RTSPReader` based on the input source and manages their lifecycle.
* `base_reader.py`: (Assumed base class for readers) Likely defines the common interface and RabbitMQ publishing logic for all frame readers.
* `video_reader.py`: Implements logic for reading frames from local video files.
* `camera_reader.py`: Implements logic for reading frames from a connected camera (e.g., webcam).
* `rtsp_reader.py`: Implements logic for reading frames from RTSP (Real-Time Streaming Protocol) streams.
* `frame_encoder.py`: (Assumed) Handles encoding of frames before publishing to RabbitMQ.
* `frame_factory.py`: (Assumed) Factory for creating frame objects or processing them.

**Implementation Flow:**

1. `main.py` receives the video source (file path, camera ID, or RTSP URL) and other configurations.
2. It determines the type of source and instantiates the appropriate reader (`VideoReader`, `CameraReader`, or `RTSPReader`) via `FrameReaderService`.
3. The chosen reader continuously captures frames from its source.
4. Each captured frame is likely encoded (e.g., to base64) and packaged with metadata (timestamp, frame ID) into a message.
5. These messages are then published to a designated queue in RabbitMQ using the `RabbitMQPublisher` (from `shared` components).

## 2. Detection Service

**Location:** `/scooper-safety-pizza-microservices/services/detection/`

**Purpose:** Subscribes to frames from the message broker, performs object detection, applies violation detection logic, and publishes detection results and violation information.

**Key Files:**

* `main.py`: Entry point for the service. Parses command-line arguments for camera ID, operating mode (`service` or `video`), model path, database path, and RabbitMQ details. It initializes `DetectionService` and starts it in the specified mode.
* `detection.py`: Contains the core `DetectionService` class. It initializes `RabbitMQConsumer`, `RabbitMQPublisher`, `Database`, `YOLODetector`, and `ViolationDetector`. It has methods for processing video files directly (`process_video`) or processing frames received via RabbitMQ (`process_frame`). It decodes/encodes frames (base64) and publishes results.
* `yolo_detector.py`: Encapsulates the YOLO object detection logic using the `ultralytics` library. It loads the pre-trained YOLO model (`fine-tuned-yolo12m-v2.pt`), handles device selection (CUDA/CPU), and provides a `detect` method to perform inference on a given frame, returning bounding boxes, confidence scores, and class names.
* `violation_detector.py`: Implements the core logic for detecting violations. It uses `deep_sort_realtime` for tracking objects (hands/persons) and determines if a scooper is used when a hand/person is within a defined ROI. It manages a state machine per ROI to emit `NO_SCOOPER` violations only once per continuous episode, with a cooldown period.
* `video_processor.py`: (Assumed) Likely handles the overall video processing pipeline, integrating detection and violation logic, and saving results.
* `container_finder.py`: (Assumed) Likely responsible for defining or loading the Regions of Interest (ROIs), potentially from `container_rois.json`.
* `rois/container_rois.json`: JSON file defining the coordinates and properties of the Regions of Interest (ROIs) within the pizza store environment.

**Implementation Flow:**

1. In `service` mode, `DetectionService` connects to RabbitMQ as a consumer to receive frame messages from the `frame_reader` service.
2. For each received frame:
   * The base64 encoded frame data is decoded into an OpenCV image (`numpy` array).
   * `YOLODetector` performs object detection, identifying hands, persons, scoopers, and pizzas.
   * `ViolationDetector` analyzes the detected objects in relation to predefined ROIs. It tracks hands/persons and checks for the presence of a scooper when an interaction occurs within an ROI.
   * If a violation is detected, the details are recorded in the SQLite database (`violations.db`) via the `Database` utility.
   * The frame is annotated with bounding boxes, labels, and violation indicators.
   * The annotated frame and detection/violation results are packaged into a `DetectionResult` object.
   * This `DetectionResult` is then published to another RabbitMQ queue, to be consumed by the `streaming_service`.
3. In `video` mode, `DetectionService` processes a local video file, performing the same detection and violation logic, and can save an output video with annotations.

## 3. Streaming Service

**Location:** `/scooper-safety-pizza-microservices/services/streaming/`

**Purpose:** Provides a real-time API and WebSocket endpoint for the frontend to consume detection results and video streams.

**Key Files:**

* `main.py`: Entry point for the FastAPI application. Configures logging, CORS middleware, includes API routes, and sets up startup/shutdown events. It also serves static files for the frontend.
* `api_routes.py`: Defines the REST API endpoints and the WebSocket endpoint.
  * `/health`: Basic health check.
  * `/api/violations/summary`: Provides aggregated violation statistics.
  * `/api/violations`: Returns detailed violation records with pagination and filtering.
  * `/api/rois`: Allows fetching and updating ROI configurations.
  * `/api/status`: Provides system status and metrics.
  * `/ws/video`: WebSocket endpoint for real-time annotated video stream and detection results.
* `websocket_manager.py`: Manages active WebSocket connections. It handles connecting, disconnecting, and broadcasting messages (annotated frames, detection results) to all connected clients.
* `config.py`: (Assumed) Contains configuration settings for the FastAPI application, such as host, port, and debug mode.
* `dependencies.py`: (Assumed) Contains FastAPI dependency injection functions, e.g., `get_database` to provide a database connection.
* `metrics.py`: (Assumed) Manages system-wide metrics like uptime and processing statistics.
* `startup_shutdown.py`: (Assumed) Contains functions to be executed on application startup (e.g., connecting to RabbitMQ) and shutdown (e.g., disconnecting).
* `frontend.py`: (Assumed) Serves the HTML content for the Streamlit frontend.

**Implementation Flow:**

1. The FastAPI application starts, exposing REST and WebSocket endpoints.
2. On startup, it connects to RabbitMQ as a consumer to receive `DetectionResult` messages published by the `detection_service`.
3. When a `DetectionResult` message is received:
   * The `streaming_service` processes the data.
   * It updates its internal state and metrics.
   * It broadcasts the annotated frame data, detections, and violations to all connected WebSocket clients via `websocket_manager.py`.
4. Frontend clients connect to the `/ws/video` WebSocket endpoint to receive real-time updates.
5. Frontend clients can also query the REST API endpoints for historical data, ROI configurations, and system status.

## 4. Shared Components

**Location:** `/scooper-safety-pizza-microservices/services/shared/`

**Purpose:** Provides common utilities, data models, and helper functions used across multiple microservices to ensure consistency and reduce code duplication.

**Key Files:**

* `database.py`: Manages interactions with the SQLite database (`violations.db`). It defines methods for:
  * Initializing database tables (`violations`, `rois`, `video_frames`, `detections`, `detection_results`, `health_checks`, `system_status`, `websocket_messages`).
  * Providing a context manager for database connections.
  * Inserting and retrieving `ViolationRecord`s, `ROI`s, `VideoFrame`s, `Detection`s, `DetectionResult`s, `HealthCheck`s, `SystemStatus`es, and `WebSocketMessage`s.
  * Querying violation summaries and counts.
* `models.py`: Defines Pydantic data models for various entities used throughout the system. This ensures data validation and clear structure for inter-service communication and database storage. Key models include:
  * `ViolationType` (Enum): Defines types of violations (e.g., `NO_SCOOPER`).
  * `DetectionClass` (Enum): Defines classes of objects detected (e.g., `HAND`, `PERSON`, `PIZZA`, `SCOOPER`).
  * `BoundingBox`: Represents the coordinates of a detected object.
  * `Detection`: Represents a single object detection with class, confidence, and bounding box.
  * `Violation`: Represents a detected violation event.
  * `FrameMetadata`: Metadata associated with a video frame.
  * `VideoFrame`: Encapsulates a video frame with its data and metadata.
  * `DetectionResult`: Aggregates detection and violation results for a single frame.
  * `ROI`: Defines a Region of Interest with its ID, name, coordinates, and associated violation type.
  * `ViolationRecord`: Represents a stored violation record in the database.
  * `ViolationSummary`: Summary statistics for violations.
  * `SystemStatus`: Overall system health and metrics.
  * `HealthCheck`: Basic health check response model.
  * `WebSocketMessage`: Structure for messages sent over WebSockets.
* `rabbitmq_client.py`: Provides `RabbitMQConsumer` and `RabbitMQPublisher` classes for interacting with the RabbitMQ message broker. These classes abstract away the complexities of connecting, declaring queues, publishing messages, and consuming messages.
* `config.py`: (Assumed) Contains shared configuration settings like RabbitMQ credentials, database paths, etc.
* `logging_config.py`: (Assumed) Centralized logging configuration for consistent log formatting across services.

**Implementation Details:**

* **Database:** Uses SQLite for simplicity and embedded usage. The `Database` class handles schema creation and all CRUD operations for the various data entities. Timestamps are stored as UNIX epoch integers for efficient querying.
* **Models:** Pydantic models are extensively used for data serialization, deserialization, and validation, ensuring type safety and clear data contracts between services.
* **RabbitMQ:** The `rabbitmq_client.py` provides a robust way to send and receive messages, enabling asynchronous communication and decoupling of services. It handles connection management and basic message operations.

## 5. Frontend UI (Streamlit)

**Location:** `/scooper-safety-pizza-microservices/frontend/`

**Purpose:** Provides a web-based user interface for visualizing the real-time video stream with detections and violations, and displaying violation statistics.

**Key Files:**

* `app.py`: The main Streamlit application script. It connects to the `streaming_service` to receive real-time video frames and detection results via WebSocket. It then uses Streamlit components to display the video, draw bounding boxes and ROIs, and show violation counts.
* `Dockerfile`: Defines the Docker image for the Streamlit application.
* `requirements.txt`: Specifies Python dependencies for the Streamlit frontend.

**Implementation Flow:**

1. The Streamlit `app.py` runs within its Docker container.
2. It establishes a WebSocket connection to the `streaming_service`'s `/ws/video` endpoint.
3. Upon receiving `WebSocketMessage`s, it extracts the annotated image data, detections, and violations.
4. Streamlit components are used to render the image and display relevant information (e.g., violation counts, ROI status).
5. The UI provides a real-time view of the pizza store, highlighting detected objects and any hygiene protocol violations.

## 6. Docker Compose Orchestration

**Location:** `/scooper-safety-pizza-microservices/docker-compose.yml`

**Purpose:** Defines and runs the multi-container Docker application. It orchestrates the startup, networking, and dependencies of all microservices and the RabbitMQ broker.

**Key Services Defined:**

* `rabbitmq`: The RabbitMQ message broker, exposed on ports 5672 (AMQP) and 15672 (management UI).
* `streaming_service`: The FastAPI application, built from the current context (`.`) using `services/streaming/Dockerfile`, exposed on port 8000.
* `detection_service`: The Python detection application, built from the current context (`.`) using `services/detection/Dockerfile`. It includes specific command-line arguments for its operation mode and model/database paths.
* `frame_reader`: The Python frame ingestion service, built from the current context (`.`) using `services/frame_reader/Dockerfile`. It specifies a video source and FPS.
* `streamlit_frontend`: The Streamlit UI, built from the `frontend/` directory using its `Dockerfile`, exposed on port 8501.

**Networking and Volumes:**

* All services are connected to a `pizza_network` bridge network, allowing them to communicate with each other using their service names (e.g., `rabbitmq`, `streaming_service`).
* Volumes are used to persist RabbitMQ data (`rabbitmq_data`) and to mount local directories (`./data`, `./violation_frames`, `./videos`, `./models`) into the containers, enabling data sharing and persistence.
* `container_rois.json` is mounted as a read-only volume into the `detection_service`.

This comprehensive overview covers the implementation details of each component, their interactions, and how they are orchestrated using Docker Compose.

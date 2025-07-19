# Pizza Store Scooper Violation Detection System

## Overview

This project implements a Computer Vision system to monitor hygiene protocol compliance in a pizza store. It detects whether workers are using a scooper when handling ingredients from designated areas (Regions of Interest - ROIs). Any instance of picking up ingredients without a scooper from these ROIs is flagged as a violation.

## Features

*   **Real-time Video Ingestion:** Reads video frames from various sources (video files or RTSP camera feeds).
*   **Object Detection:** Utilizes a pre-trained YOLO model to detect hands, persons, pizzas, and scoopers.
*   **Violation Detection Logic:** Identifies and flags instances where ingredients are handled without a scooper within defined ROIs.
*   **Microservices Architecture:** Built with a modular design using Docker and Docker Compose for scalability and maintainability.
*   **Message Queuing:** Employs RabbitMQ for efficient inter-service communication.
*   **Real-time Streaming:** Provides live video feeds with detection overlays and violation alerts to a frontend.
*   **User-friendly Frontend:** A Streamlit-based interface for visualizing detections, ROIs, and violations.

## System Architecture

The system is composed of several microservices:

1.  **Frame Reader Service:** Ingests video frames and publishes them to a message broker.
2.  **RabbitMQ:** Acts as the central message broker for inter-service communication.
3.  **Detection Service:** Consumes frames, performs object detection, applies violation logic, and stores violation data.
4.  **Streaming Service:** Provides real-time video streams and API endpoints for frontend consumption.
5.  **Frontend UI (Streamlit):** Displays the video feed with detections, ROIs, and violation alerts.

## Setup and Running the Project

### Prerequisites

*   [Docker](https://www.docker.com/get-started) installed.
*   [Docker Compose](https://docs.docker.com/compose/install/) installed.

### Quick Start with Docker Compose

1.  **Navigate to the project root:**
    Open your terminal or command prompt and navigate to the directory where `docker-compose.yml` is located.

    ```bash
    cd /path/to/your/project/project
    ```

2.  **Build and run the services:**
    Execute the following command to build the Docker images and start all the services defined in `docker-compose.yml`:

    ```bash
    docker-compose up --build
    ```

    This command will:
    *   Pull the `rabbitmq` image and start the RabbitMQ server.
    *   Build the Docker images for the `streaming_service`, `detection_service`, `frame_reader`, and `streamlit_frontend`.
    *   Start all these services, linking them via the `pizza_network`.

3.  **Access the Frontend:**
    Once all services are up and running, you can access the Streamlit frontend in your web browser:

    *   **Streamlit Frontend:** `http://localhost:8501`

4.  **Access Streaming Service API (for developers):**
    The FastAPI streaming service will be available at:

    *   **Streaming Service:** `http://localhost:8000`
    *   **API Documentation (Swagger UI):** `http://localhost:8000/docs`

### Stopping the Services

To stop and remove the Docker containers, networks, and volumes created by `docker-compose up`:

```bash
docker-compose down
```

## Project Structure

```
project/
├── data/                       # Stores database files (e.g., violations.db)
├── docs/                       # Project documentation (e.g., EagleVisionTask.pdf)
├── docker-compose.yml          # Defines the multi-service Docker application
├── frontend/                   # Streamlit frontend application
├── models/                     # Pre-trained machine learning models
├── openCV-output-videos/       # Output directory for processed videos
├── requirements.txt            # Main Python dependencies
├── services/                   # Contains individual microservices
│   ├── detection/              # Object detection and violation logic
│   ├── frame_reader/           # Video frame ingestion
│   ├── shared/                 # Common utilities and models
│   └── streaming/              # Real-time streaming and API
├── videos/                     # Input directory for video files
└── violation_frames/           # Stores captured violation frames
```

## Technologies Used

*   **Containerization:** Docker, Docker Compose
*   **Message Broker:** RabbitMQ
*   **Video Processing:** OpenCV
*   **Object Detection:** YOLO (Ultralytics)
*   **Backend Framework:** FastAPI
*   **Frontend Framework:** Streamlit
*   **Database:** SQLite
*   **Programming Language:** Python

# Dockerfile for Frame Reader Service

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY services/frame_reader/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules and service code
COPY services/shared /app/shared
COPY services/frame_reader /app

# Create necessary directories
RUN mkdir -p /app/videos

# Run the application
CMD ["python", "main.py", "--source", "0", "--fps", "10"]

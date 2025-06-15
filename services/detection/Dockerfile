FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version) and Ultralytics YOLO
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install ultralytics

# Copy requirements and install dependencies
COPY services/detection/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules and service code
COPY services/shared /app/shared
COPY services/detection /app

# Create necessary directories
RUN mkdir -p /app/data /app/violation_frames /app/models

# Run the application
CMD ["python", "main.py"]

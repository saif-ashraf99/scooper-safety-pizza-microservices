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

# Copy requirements first to leverage Docker cache
COPY services/detection/requirements.txt .

# Install PyTorch (CPU version) first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the latest Ultralytics version to ensure C3k2 support
RUN pip install --no-cache-dir ultralytics
# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules and service code
COPY services/shared /app/shared
COPY services/detection /app

# Create necessary directories
RUN mkdir -p /app/data /app/violation_frames /app/models

# Run the application
CMD ["python", "main.py"]
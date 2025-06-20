FROM python:3.11-slim

WORKDIR /app

# 1) System deps for OpenCV/YOLO
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 2) Copy requirements & install
COPY services/detection/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install -U ultralytics

# 3) Copy your application code into a /app/services folder
COPY services /app/services

# 4) Copy your entrypoint script
COPY services/detection/main.py /app/main.py

# Copy the template‐images folder into the same tree
COPY services/detection/imgs /app/services/detection/imgs

# 5) Create data dirs
RUN mkdir -p /app/data /app/violation_frames /app/models

# 6) Set PYTHONPATH so Python picks up /app/services
ENV PYTHONPATH=/app

# 7) Run
CMD ["python", "main.py"]

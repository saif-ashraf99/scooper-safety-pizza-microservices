import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import base64
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Import shared modules
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import (
    Database, RabbitMQConsumer, ViolationRecord, ROI, ViolationSummary,
    SystemStatus, HealthCheck, WebSocketMessage, ViolationType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
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

# Initialize FastAPI app
app = FastAPI(
    title="Pizza Violation Detection API",
    description="Real-time pizza store scooper violation detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db = Database(config.DATABASE_PATH)
rabbitmq_consumer = None
websocket_connections: List[WebSocket] = []
system_metrics = {
    'frames_processed': 0,
    'violations_detected': 0,
    'processing_fps': 0.0,
    'queue_size': 0,
    'start_time': time.time()
}

# Dependency for database
def get_database():
    return db

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# API Endpoints

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/api/violations/summary", response_model=ViolationSummary)
async def get_violation_summary(db: Database = Depends(get_database)):
    """Get violation statistics and metadata"""
    try:
        summary_data = db.get_violation_summary()
        rois = db.get_rois()
        active_rois = [roi.id for roi in rois if roi.active]
        
        return ViolationSummary(
            total_violations=summary_data['total_violations'],
            violations_by_type=summary_data['violations_by_type'],
            last_violation=summary_data['last_violation'],
            active_rois=active_rois,
            processing_status="active" if rabbitmq_consumer else "inactive"
        )
    except Exception as e:
        logger.error(f"Error getting violation summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/violations")
async def get_violations(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    db: Database = Depends(get_database)
):
    """Get detailed violation records with pagination"""
    try:
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        
        violations = db.get_violations(limit, offset, start_dt, end_dt)
        total = db.get_violation_count(start_dt, end_dt)
        
        return {
            "violations": [violation.dict() for violation in violations],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")
    except Exception as e:
        logger.error(f"Error getting violations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/rois", response_model=List[ROI])
async def get_rois(db: Database = Depends(get_database)):
    """Get current ROI configurations"""
    try:
        return db.get_rois()
    except Exception as e:
        logger.error(f"Error getting ROIs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/rois")
async def create_or_update_roi(roi: ROI, db: Database = Depends(get_database)):
    """Create or update ROI configuration"""
    try:
        db.upsert_roi(roi)
        return {"message": f"ROI {roi.id} updated successfully"}
    except Exception as e:
        logger.error(f"Error updating ROI: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and metrics"""
    try:
        uptime_seconds = time.time() - system_metrics['start_time']
        uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
        
        # Check RabbitMQ connection
        rabbitmq_status = "connected" if rabbitmq_consumer else "disconnected"
        
        return SystemStatus(
            services={
                "frame_reader": "unknown",  # Would need to check actual service
                "detection": "unknown",     # Would need to check actual service
                "rabbitmq": rabbitmq_status,
                "streaming": "active"
            },
            metrics=system_metrics,
            uptime=uptime_str
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video stream"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# RabbitMQ message handler
def handle_detection_result(message: dict, loop: asyncio.AbstractEventLoop):
    try:
        logger.info("📨 Received detection message from RabbitMQ")
        frame_data = message.get('frame_data')
        logger.info(f"Frame data present: {bool(frame_data)}")

        system_metrics['frames_processed'] += 1
        if message.get('violations'):
            system_metrics['violations_detected'] += len(message['violations'])

        websocket_message = {
            "type": "frame",
            "frame_id": message.get('frame_id'),
            "timestamp": message.get('timestamp'),
            "image_data": frame_data,
            "detections": message.get('detections', []),
            "violations": message.get('violations', [])
        }

        asyncio.run_coroutine_threadsafe(manager.broadcast(websocket_message), loop)

    except Exception as e:
        logger.error(f"Error handling detection result: {e}")



# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    global rabbitmq_consumer, fastapi_event_loop
    try:
        fastapi_event_loop = asyncio.get_running_loop()  # <-- capture main loop here

        rabbitmq_consumer = RabbitMQConsumer(
            host=config.RABBITMQ_HOST,
            port=config.RABBITMQ_PORT,
            username=config.RABBITMQ_USER,
            password=config.RABBITMQ_PASSWORD
        )

        import threading
        thread = threading.Thread(
            target=rabbitmq_consumer.consume_detection_results,
            args=(lambda msg: handle_detection_result(msg, fastapi_event_loop),),
            daemon=True
        )
        thread.start()

        logger.info("Streaming service started and consumer thread launched.")
    except Exception as e:
        logger.error(f"Failed to start streaming service: {e}")



@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rabbitmq_consumer
    if rabbitmq_consumer:
        rabbitmq_consumer.disconnect()
    logger.info("Streaming service shut down")

# Static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pizza Violation Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .video-container { text-align: center; margin: 20px 0; }
            .stats { display: flex; justify-content: space-around; margin: 20px 0; }
            .stat-box { padding: 20px; border: 1px solid #ccc; border-radius: 5px; }
            #video { max-width: 100%; height: auto; border: 2px solid #333; }
            .violation-alert { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pizza Store Violation Detection System</h1>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>Total Violations</h3>
                    <div id="total-violations">0</div>
                </div>
                <div class="stat-box">
                    <h3>Processing Status</h3>
                    <div id="processing-status">Inactive</div>
                </div>
                <div class="stat-box">
                    <h3>Frames Processed</h3>
                    <div id="frames-processed">0</div>
                </div>
            </div>
            
            <div class="video-container">
                <h2>Live Video Feed</h2>
                <img id="video" src="" alt="Video feed will appear here" />
                <div id="violation-alert" class="violation-alert" style="display: none;">
                    VIOLATION DETECTED!
                </div>
            </div>
        </div>

        <script>
            const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const ws = new WebSocket(`${wsProtocol}://${window.location.host}/ws/video`);

            const video = document.getElementById('video');
            const violationAlert = document.getElementById('violation-alert');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'frame') {
                    video.src = 'data:image/jpeg;base64,' + data.image_data;
                    
                    if (data.violations && data.violations.length > 0) {
                        violationAlert.style.display = 'block';
                        setTimeout(() => {
                            violationAlert.style.display = 'none';
                        }, 3000);
                    }
                }
            };
            
            // Update stats periodically
            setInterval(async () => {
                try {
                    const response = await fetch('/api/violations/summary');
                    const data = await response.json();
                    document.getElementById('total-violations').textContent = data.total_violations;
                    document.getElementById('processing-status').textContent = data.processing_status;
                    
                    const statusResponse = await fetch('/api/status');
                    const statusData = await statusResponse.json();
                    document.getElementById('frames-processed').textContent = statusData.metrics.frames_processed;
                } catch (error) {
                    console.error('Error updating stats:', error);
                }
            }, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )


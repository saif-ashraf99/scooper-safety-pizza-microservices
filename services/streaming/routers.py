import asyncio
from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from typing import List, Optional
from datetime import datetime
import time

from shared import HealthCheck, ViolationSummary, ROI, SystemStatus
from shared.database import Database
from dependencies import get_database
from consumer import system_metrics
from events import rabbitmq_consumer
from connection_manager import manager

router = APIRouter()

# Health check endpoint
@router.get("/health", response_model=HealthCheck)
async def health_check():
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

# Violation summary
@router.get("/api/violations/summary", response_model=ViolationSummary)
async def get_violation_summary(db = Depends(get_database)):
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
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

# Detailed violations listing
@router.get("/api/violations")
async def get_violations(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    db = Depends(get_database)
):
    try:
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        violations = db.get_violations(limit, offset, start_dt, end_dt)
        total = db.get_violation_count(start_dt, end_dt)
        return {
            "violations": [v.dict() for v in violations],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

# ROI endpoints
@router.get("/api/rois", response_model=List[ROI])
async def get_rois(db = Depends(get_database)):
    try:
        return db.get_rois()
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/api/rois")
async def create_or_update_roi(roi: ROI, db = Depends(get_database)):
    try:
        db.upsert_roi(roi)
        return {"message": f"ROI {roi.id} updated successfully"}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

# System status endpoint
@router.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    try:
        uptime_seconds = time.time() - system_metrics['start_time']
        uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
        rabbitmq_status = "connected" if rabbitmq_consumer else "disconnected"
        return SystemStatus(
            services={
                "frame_reader": "unknown",
                "detection": "unknown",
                "rabbitmq": rabbitmq_status,
                "streaming": "active"
            },
            metrics=system_metrics,
            uptime=uptime_str
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

# WebSocket endpoint for video stream
@router.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Frontend HTML endpoint
@router.get("/", response_class=HTMLResponse)
async def get_frontend():
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
    """
    return HTMLResponse(content=html_content)
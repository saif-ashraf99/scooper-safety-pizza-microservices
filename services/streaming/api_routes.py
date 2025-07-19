import asyncio
import logging
import time
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.responses import HTMLResponse
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import (
    Database, ViolationRecord, ROI, ViolationSummary,
    SystemStatus, HealthCheck, WebSocketMessage, ViolationType
)
from dependencies import get_database
from websocket_manager import manager
from metrics import system_metrics
from startup_shutdown import get_rabbitmq_consumer
from frontend import get_frontend_html

logger = logging.getLogger(__name__)

# Create API router
api_router = APIRouter()

@api_router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@api_router.get("/api/violations/summary", response_model=ViolationSummary)
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
            processing_status="active" if get_rabbitmq_consumer() else "inactive"
        )
    except Exception as e:
        logger.error(f"Error getting violation summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/api/violations")
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

@api_router.get("/api/rois", response_model=List[ROI])
async def get_rois(db: Database = Depends(get_database)):
    """Get current ROI configurations"""
    try:
        return db.get_rois()
    except Exception as e:
        logger.error(f"Error getting ROIs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/api/rois")
async def create_or_update_roi(roi: ROI, db: Database = Depends(get_database)):
    """Create or update ROI configuration"""
    try:
        db.upsert_roi(roi)
        return {"message": f"ROI {roi.id} updated successfully"}
    except Exception as e:
        logger.error(f"Error updating ROI: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and metrics"""
    try:
        uptime_seconds = time.time() - system_metrics['start_time']
        uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
        
        # Check RabbitMQ connection
        rabbitmq_status = "connected" if get_rabbitmq_consumer() else "disconnected"
        
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

@api_router.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video stream"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@api_router.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend HTML"""
    return get_frontend_html()
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum


class ViolationType(str, Enum):
    NO_SCOOPER = "no_scooper"


class DetectionClass(str, Enum):
    HAND = "hand"
    PERSON = "person"
    PIZZA = "pizza"
    SCOOPER = "scooper"


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")


class Detection(BaseModel):
    class_name: DetectionClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    track_id: Optional[int] = None         # DeepSORT id (all classes)
    hand_id: Optional[int] = None          # Filled only for HAND tracks


class Violation(BaseModel):
    type: ViolationType
    roi_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    description: Optional[str] = None
    track_id: Optional[int] = None 
    hand_id: Optional[int] = None
    

class FrameMetadata(BaseModel):
    width: int
    height: int
    fps: float
    source: str


class VideoFrame(BaseModel):
    frame_id: str
    timestamp: datetime
    frame_data: str  # base64 encoded
    metadata: FrameMetadata

    @field_validator("timestamp", mode="before")
    def set_timestamp_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class DetectionResult(BaseModel):
    frame_id: str
    timestamp: datetime
    detections: List[Detection]
    violations: List[Violation]
    frame_data: str  # base64 encoded with detections drawn
    processing_time: float

    @field_validator("timestamp", mode="before")
    def set_timestamp_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ROI(BaseModel):
    id: str
    name: str
    coordinates: List[float] = Field(..., min_items=4, max_items=4)
    active: bool = True
    violation_type: ViolationType
    

class ViolationRecord(BaseModel):
    id: Optional[int] = None
    frame_id: str
    # camera_id: str
    timestamp: datetime
    violation_type: ViolationType
    roi_id: str
    confidence: float
    frame_path: Optional[str] = None
    bounding_boxes: List[Detection]
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    @field_validator("timestamp", "created_at", mode="before")
    def set_timestamp_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ViolationSummary(BaseModel):
    total_violations: int
    violations_by_type: Dict[str, int]
    last_violation: Optional[datetime] = None
    active_rois: List[str]
    processing_status: str

    @field_validator("last_violation", mode="before")
    def set_timestamp_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class SystemStatus(BaseModel):
    services: Dict[str, str]
    metrics: Dict[str, Any]
    uptime: str


class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"

    @field_validator("timestamp", mode="before")
    def set_timestamp_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class WebSocketMessage(BaseModel):
    type: str
    frame_id: str
    timestamp: datetime
    image_data: str
    detections: List[Detection]
    violations: List[Violation]

    @field_validator("timestamp", mode="before")
    def set_timestamp_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v



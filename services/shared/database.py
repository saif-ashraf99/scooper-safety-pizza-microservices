import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import logging
from .models import (ViolationRecord, ROI, ViolationType, VideoFrame, 
                     Detection, DetectionResult, HealthCheck, SystemStatus, WebSocketMessage)

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = "violations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables and indexes"""
        with self.get_connection() as conn:
            # Create violations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id VARCHAR(36) NOT NULL,
                    camera_id VARCHAR(50) NOT NULL,
                    timestamp INTEGER NOT NULL, -- Storing as UNIX epoch integer
                    violation_type VARCHAR(50) NOT NULL,
                    roi_id VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    frame_path VARCHAR(255),
                    bounding_boxes TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create ROIs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rois (
                    id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    coordinates TEXT NOT NULL,
                    active BOOLEAN DEFAULT TRUE,
                    violation_type VARCHAR(50) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_frames (
                    frame_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    frame_data TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    source TEXT NOT NULL
                )
            """)

            # Create detections table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT NOT NULL,
                    track_id INTEGER,
                    hand_id INTEGER,
                    FOREIGN KEY(frame_id) REFERENCES video_frames(frame_id)
                )
            """)

            # Create detection_results table (aggregate)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_results (
                    frame_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    detections TEXT NOT NULL,
                    violations TEXT NOT NULL,
                    frame_data TEXT NOT NULL,
                    processing_time REAL NOT NULL
                )
            """)

            # Create health_checks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    timestamp INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    version TEXT NOT NULL
                )
            """)

            # Create system_status table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_status (
                    timestamp INTEGER PRIMARY KEY,
                    services TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    uptime TEXT NOT NULL
                )
            """)

            # Create websocket_messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS websocket_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    image_data TEXT NOT NULL,
                    detections TEXT NOT NULL,
                    violations TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_roi_id ON violations (roi_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_frames_ts ON video_frames(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ws_msgs_ts ON websocket_messages(timestamp)")


    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def insert_violation(self, violation: ViolationRecord) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO violations 
                    (camera_id, frame_id, timestamp, violation_type, roi_id, confidence,
                    frame_path, bounding_boxes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    violation.camera_id,
                    violation.frame_id,
                    int(violation.timestamp.replace(tzinfo=timezone.utc).timestamp()),
                    violation.violation_type.value,
                    violation.roi_id,
                    violation.confidence,
                    violation.frame_path,
                    json.dumps([det.dict() for det in violation.bounding_boxes]),
                    json.dumps(violation.metadata) if violation.metadata else None
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_violations(
        self,
        limit: int = 50,
        offset: int = 0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ViolationRecord]:
        """Get violation records with pagination and filtering"""
        query = "SELECT * FROM violations WHERE 1=1"
        params: List[Any] = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.replace(tzinfo=timezone.utc).timestamp()))

        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.replace(tzinfo=timezone.utc).timestamp()))
            
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            violations: List[ViolationRecord] = []
            for row in rows:
                # SQLite returns DATETIME columns as strings; parse created_at safely
                raw_ca = row["created_at"]
                if raw_ca is None:
                    created_at = None
                elif isinstance(raw_ca, (int, float)):
                    created_at = datetime.fromtimestamp(raw_ca, tz=timezone.utc)
                else:
                    # assume ISO-format "YYYY-MM-DD HH:MM:SS"
                    created_at = datetime.fromisoformat(raw_ca).replace(tzinfo=timezone.utc)

                violation = ViolationRecord(
                    id=row["id"],
                    frame_id=row["frame_id"],
                    timestamp=datetime.fromtimestamp(row["timestamp"], tz=timezone.utc),
                    violation_type=ViolationType(row["violation_type"]),
                    roi_id=row["roi_id"],
                    confidence=row["confidence"],
                    frame_path=row["frame_path"],
                    bounding_boxes=json.loads(row["bounding_boxes"]) if row["bounding_boxes"] else [],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                    created_at=created_at
                )
                violations.append(violation)

            return violations

    def get_violation_count_by_camera_frame(
        self,
        camera_id: str,
        frame_id: str
    ) -> int:
        """Return how many violations for a given camera and frame."""
        query = """
            SELECT COUNT(*) as count
            FROM violations
            WHERE camera_id = ? AND frame_id = ?
        """
        with self.get_connection() as conn:
            row = conn.execute(query, (camera_id, frame_id)).fetchone()
            return row["count"] if row else 0


    def get_violation_count(self, start_time: Optional[datetime] = None, 
                           end_time: Optional[datetime] = None) -> int:
        """Get total violation count"""
        query = "SELECT COUNT(*) as count FROM violations WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.replace(tzinfo=timezone.utc).timestamp()))
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.replace(tzinfo=timezone.utc).timestamp()))
        
        with self.get_connection() as conn:
            result = conn.execute(query, params).fetchone()
            return result["count"]
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get violation summary statistics"""
        with self.get_connection() as conn:
            # Total violations
            total = conn.execute("SELECT COUNT(*) as count FROM violations").fetchone()["count"]
            
            # Violations by type (ensure count is int)
            type_counts = conn.execute("""
                SELECT violation_type, COUNT(*) as count 
                FROM violations 
                GROUP BY violation_type
            """).fetchall()
            
            # Last violation timestamp handling
            last_violation_timestamp = conn.execute("""
                SELECT timestamp FROM violations 
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            
            last_violation = None
            if last_violation_timestamp and "timestamp" in last_violation_timestamp:
                ts = last_violation_timestamp["timestamp"]
                if isinstance(ts, str):
                    try:
                        last_violation = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
                    except ValueError:
                        # fallback for alternative timestamp formats
                        last_violation = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                elif isinstance(ts, (int, float)):
                    last_violation = datetime.fromtimestamp(ts, tz=timezone.utc)

            return {
                "total_violations": int(total),
                "violations_by_type": {row["violation_type"]: int(row["count"]) for row in type_counts},
                "last_violation": last_violation
            }
    def get_rois(self) -> List[ROI]:
        """Get all ROI configurations"""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM rois").fetchall()
            rois = []
            for row in rows:
                roi = ROI(
                    id=row["id"],
                    name=row["name"],
                    coordinates=json.loads(row["coordinates"]),
                    active=bool(row["active"]),
                    violation_type=ViolationType(row["violation_type"])
                )
                rois.append(roi)
            return rois
    
    def upsert_roi(self, roi: ROI) -> None:
        """Insert or update ROI configuration"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rois 
                (id, name, coordinates, active, violation_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                roi.id,
                roi.name,
                json.dumps(roi.coordinates),
                roi.active,
                roi.violation_type.value,
                datetime.now()
            ))
            conn.commit()
            logger.info(f"ROI {roi.id} updated successfully")

    def insert_video_frame(self, vf: VideoFrame) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO video_frames
                (frame_id, timestamp, frame_data, width, height, fps, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                vf.frame_id,
                int(vf.timestamp.timestamp()),
                vf.frame_data,
                vf.metadata.width,
                vf.metadata.height,
                vf.metadata.fps,
                vf.metadata.source
            ))
            conn.commit()

    def insert_detection(self, det: Detection, frame_id: str) -> int:
        with self.get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO detections
                (frame_id, class_name, confidence, bbox, track_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                frame_id,
                det.class_name.value,
                det.confidence,
                json.dumps(det.bbox.dict()),
                det.track_id,
                det.hand_id,
            ))
            conn.commit()
            return cur.lastrowid

    def insert_detection_result(self, dr: DetectionResult) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO detection_results
                (frame_id, timestamp, detections, violations, frame_data, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dr.frame_id,
                int(dr.timestamp.timestamp()),
                json.dumps([d.dict() for d in dr.detections]),
                json.dumps([v.dict() for v in dr.violations]),
                dr.frame_data,
                dr.processing_time
            ))
            conn.commit()

    def insert_health_check(self, hc: HealthCheck) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO health_checks
                (timestamp, status, version)
                VALUES (?, ?, ?)
            """, (
                int(hc.timestamp.timestamp()),
                hc.status,
                hc.version
            ))
            conn.commit()

    def insert_system_status(self, ss: SystemStatus) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO system_status
                (timestamp, services, metrics, uptime)
                VALUES (?, ?, ?, ?)
            """, (
                int(datetime.now(timezone.utc).timestamp()),
                json.dumps(ss.services),
                json.dumps(ss.metrics),
                ss.uptime
            ))
            conn.commit()

    def insert_websocket_message(self, msg: WebSocketMessage) -> int:
        with self.get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO websocket_messages
                (type, frame_id, timestamp, image_data, detections, violations)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                msg.type,
                msg.frame_id,
                int(msg.timestamp.timestamp()),
                msg.image_data,
                json.dumps([d.dict() for d in msg.detections]),
                json.dumps([v.dict() for v in msg.violations])
            ))
            conn.commit()
            return cur.lastrowid
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import logging
from .models import ViolationRecord, ROI, ViolationType

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = "violations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            # Create violations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id VARCHAR(36) NOT NULL,
                    timestamp DATETIME NOT NULL,
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
            
            # Insert default ROI if not exists
            conn.execute("""
                INSERT OR IGNORE INTO rois (id, name, coordinates, violation_type)
                VALUES ('protein_container', 'Protein Container', '[50, 50, 300, 300]', 'no_scooper')
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
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
        """Insert a new violation record"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO violations 
                (frame_id, timestamp, violation_type, roi_id, confidence, frame_path, bounding_boxes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation.frame_id,
                violation.timestamp,
                violation.violation_type.value,
                violation.roi_id,
                violation.confidence,
                violation.frame_path,
                json.dumps([det.dict() for det in violation.bounding_boxes]),
                json.dumps(violation.metadata) if violation.metadata else None
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_violations(self, limit: int = 50, offset: int = 0, 
                      start_time: Optional[datetime] = None, 
                      end_time: Optional[datetime] = None) -> List[ViolationRecord]:
        """Get violation records with pagination and filtering"""
        query = "SELECT * FROM violations WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            violations = []
            for row in rows:
                violation = ViolationRecord(
                    id=row['id'],
                    frame_id=row['frame_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    violation_type=ViolationType(row['violation_type']),
                    roi_id=row['roi_id'],
                    confidence=row['confidence'],
                    frame_path=row['frame_path'],
                    bounding_boxes=json.loads(row['bounding_boxes']) if row['bounding_boxes'] else [],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None,
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                violations.append(violation)
            return violations
    
    def get_violation_count(self, start_time: Optional[datetime] = None, 
                           end_time: Optional[datetime] = None) -> int:
        """Get total violation count"""
        query = "SELECT COUNT(*) as count FROM violations WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        with self.get_connection() as conn:
            result = conn.execute(query, params).fetchone()
            return result['count']
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get violation summary statistics"""
        with self.get_connection() as conn:
            # Total violations
            total = conn.execute("SELECT COUNT(*) as count FROM violations").fetchone()['count']
            
            # Violations by type
            type_counts = conn.execute("""
                SELECT violation_type, COUNT(*) as count 
                FROM violations 
                GROUP BY violation_type
            """).fetchall()
            
            # Last violation
            last_violation = conn.execute("""
                SELECT timestamp FROM violations 
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            
            return {
                'total_violations': total,
                'violations_by_type': {row['violation_type']: row['count'] for row in type_counts},
                'last_violation': datetime.fromisoformat(last_violation['timestamp']) if last_violation else None
            }
    
    def get_rois(self) -> List[ROI]:
        """Get all ROI configurations"""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM rois").fetchall()
            rois = []
            for row in rows:
                roi = ROI(
                    id=row['id'],
                    name=row['name'],
                    coordinates=json.loads(row['coordinates']),
                    active=bool(row['active']),
                    violation_type=ViolationType(row['violation_type'])
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


# Unit Tests for Shared Components

import unittest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch

import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import (
    Database, RabbitMQClient, ViolationRecord, ROI, 
    ViolationType, DetectionClass, Detection, BoundingBox
)

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db = Database(self.temp_db.name)
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_init_database(self):
        """Test database initialization"""
        # Check if tables exist
        with self.db.get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [table['name'] for table in tables]
            
            self.assertIn('violations', table_names)
            self.assertIn('rois', table_names)
    
    def test_insert_violation(self):
        """Test violation insertion"""
        violation = ViolationRecord(
            frame_id="test-frame-123",
            timestamp=datetime.now(),
            violation_type=ViolationType.NO_SCOOPER,
            roi_id="protein_container",
            confidence=0.95,
            frame_path="/path/to/frame.jpg",
            bounding_boxes=[],
            metadata={"test": "data"}
        )
        
        violation_id = self.db.insert_violation(violation)
        self.assertIsInstance(violation_id, int)
        self.assertGreater(violation_id, 0)
    
    def test_get_violations(self):
        """Test violation retrieval"""
        # Insert test violation
        violation = ViolationRecord(
            frame_id="test-frame-123",
            timestamp=datetime.now(),
            violation_type=ViolationType.NO_SCOOPER,
            roi_id="protein_container",
            confidence=0.95,
            frame_path="/path/to/frame.jpg",
            bounding_boxes=[],
            metadata={"test": "data"}
        )
        self.db.insert_violation(violation)
        
        # Retrieve violations
        violations = self.db.get_violations(limit=10)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].frame_id, "test-frame-123")
    
    def test_get_rois(self):
        """Test ROI retrieval"""
        rois = self.db.get_rois()
        self.assertGreater(len(rois), 0)  # Should have default ROI
        self.assertEqual(rois[0].id, "protein_container")
    
    def test_upsert_roi(self):
        """Test ROI insertion/update"""
        roi = ROI(
            id="test_roi",
            name="Test ROI",
            coordinates=[10, 10, 100, 100],
            active=True,
            violation_type=ViolationType.NO_SCOOPER
        )
        
        self.db.upsert_roi(roi)
        
        # Retrieve and verify
        rois = self.db.get_rois()
        test_roi = next((r for r in rois if r.id == "test_roi"), None)
        self.assertIsNotNone(test_roi)
        self.assertEqual(test_roi.name, "Test ROI")

class TestModels(unittest.TestCase):
    def test_detection_model(self):
        """Test Detection model"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detection = Detection(
            class_name=DetectionClass.HAND,
            confidence=0.95,
            bbox=bbox
        )
        
        self.assertEqual(detection.class_name, DetectionClass.HAND)
        self.assertEqual(detection.confidence, 0.95)
        self.assertEqual(detection.bbox.x1, 10)
    
    def test_roi_model(self):
        """Test ROI model"""
        roi = ROI(
            id="test_roi",
            name="Test ROI",
            coordinates=[10, 10, 100, 100],
            violation_type=ViolationType.NO_SCOOPER
        )
        
        self.assertEqual(roi.id, "test_roi")
        self.assertEqual(len(roi.coordinates), 4)
        self.assertTrue(roi.active)  # Default value

class TestRabbitMQClient(unittest.TestCase):
    @patch('pika.BlockingConnection')
    def test_connect(self, mock_connection):
        """Test RabbitMQ connection"""
        mock_channel = Mock()
        mock_connection.return_value.channel.return_value = mock_channel
        
        client = RabbitMQClient()
        
        # Verify connection was attempted
        mock_connection.assert_called_once()
        mock_channel.queue_declare.assert_called()
    
    @patch('pika.BlockingConnection')
    def test_publish_message(self, mock_connection):
        """Test message publishing"""
        mock_channel = Mock()
        mock_connection.return_value.channel.return_value = mock_channel
        mock_connection.return_value.is_closed = False
        
        client = RabbitMQClient()
        message = {"test": "data"}
        
        client.publish_message("test_queue", message)
        
        # Verify publish was called
        mock_channel.basic_publish.assert_called_once()

if __name__ == '__main__':
    unittest.main()


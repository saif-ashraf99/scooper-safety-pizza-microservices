# Integration Tests for the Complete System

import unittest
import time
import threading
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch

import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import (
    Database, RabbitMQPublisher, RabbitMQConsumer,
    ViolationRecord, ViolationType
)

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db = Database(self.temp_db.name)
        
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    @patch('pika.BlockingConnection')
    def test_frame_processing_pipeline(self, mock_connection):
        """Test the complete frame processing pipeline"""
        # Mock RabbitMQ
        mock_channel = Mock()
        mock_connection.return_value.channel.return_value = mock_channel
        mock_connection.return_value.is_closed = False
        
        # Create publisher
        publisher = RabbitMQPublisher()
        
        # Test frame message
        frame_message = {
            "frame_id": "test-frame-123",
            "timestamp": datetime.now().isoformat(),
            "frame_data": "base64_encoded_frame_data",
            "metadata": {
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "source": "test_video.mp4"
            }
        }
        
        # Publish frame
        publisher.publish_frame(frame_message)
        
        # Verify message was published
        mock_channel.basic_publish.assert_called()
        
        # Get the published message
        call_args = mock_channel.basic_publish.call_args
        published_body = call_args[1]['body']
        published_message = json.loads(published_body)
        
        self.assertEqual(published_message['frame_id'], "test-frame-123")
        self.assertIn('timestamp', published_message)
    
    def test_violation_detection_and_storage(self):
        """Test violation detection and database storage"""
        # Create a test violation
        violation = ViolationRecord(
            frame_id="test-frame-456",
            timestamp=datetime.now(),
            violation_type=ViolationType.NO_SCOOPER,
            roi_id="protein_container",
            confidence=0.95,
            frame_path="/path/to/violation_frame.jpg",
            bounding_boxes=[],
            metadata={"description": "Hand in ROI without scooper"}
        )
        
        # Store violation
        violation_id = self.db.insert_violation(violation)
        self.assertIsInstance(violation_id, int)
        
        # Retrieve and verify
        violations = self.db.get_violations(limit=1)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].frame_id, "test-frame-456")
        self.assertEqual(violations[0].violation_type, ViolationType.NO_SCOOPER)
        
        # Test summary
        summary = self.db.get_violation_summary()
        self.assertEqual(summary['total_violations'], 1)
        self.assertEqual(summary['violations_by_type']['no_scooper'], 1)
    
    def test_roi_configuration(self):
        """Test ROI configuration management"""
        from shared import ROI
        
        # Create custom ROI
        custom_roi = ROI(
            id="custom_container",
            name="Custom Container",
            coordinates=[100, 100, 400, 400],
            active=True,
            violation_type=ViolationType.NO_SCOOPER
        )
        
        # Store ROI
        self.db.upsert_roi(custom_roi)
        
        # Retrieve and verify
        rois = self.db.get_rois()
        custom = next((r for r in rois if r.id == "custom_container"), None)
        self.assertIsNotNone(custom)
        self.assertEqual(custom.name, "Custom Container")
        self.assertEqual(custom.coordinates, [100, 100, 400, 400])

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db = Database(self.temp_db.name)
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_database_performance(self):
        """Test database performance with multiple violations"""
        start_time = time.time()
        
        # Insert 100 violations
        for i in range(100):
            violation = ViolationRecord(
                frame_id=f"test-frame-{i}",
                timestamp=datetime.now(),
                violation_type=ViolationType.NO_SCOOPER,
                roi_id="protein_container",
                confidence=0.95,
                frame_path=f"/path/to/frame_{i}.jpg",
                bounding_boxes=[],
                metadata={"test": f"violation_{i}"}
            )
            self.db.insert_violation(violation)
        
        insert_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        violations = self.db.get_violations(limit=50)
        retrieval_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(len(violations), 50)
        
        # Performance assertions (adjust based on requirements)
        self.assertLess(insert_time, 5.0)  # Should insert 100 records in < 5 seconds
        self.assertLess(retrieval_time, 1.0)  # Should retrieve 50 records in < 1 second
        
        print(f"Insert time for 100 records: {insert_time:.3f}s")
        print(f"Retrieval time for 50 records: {retrieval_time:.3f}s")

if __name__ == '__main__':
    unittest.main()


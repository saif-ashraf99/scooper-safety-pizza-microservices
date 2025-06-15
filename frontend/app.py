import streamlit as st
import requests
import json
import time
import base64
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import threading
import queue
import websocket
import logging

# Configure page
st.set_page_config(
    page_title="Pizza Violation Detection System",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
DEFAULT_API_BASE_URL = "http://streaming_service:8000"
DEFAULT_WS_URL = "ws://streaming_service:8000/ws/video"

# Initialize session state
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_BASE_URL
if 'ws_url' not in st.session_state:
    st.session_state.ws_url = DEFAULT_WS_URL
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'violation_alert' not in st.session_state:
    st.session_state.violation_alert = False
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=10)

# Utility functions
def get_api_data(endpoint):
    """Fetch data from FastAPI endpoint"""
    try:
        response = requests.get(f"{st.session_state.api_base_url}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def post_api_data(endpoint, data):
    """Post data to FastAPI endpoint"""
    try:
        response = requests.post(
            f"{st.session_state.api_base_url}{endpoint}", 
            json=data, 
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def decode_base64_image(base64_string):
    """Decode base64 image string to bytes"""
    try:
        image_data = base64.b64decode(base64_string)
        return BytesIO(image_data)
    except Exception as e:
        st.error(f"Image decode error: {str(e)}")
        return None

# WebSocket handler (simplified approach)
class WebSocketHandler:
    def __init__(self, url, frame_queue):
        self.url = url
        self.frame_queue = frame_queue
        self.ws = None
        self.running = False
    
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'frame':
                # Put frame data in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(data)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(data)
                    except queue.Empty:
                        pass
        except Exception as e:
            logging.error(f"WebSocket message error: {e}")
    
    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        logging.info("WebSocket connection closed")
        self.running = False
    
    def on_open(self, ws):
        logging.info("WebSocket connection opened")
        self.running = True
    
    def start(self):
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            self.ws.run_forever()
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")

# Sidebar
st.sidebar.title("🍕 Pizza Violation Detection")
st.sidebar.markdown("---")

# API Configuration
with st.sidebar.expander("⚙️ Configuration", expanded=False):
    api_url = st.text_input(
        "FastAPI Base URL", 
        value=st.session_state.api_base_url,
        help="Base URL for the FastAPI backend"
    )
    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url
        st.session_state.ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/video"
        st.rerun()

# Connection status
health_data = get_api_data("/health")
if health_data:
    st.sidebar.success("✅ Connected to Backend")
else:
    st.sidebar.error("❌ Backend Disconnected")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)

# Main content
st.title("🍕 Pizza Store Violation Detection System")
st.markdown("Real-time monitoring of scooper hygiene compliance")

# Create main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Live Video Feed")
    
    # Video display area
    video_placeholder = st.empty()
    
    # Try to get latest frame from queue or API
    latest_frame_data = None
    
    # Check if we have frames in queue (from WebSocket)
    try:
        latest_frame_data = st.session_state.frame_queue.get_nowait()
    except queue.Empty:
        # Fallback: try to get a static frame from API (if available)
        # Note: This is a simplified approach since the original API doesn't have a static frame endpoint
        # In a real implementation, you might add a /api/latest_frame endpoint to the FastAPI backend
        pass
    
    if latest_frame_data and latest_frame_data.get('image_data'):
        # Display the frame
        image_bytes = decode_base64_image(latest_frame_data['image_data'])
        if image_bytes:
            video_placeholder.image(image_bytes, caption="Live Feed with Detections", use_column_width=True)
            
            # Check for violations
            violations = latest_frame_data.get('violations', [])
            if violations:
                st.error(f"🚨 VIOLATION DETECTED! {len(violations)} violation(s) found")
                for violation in violations:
                    st.warning(f"• {violation.get('description', 'Violation in ' + violation.get('roi_id', 'unknown ROI'))}")
    else:
        video_placeholder.info("📺 Waiting for video feed... Make sure the frame reader service is running.")

with col2:
    st.header("📊 System Metrics")
    
    # Get violation summary
    violation_summary = get_api_data("/api/violations/summary")
    if violation_summary:
        # Key metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Total Violations", 
                violation_summary.get('total_violations', 0)
            )
        with col_b:
            st.metric(
                "Processing Status", 
                violation_summary.get('processing_status', 'Unknown').title()
            )
        
        # Violations by type
        violations_by_type = violation_summary.get('violations_by_type', {})
        if violations_by_type:
            st.subheader("Violations by Type")
            for violation_type, count in violations_by_type.items():
                st.metric(violation_type.replace('_', ' ').title(), count)
    
    # Get system status
    system_status = get_api_data("/api/status")
    if system_status:
        st.subheader("System Status")
        
        # Service status
        services = system_status.get('services', {})
        for service, status in services.items():
            if status == 'active' or status == 'connected':
                st.success(f"✅ {service.replace('_', ' ').title()}: {status}")
            else:
                st.error(f"❌ {service.replace('_', ' ').title()}: {status}")
        
        # Performance metrics
        metrics = system_status.get('metrics', {})
        if metrics:
            st.subheader("Performance")
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Frames Processed", metrics.get('frames_processed', 0))
                st.metric("Processing FPS", f"{metrics.get('processing_fps', 0):.1f}")
            with col_d:
                st.metric("Violations Detected", metrics.get('violations_detected', 0))
                st.metric("Uptime", system_status.get('uptime', 'Unknown'))

# Violation History Section
st.header("📋 Recent Violations")

# Get violation history
violations_data = get_api_data("/api/violations?limit=10")
if violations_data and violations_data.get('violations'):
    violations_list = violations_data['violations']
    
    # Convert to DataFrame for better display
    df_data = []
    for violation in violations_list:
        df_data.append({
            'ID': violation.get('id', 'N/A'),
            'Timestamp': violation.get('timestamp', 'N/A'),
            'Type': violation.get('violation_type', 'N/A').replace('_', ' ').title(),
            'ROI': violation.get('roi_id', 'N/A'),
            'Confidence': f"{violation.get('confidence', 0):.2f}",
            'Frame ID': violation.get('frame_id', 'N/A')[:8] + '...' if violation.get('frame_id') else 'N/A'
        })
    
    if df_data:
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No violations recorded yet.")
else:
    st.info("No violation data available. Check if the detection service is running.")

# ROI Configuration Section
st.header("🎯 ROI Configuration")

# Get current ROIs
rois_data = get_api_data("/api/rois")
if rois_data:
    st.subheader("Current ROIs")
    
    roi_df_data = []
    for roi in rois_data:
        roi_df_data.append({
            'ID': roi.get('id', 'N/A'),
            'Name': roi.get('name', 'N/A'),
            'Coordinates': str(roi.get('coordinates', [])),
            'Active': '✅' if roi.get('active', False) else '❌',
            'Violation Type': roi.get('violation_type', 'N/A').replace('_', ' ').title()
        })
    
    if roi_df_data:
        roi_df = pd.DataFrame(roi_df_data)
        st.dataframe(roi_df, use_container_width=True)
    
    # Add new ROI form
    with st.expander("➕ Add New ROI", expanded=False):
        with st.form("add_roi_form"):
            roi_id = st.text_input("ROI ID", placeholder="e.g., sauce_container")
            roi_name = st.text_input("ROI Name", placeholder="e.g., Sauce Container")
            
            col_x1, col_y1, col_x2, col_y2 = st.columns(4)
            with col_x1:
                x1 = st.number_input("X1", min_value=0, value=50)
            with col_y1:
                y1 = st.number_input("Y1", min_value=0, value=50)
            with col_x2:
                x2 = st.number_input("X2", min_value=0, value=300)
            with col_y2:
                y2 = st.number_input("Y2", min_value=0, value=300)
            
            roi_active = st.checkbox("Active", value=True)
            
            submitted = st.form_submit_button("Add ROI")
            
            if submitted and roi_id and roi_name:
                roi_data = {
                    "id": roi_id,
                    "name": roi_name,
                    "coordinates": [x1, y1, x2, y2],
                    "active": roi_active,
                    "violation_type": "no_scooper"
                }
                
                result = post_api_data("/api/rois", roi_data)
                if result:
                    st.success(f"ROI '{roi_name}' added successfully!")
                    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Pizza Violation Detection System | Powered by FastAPI, RabbitMQ & Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Start WebSocket connection in background (simplified approach)
# Note: In a production environment, you might want to use a more robust WebSocket client
# or consider using streamlit-webrtc for real-time video streaming
if 'ws_handler' not in st.session_state:
    try:
        import websocket
        st.session_state.ws_handler = WebSocketHandler(st.session_state.ws_url, st.session_state.frame_queue)
        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(target=st.session_state.ws_handler.start, daemon=True)
        ws_thread.start()
    except ImportError:
        st.warning("WebSocket library not available. Real-time video streaming disabled. Install 'websocket-client' for full functionality.")
    except Exception as e:
        st.warning(f"WebSocket connection failed: {e}. Using polling mode instead.")


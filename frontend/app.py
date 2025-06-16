import streamlit as st
import requests
import json
import time
import base64
from datetime import datetime
import pandas as pd
import threading
import queue
import websocket
import logging
from io import BytesIO

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

# WebSocket handler class
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
                try:
                    self.frame_queue.put_nowait(data)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(data)
                    except queue.Empty:
                        pass
        except Exception as e:
            logging.error(f"WebSocket message error: {e}")

    def on_error(self, ws, error):
        logging.error(f"❌ WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"⚠️ WebSocket closed: {close_status_code}, {close_msg}")
        self.running = False

    def on_open(self, ws):
        logging.info("✅ WebSocket connection opened from Streamlit")
        self.running = True

    def start(self):
        while True:
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
            time.sleep(5)

# Initialize WebSocket thread
if 'ws_handler' not in st.session_state or 'ws_thread' not in st.session_state:
    try:
        ws_handler = WebSocketHandler(st.session_state.ws_url, st.session_state.frame_queue)
        ws_thread = threading.Thread(target=ws_handler.start, daemon=True)
        ws_thread.start()
        st.session_state.ws_handler = ws_handler
        st.session_state.ws_thread = ws_thread
        logging.info("WebSocket thread started")
    except Exception as e:
        logging.error(f"Failed to start WebSocket handler: {e}")
        st.session_state.ws_handler = None
        st.session_state.ws_thread = None

# API utilities
def get_api_data(endpoint):
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
    try:
        image_data = base64.b64decode(base64_string)
        return BytesIO(image_data)
    except Exception as e:
        st.error(f"Image decode error: {str(e)}")
        return None

# Sidebar
st.sidebar.title("🍕 Pizza Violation Detection")
st.sidebar.markdown("---")

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

health_data = get_api_data("/health")
if health_data:
    st.sidebar.success("✅ Connected to Backend")
else:
    st.sidebar.error("❌ Backend Disconnected")

auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)

# Main UI
st.title("🍕 Pizza Store Violation Detection System")
st.markdown("Real-time monitoring of scooper hygiene compliance")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Live Video Feed")
    video_placeholder = st.empty()
    latest_frame_data = None
    try:
        latest_frame_data = st.session_state.frame_queue.get_nowait()
    except queue.Empty:
        pass

    if latest_frame_data and latest_frame_data.get('image_data'):
        image_bytes = decode_base64_image(latest_frame_data['image_data'])
        if image_bytes:
            video_placeholder.image(image_bytes, caption="Live Feed with Detections", use_column_width=True)
            violations = latest_frame_data.get('violations', [])
            if violations:
                st.error(f"🚨 VIOLATION DETECTED! {len(violations)} violation(s) found")
                for violation in violations:
                    st.warning(f"• {violation.get('description', 'Violation in ' + violation.get('roi_id', 'unknown ROI'))}")
    else:
        video_placeholder.info("📺 Waiting for video feed... Make sure the frame reader service is running.")

with col2:
    st.header("📊 System Metrics")
    violation_summary = get_api_data("/api/violations/summary")
    if violation_summary:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Violations", violation_summary.get('total_violations', 0))
        with col_b:
            st.metric("Processing Status", violation_summary.get('processing_status', 'Unknown').title())

        violations_by_type = violation_summary.get('violations_by_type', {})
        if violations_by_type:
            st.subheader("Violations by Type")
            for violation_type, count in violations_by_type.items():
                st.metric(violation_type.replace('_', ' ').title(), count)

    system_status = get_api_data("/api/status")
    if system_status:
        st.subheader("System Status")
        services = system_status.get('services', {})
        for service, status in services.items():
            if status in ['active', 'connected']:
                st.success(f"✅ {service.replace('_', ' ').title()}: {status}")
            # else:
            #     st.error(f"❌ {service.replace('_', ' ').title()}: {status}")
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

# Violation History
st.header("📋 Recent Violations")
violations_data = get_api_data("/api/violations?limit=10")
if violations_data and violations_data.get('violations'):
    violations_list = violations_data['violations']
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
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No violation data available. Check if the detection service is running.")

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

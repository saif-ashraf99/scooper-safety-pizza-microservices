from fastapi.responses import HTMLResponse

def get_frontend_html():
    """Return the frontend HTML content"""
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
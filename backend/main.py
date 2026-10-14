import time
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from backend.camera import VideoCamera

camera: VideoCamera = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera
    print("[INFO] Initializing Arducam Video Capture...")
    camera = VideoCamera(device_id=0, width=1280, height=720, fps=30)
    yield
    print("[INFO] Releasing camera resources...")
    if camera:
        camera.stop()


app = FastAPI(title="Edge Nocturne", lifespan=lifespan)


def generate_mjpeg_stream():
    """Generator yielding multipart JPEG frames for live browser streaming."""
    while True:
        if camera:
            frame_bytes = camera.get_jpeg_bytes(quality=80)
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        time.sleep(0.033)  # ~30 FPS throttle


@app.get("/api/stream")
def video_stream():
    """MJPEG low-latency streaming endpoint."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/snapshot")
def get_snapshot():
    """Returns a single high-quality JPEG snapshot."""
    if not camera:
        return Response(status_code=503, content="Camera not initialized")
    frame_bytes = camera.get_jpeg_bytes(quality=95)
    if not frame_bytes:
        return Response(status_code=500, content="Failed to capture frame")
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/api/stats")
def get_system_stats():
    """Returns system temperature and camera status."""
    temp_str = "N/A"
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True)
        if res.returncode == 0:
            temp_str = res.stdout.strip().replace("temp=", "")
    except Exception:
        pass

    return {
        "status": "online",
        "cpu_temp": temp_str,
        "resolution": f"{camera.width}x{camera.height}" if camera else "offline",
        "fps": camera.fps if camera else 0,
    }


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Dark-mode responsive web dashboard."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge Nocturne - Live Monitor</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #131b2e;
            --border-color: #212d4a;
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --text-primary: #f0f4fc;
            --text-muted: #7e8ba8;
            --status-green: #00e676;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .brand-dot {
            width: 10px;
            height: 10px;
            background-color: var(--status-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--status-green);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .meta-badges {
            display: flex;
            gap: 0.75rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        .badge {
            background: rgba(255, 255, 255, 0.05);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }
        main {
            flex: 1;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .video-container {
            width: 100%;
            max-width: 1100px;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
            position: relative;
        }
        .video-feed {
            width: 100%;
            height: auto;
            display: block;
            aspect-ratio: 16 / 9;
            object-fit: cover;
        }
        .hud-overlay {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(10, 14, 23, 0.75);
            backdrop-filter: blur(8px);
            padding: 0.5rem 0.8rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-family: monospace;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .controls {
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
        }
        .btn {
            background: linear-gradient(135deg, #1e2942, #141b2d);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        .btn:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-dot"></div>
            <div class="brand-title">EDGE NOCTURNE</div>
        </div>
        <div class="meta-badges">
            <div class="badge" id="temp-badge">CPU: --°C</div>
            <div class="badge">Arducam 1080P IR</div>
            <div class="badge">Live MJPEG</div>
        </div>
    </header>
    <main>
        <div class="video-container">
            <div class="hud-overlay" id="hud">FEED: 1280x720 @ 30 FPS</div>
            <img class="video-feed" src="/api/stream" alt="Edge Nocturne Live Feed" />
        </div>
        <div class="controls">
            <a class="btn" href="/api/snapshot" target="_blank">📸 Take Snapshot</a>
            <button class="btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
        </div>
    </main>
    <script>
        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                if (data.cpu_temp) {
                    document.getElementById('temp-badge').innerText = 'CPU: ' + data.cpu_temp;
                }
            } catch(e) {}
        }
        setInterval(updateStats, 3000);
        updateStats();

        function toggleFullscreen() {
            const el = document.querySelector('.video-container');
            if (!document.fullscreenElement) {
                el.requestFullscreen().catch(err => alert(err.message));
            } else {
                document.exitFullscreen();
            }
        }
    </script>
</body>
</html>
    """

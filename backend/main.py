import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from backend.camera import VideoCamera
from backend.database import (
    EVENTS_DIR,
    auto_match_sightings_for_entity,
    backfill_sightings_embeddings,
    delete_entity,
    get_grouped_entities,
    get_recent_sightings,
    get_storage_stats,
    init_db,
    merge_entities,
    merge_multiple_entities,
    tag_sighting_entity,
)
from backend.version import __version__

camera: VideoCamera = None


class TagRequest(BaseModel):
    name: str
    category: Optional[str] = None


class MergeEntityRequest(BaseModel):
    target_id: int


class MergeMultipleRequest(BaseModel):
    entity_ids: list[int]
    target_name: Optional[str] = None


class ContinuousSettingRequest(BaseModel):
    enabled: bool


class CaptureSettingRequest(BaseModel):
    enabled: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera
    print(f"[INFO] Initializing Edge Nocturne v{__version__}...")
    init_db()
    # Run backfill asynchronously in background so web server starts immediately
    import threading
    threading.Thread(target=backfill_sightings_embeddings, daemon=True).start()

    camera = VideoCamera(device_id=None, width=1920, height=1080, fps=30, enable_ml=True)
    yield
    print("[INFO] Shutting down camera & recorder...")
    if camera:
        camera.stop()


app = FastAPI(title="Edge Nocturne", version=__version__, lifespan=lifespan)


def generate_mjpeg_stream(resolution: str = "1080p"):
    """Generator yielding multipart JPEG frames for live browser streaming at requested resolution."""
    while True:
        try:
            if camera:
                frame_bytes = camera.get_jpeg_bytes(quality=80, resolution=resolution)
                if frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
            time.sleep(0.033)
        except (GeneratorExit, StopIteration):
            break
        except Exception:
            time.sleep(0.05)


@app.get("/api/stream")
def video_stream(res: Optional[str] = "1080p"):
    """MJPEG low-latency streaming endpoint with dynamic resolution selector (1080p, 720p, 480p)."""
    return StreamingResponse(
        generate_mjpeg_stream(resolution=res),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/snapshot")
def get_snapshot():
    """Returns a single clean high-quality JPEG snapshot."""
    if not camera:
        return Response(status_code=503, content="Camera not initialized")
    frame_bytes = camera.get_jpeg_bytes(quality=95)
    if not frame_bytes:
        return Response(status_code=500, content="Failed to capture frame")
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/api/stats")
def get_system_stats():
    """Returns system temperature, camera status, ML detections, and storage telemetry."""
    temp_str = "N/A"
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True)
        if res.returncode == 0:
            temp_str = res.stdout.strip().replace("temp=", "")
    except Exception:
        pass

    detections = []
    inference_ms = 0.0
    is_dvr_active = False
    is_recording_event = False
    is_capture_active = True

    if camera:
        detections, inference_ms = camera.get_latest_detections()
        is_dvr_active = camera.is_continuous_recording()
        is_recording_event = camera.recorder.is_recording_event
        is_capture_active = camera.is_capture_active

    return {
        "status": "online" if is_capture_active else "paused",
        "is_capture_active": is_capture_active,
        "version": __version__,
        "cpu_temp": temp_str,
        "resolution": f"{camera.width}x{camera.height}" if camera else "offline",
        "fps": camera.fps if (camera and is_capture_active) else 0,
        "inference_ms": inference_ms,
        "detections": detections,
        "storage": get_storage_stats(),
        "continuous_recording": is_dvr_active,
        "is_recording_event": is_recording_event,
    }


@app.get("/api/events")
def list_events(limit: int = 50, category: Optional[str] = None):
    """Returns chronologically ordered sightings with thumbnails and clip links."""
    return get_recent_sightings(limit=limit, category=category)


@app.get("/api/entities")
def list_entities():
    """Returns unique identified entities with their sightings count and history."""
    return get_grouped_entities()


@app.post("/api/events/{sighting_id}/tag")
def tag_event(sighting_id: int, req: TagRequest):
    """Tags a sighting with an identity name (e.g. 'My VW Touran' or 'Stefan')."""
    entity_id = tag_sighting_entity(sighting_id, req.name.strip(), req.category)
    return {"status": "ok", "entity_id": entity_id, "name": req.name.strip()}


@app.post("/api/entities/{entity_id}/auto-match")
def auto_match_entity(entity_id: int):
    """Scans untagged sightings and retroactively links any matching this entity's visual fingerprint."""
    matched_count = auto_match_sightings_for_entity(entity_id, min_similarity=0.88, min_margin=0.03)
    return {"status": "ok", "matched_count": matched_count}


@app.post("/api/entities/{source_id}/merge")
def merge_entity(source_id: int, req: MergeEntityRequest):
    """Merges all sightings and visual embedding prototypes from source_id into target_id."""
    res = merge_entities(source_id, req.target_id)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/entities/merge-multiple")
def merge_multiple(req: MergeMultipleRequest):
    """Merges multiple selected entities into one unified entity."""
    if len(req.entity_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 entities required for merging")
    res = merge_multiple_entities(req.entity_ids, req.target_name)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.delete("/api/entities/{entity_id}")
def remove_entity(entity_id: int, delete_sightings: bool = True):
    """Deletes an entity and optionally removes its false alarm sightings and media files."""
    res = delete_entity(entity_id, delete_sightings=delete_sightings)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/re-index")
def trigger_reindex(force: bool = False):
    """Extracts missing or upgraded embeddings from stored snapshot files."""
    count = backfill_sightings_embeddings(force_recompute=force)
    return {"status": "ok", "backfilled_count": count}


@app.post("/api/settings/continuous")
def toggle_continuous_loop(req: ContinuousSettingRequest):
    """Enables or disables continuous DVR 5-minute loop recording."""
    if not camera:
        raise HTTPException(status_code=503, detail="Camera not initialized")
    camera.set_continuous_recording(req.enabled)
    return {"status": "ok", "continuous_recording": req.enabled}


@app.post("/api/settings/capture")
def toggle_camera_capture(req: CaptureSettingRequest):
    """Starts or stops live camera capture, streaming, and ML inference."""
    if not camera:
        raise HTTPException(status_code=503, detail="Camera not initialized")
    camera.set_capture_active(req.enabled)
    return {"status": "ok", "is_capture_active": req.enabled}


@app.get("/api/media/clips/{filename}")
def stream_clip(filename: str):
    """Serves recorded MP4 event video clips with range request support."""
    filepath = os.path.join(EVENTS_DIR, os.path.basename(filename))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(filepath, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


@app.get("/api/media/snapshots/{filename}")
def get_sighting_snapshot(filename: str):
    """Serves 1080p high-resolution event snapshot images."""
    filepath = os.path.join(EVENTS_DIR, os.path.basename(filename))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(filepath, media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Dark-mode dashboard with Live Monitor, Sightings Gallery, and Video Modal."""
    return HTML_TEMPLATE.replace("{{VERSION}}", __version__)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge Nocturne - Edge Vision & Security</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #131b2e;
            --bg-card: #18223a;
            --border-color: #212d4a;
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --text-primary: #f0f4fc;
            --text-muted: #7e8ba8;
            --status-green: #00e676;
            --alert-orange: #ff9100;
            --alert-red: #ff5252;
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
            padding: 0.8rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
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
            gap: 0.6rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .badge {
            background: rgba(255, 255, 255, 0.05);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            font-size: 0.85rem;
            color: var(--text-muted);
            transition: all 0.3s ease;
        }
        .badge-detection {
            font-weight: 600;
            border-color: rgba(0, 230, 118, 0.3);
            color: var(--status-green);
        }
        .badge-detection.detected {
            border-color: rgba(255, 145, 0, 0.6);
            color: var(--alert-orange);
            background: rgba(255, 145, 0, 0.12);
            box-shadow: 0 0 12px rgba(255, 145, 0, 0.25);
            animation: pulse-alert 1.5s infinite;
        }
        @keyframes pulse-alert {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        .nav-tabs {
            display: flex;
            gap: 0.5rem;
            background: var(--bg-secondary);
            padding: 0.5rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        .tab-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .tab-btn.active {
            color: var(--text-primary);
            background: rgba(0, 242, 254, 0.1);
            border-color: var(--accent-cyan);
        }
        main {
            flex: 1;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .tab-content {
            width: 100%;
            max-width: 1100px;
            display: none;
        }
        .tab-content.active {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-container {
            width: 100%;
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
        .hud-detection-banner {
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            right: 1rem;
            background: rgba(10, 14, 23, 0.85);
            backdrop-filter: blur(10px);
            padding: 0.6rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            border: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .controls-row {
            width: 100%;
            margin-top: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .dvr-toggle-box {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--bg-card);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        .switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background-color: #2a3756;
            transition: .3s;
            border-radius: 24px;
        }
        .slider:before {
            position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }
        input:checked + .slider { background-color: var(--status-green); }
        input:checked + .slider:before { transform: translateX(20px); }
        .res-bar {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--bg-card);
            padding: 0.3rem 0.6rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        .res-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .res-btn.active {
            background: rgba(0, 242, 254, 0.15);
            color: var(--accent-cyan);
            border-color: var(--accent-cyan);
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
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }
        .btn:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
        }
        /* Sightings Gallery Cards */
        .sightings-grid {
            width: 100%;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.25rem;
            margin-top: 1rem;
        }
        .sighting-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .sighting-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-cyan);
        }
        .sighting-thumb {
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
            background: #000;
        }
        .sighting-body {
            padding: 1rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .sighting-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sighting-label {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: capitalize;
        }
        .sighting-conf {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--status-green);
        }
        .sighting-time {
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        .entity-pill {
            display: inline-block;
            background: rgba(0, 242, 254, 0.1);
            color: var(--accent-cyan);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(0, 242, 254, 0.2);
            width: fit-content;
        }
        .sighting-actions {
            margin-top: auto;
            padding-top: 0.75rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .btn-sm {
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
        }
        .tag-input-row {
            display: flex;
            gap: 0.4rem;
            margin-top: 0.5rem;
        }
        .tag-input {
            flex: 1;
            background: #0d1424;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }
        /* Modal Video Player */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(8px);
            align-items: center;
            justify-content: center;
        }
        .modal.open { display: flex; }
        .modal-content {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            max-width: 900px;
            width: 92%;
            overflow: hidden;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.8);
        }
        .modal-header {
            padding: 0.8rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }
        .modal-close {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 1.4rem;
            cursor: pointer;
        }
        .modal-close:hover { color: var(--text-primary); }
        .modal-video {
            width: 100%;
            max-height: 70vh;
            display: block;
            background: #000;
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
            <button class="btn btn-sm" onclick="openTailscaleModal()" style="padding: 0.2rem 0.6rem; font-size: 0.8rem;">🌐 Remote Access</button>
            <div class="badge" id="capture-badge" style="font-weight: 600; color: var(--status-green);">📹 Capture: Active</div>
            <div class="badge badge-detection" id="detection-badge">🟢 Scene Clear</div>
            <div class="badge" id="storage-badge">💾 Disk: --% Free</div>
            <div class="badge" id="latency-badge">⚡ ML: -- ms</div>
            <div class="badge" id="temp-badge">CPU: --°C</div>
            <div class="badge" id="version-badge" style="font-family: monospace; color: var(--accent-cyan); font-weight: 600;">v{{VERSION}}</div>
        </div>
    </header>
    
    <div class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('live')">📹 Live Monitor</button>
        <button class="tab-btn" onclick="switchTab('sightings')">📁 Sightings Timeline (<span id="sightings-count">0</span>)</button>
    </div>

    <main>
        <!-- Tab 1: Live Monitor -->
        <div class="tab-content active" id="tab-live">
            <div class="video-container">
                <div class="hud-overlay" id="hud-res">FEED: 1920x1080 @ 30 FPS</div>
                <img class="video-feed" src="/api/stream" alt="Edge Nocturne Live Feed" />
                <div class="hud-detection-banner" id="detection-banner">
                    <span id="banner-text">AI Scanner: Watching scene...</span>
                    <span style="color: var(--text-muted); font-size: 0.8rem;">Clean 1080p Stream</span>
                </div>
            </div>
            <div class="controls-row">
                <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;">
                    <button id="btn-toggle-capture" class="btn" onclick="toggleCameraCapture()" style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; padding: 0.55rem 1rem;">
                        <span id="capture-icon">⏸️</span>
                        <span id="capture-text">Stop Capture</span>
                    </button>
                    <div class="dvr-toggle-box">
                        <label class="switch">
                            <input type="checkbox" id="dvr-toggle" onchange="toggleContinuousDVR(this.checked)">
                            <span class="slider"></span>
                        </label>
                        <div>
                            <div style="font-size: 0.9rem; font-weight: 600;">Continuous DVR Loop</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">Keep last 24h • 15% Min Free Guard</div>
                        </div>
                    </div>
                </div>
                <div class="res-bar">
                    <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600;">Stream:</span>
                    <button class="res-btn active" id="btn-res-1080p" onclick="setResolution('1080p')">1080p</button>
                    <button class="res-btn" id="btn-res-720p" onclick="setResolution('720p')">720p</button>
                    <button class="res-btn" id="btn-res-480p" onclick="setResolution('480p')">480p</button>
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <a class="btn" href="/api/snapshot" target="_blank">📸 Snapshot</a>
                    <button class="btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
                </div>
            </div>
        </div>

        <!-- Tab 2: Sightings Timeline & Unique Entities -->
        <div class="tab-content" id="tab-sightings">
            <div style="width: 100%; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
                <div style="display: flex; gap: 0.5rem; background: var(--bg-card); padding: 0.3rem; border-radius: 8px; border: 1px solid var(--border-color);">
                    <button class="tab-btn active" id="btn-view-entities" onclick="switchSightingsSubView('entities')">👥 Unique Entities</button>
                    <button class="tab-btn" id="btn-view-all" onclick="switchSightingsSubView('all')">📅 All Sightings Feed</button>
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <button class="btn btn-sm" onclick="triggerReindex()">🔍 Re-Scan Embeddings</button>
                    <button class="btn btn-sm" onclick="refreshCurrentSightingsView()">🔄 Refresh</button>
                </div>
            </div>
            <div class="sightings-grid" id="sightings-container">
                <p style="color: var(--text-muted); padding: 2rem;">Loading detection events...</p>
            </div>
        </div>
    </main>

    <!-- Modal Video Player -->
    <div class="modal" id="video-modal">
        <div class="modal-content">
            <div class="modal-header">
                <span style="font-weight: 600;" id="modal-title">Playback Sighting Clip</span>
                <button class="modal-close" onclick="closeModal()">✕</button>
            </div>
            <video class="modal-video" id="player" controls playsinline preload="auto"></video>
        </div>
    </div>

    <!-- Modal Tailscale Setup -->
    <div class="modal" id="tailscale-modal">
        <div class="modal-content" style="max-width: 580px; padding: 1.5rem;">
            <div class="modal-header" style="border: none; padding: 0 0 1rem 0;">
                <div style="font-size: 1.15rem; font-weight: 700; color: var(--accent-cyan); display: flex; align-items: center; gap: 0.5rem;">
                    🌐 Tailscale VPN Remote Access
                </div>
                <button class="modal-close" onclick="closeTailscaleModal()">✕</button>
            </div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.2rem; line-height: 1.4;">
                Access your Edge Nocturne camera stream and historical sightings anywhere in the world from your smartphone or laptop securely without opening any ports or touching your home router.
            </p>
            <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.9rem; margin-bottom: 0.9rem; font-size: 0.82rem;">
                <strong style="color: var(--text-primary); display: block; margin-bottom: 0.4rem;">1. Install & Login on Raspberry Pi (One-time)</strong>
                <code style="background: #090e1a; padding: 0.4rem 0.6rem; border-radius: 4px; display: block; word-break: break-all; color: var(--accent-cyan); font-family: monospace;">
                    curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
                </code>
                <span style="color: var(--text-muted); display: block; margin-top: 0.4rem;">Click the Tailscale authentication URL shown in the terminal to connect your Pi to your account.</span>
            </div>
            <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.9rem; margin-bottom: 0.9rem; font-size: 0.82rem;">
                <strong style="color: var(--text-primary); display: block; margin-bottom: 0.4rem;">2. Install Tailscale on your Smartphone</strong>
                <span style="color: var(--text-muted);">Download the Tailscale app (iOS / Android), sign in with the same account, and turn VPN switch ON.</span>
            </div>
            <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.9rem; font-size: 0.82rem;">
                <strong style="color: var(--text-primary); display: block; margin-bottom: 0.4rem;">3. Connect from Mobile Browser</strong>
                <span style="color: var(--text-muted);">Open your browser on your phone outside Wi-Fi and navigate to: <code style="color: var(--status-green);">http://&lt;Pi-Tailscale-IP&gt;:8000</code>.</span>
            </div>
            <div style="text-align: right; margin-top: 1.2rem;">
                <button class="btn btn-sm" onclick="closeTailscaleModal()">Got it</button>
            </div>
        </div>
    </div>

    <script>
        let currentSubView = 'entities';

        function switchTab(tab) {
            document.querySelectorAll('.nav-tabs .tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            if (tab === 'live') {
                document.querySelectorAll('.nav-tabs .tab-btn')[0].classList.add('active');
                document.getElementById('tab-live').classList.add('active');
            } else {
                document.querySelectorAll('.nav-tabs .tab-btn')[1].classList.add('active');
                document.getElementById('tab-sightings').classList.add('active');
                refreshCurrentSightingsView();
            }
        }

        function switchSightingsSubView(view) {
            currentSubView = view;
            document.getElementById('btn-view-entities').classList.toggle('active', view === 'entities');
            document.getElementById('btn-view-all').classList.toggle('active', view === 'all');
            refreshCurrentSightingsView();
        }

        function refreshCurrentSightingsView() {
            if (currentSubView === 'entities') {
                loadGroupedEntities();
            } else {
                loadAllSightings();
            }
        }

        let isCaptureActive = true;

        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                if (data.version) {
                    document.getElementById('version-badge').innerText = 'v' + data.version;
                }
                if (data.resolution && document.getElementById('hud-res')) {
                    if (data.is_capture_active === false) {
                        document.getElementById('hud-res').innerText = 'FEED: PAUSED (Standby Mode)';
                    } else {
                        document.getElementById('hud-res').innerText = 'FEED: ' + data.resolution + ' @ ' + (data.fps || 30) + ' FPS';
                    }
                }
                if (data.cpu_temp) {
                    document.getElementById('temp-badge').innerText = 'CPU: ' + data.cpu_temp;
                }
                if (data.inference_ms !== undefined) {
                    document.getElementById('latency-badge').innerText = '⚡ ML: ' + data.inference_ms + ' ms';
                }
                if (data.storage && data.storage.free_percent !== undefined) {
                    document.getElementById('storage-badge').innerText = '💾 Disk: ' + data.storage.free_percent + '% Free';
                }
                if (data.continuous_recording !== undefined) {
                    document.getElementById('dvr-toggle').checked = data.continuous_recording;
                }

                // Capture Active State Handler
                if (data.is_capture_active !== undefined) {
                    isCaptureActive = data.is_capture_active;
                    const btn = document.getElementById('btn-toggle-capture');
                    const icon = document.getElementById('capture-icon');
                    const text = document.getElementById('capture-text');
                    const badge = document.getElementById('capture-badge');
                    
                    if (isCaptureActive) {
                        if (btn) {
                            btn.style.background = 'rgba(239, 68, 68, 0.15)';
                            btn.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                            btn.style.color = '#fca5a5';
                        }
                        if (icon) icon.innerText = '⏸️';
                        if (text) text.innerText = 'Stop Capture';
                        if (badge) {
                            badge.innerText = '📹 Capture: Active';
                            badge.style.color = 'var(--status-green)';
                            badge.style.borderColor = 'var(--border-color)';
                        }
                    } else {
                        if (btn) {
                            btn.style.background = 'rgba(16, 185, 129, 0.2)';
                            btn.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                            btn.style.color = '#6ee7b7';
                        }
                        if (icon) icon.innerText = '▶️';
                        if (text) text.innerText = 'Start Capture';
                        if (badge) {
                            badge.innerText = '⏸️ Capture: Paused';
                            badge.style.color = '#fca5a5';
                            badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                        }
                    }
                }

                // Live Detections
                const detBadge = document.getElementById('detection-badge');
                const bannerText = document.getElementById('banner-text');

                if (!isCaptureActive) {
                    detBadge.innerText = '⏸️ Scanner Idle';
                    detBadge.classList.remove('detected');
                    bannerText.innerHTML = '<span style="color: #fca5a5;">⏸️ Live Camera Ingestion Paused</span>';
                } else if (data.detections && data.detections.length > 0) {
                    const topItems = data.detections.map(d => `${d.label} (${Math.round(d.confidence * 100)}%)`).join(', ');
                    detBadge.innerText = '🎯 Detected: ' + topItems;
                    detBadge.classList.add('detected');
                    bannerText.innerHTML = '🎯 <strong style="color: var(--alert-orange);">' + topItems + '</strong> in frame';
                } else {
                    detBadge.innerText = '🟢 Scene Clear';
                    detBadge.classList.remove('detected');
                    bannerText.innerText = 'AI Scanner: Scene Clear';
                }
            } catch(e) {}
        }
        setInterval(updateStats, 1000);
        updateStats();

        async function toggleCameraCapture() {
            const btn = document.getElementById('btn-toggle-capture');
            const targetState = !isCaptureActive;
            if (btn) btn.disabled = true;
            try {
                const res = await fetch('/api/settings/capture', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ enabled: targetState })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    isCaptureActive = data.is_capture_active;
                    const feed = document.querySelector('.video-feed');
                    if (feed) {
                        feed.src = '/api/stream?res=' + currentResolution + '&t=' + Date.now();
                    }
                    updateStats();
                }
            } catch(e) {
                alert('Failed to toggle camera capture: ' + e);
            } finally {
                if (btn) btn.disabled = false;
            }
        }

        async function toggleContinuousDVR(enabled) {
            try {
                await fetch('/api/settings/continuous', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ enabled })
                });
            } catch(e) {
                alert('Failed to update DVR setting');
            }
        }

        // 1. Grouped Unique Entities View
        let allEntitiesCache = [];
        const selectedMergeSet = new Set();

        async function loadGroupedEntities(showSpinner = false) {
            const container = document.getElementById('sightings-container');
            const savedScrollY = window.scrollY;
            if (showSpinner || allEntitiesCache.length === 0) {
                container.innerHTML = '<p style="color: var(--text-muted); padding: 2rem;">Loading unique subjects...</p>';
            }
            try {
                const res = await fetch('/api/entities');
                const data = await res.json();
                const entities = data.entities || [];
                const untagged = data.untagged || [];
                allEntitiesCache = entities;
                
                const totalCount = entities.reduce((sum, e) => sum + (e.sightings_count || 0), 0) + untagged.length;
                document.getElementById('sightings-count').innerText = totalCount;

                if (entities.length === 0 && untagged.length === 0) {
                    container.innerHTML = '<p style="color: var(--text-muted); padding: 2rem;">No subjects recorded yet. Step in front of the camera!</p>';
                    return;
                }

                let html = '<div id="merge-action-bar" style="grid-column: 1 / -1; display: none;"></div>';

                // Render Identified Entities
                entities.forEach(ent => {
                    const primarySnap = ent.primary_snapshot ? `/api/media/snapshots/${ent.primary_snapshot}` : '';
                    const sightingsList = ent.sightings || [];
                    const isSelected = selectedMergeSet.has(ent.id);
                    const cardBorder = isSelected 
                        ? 'border: 2px solid #a855f7; box-shadow: 0 0 15px rgba(168, 85, 247, 0.35);' 
                        : 'border: 1px solid var(--accent-cyan);';

                    html += `
                        <div class="sighting-card" id="card-ent-${ent.id}" style="${cardBorder}">
                            <img class="sighting-thumb" src="${primarySnap}" alt="${ent.name}">
                            <div class="sighting-body">
                                <div class="sighting-title">
                                    <span class="sighting-label">🏷️ ${ent.name}</span>
                                    <span class="entity-pill">${ent.sightings_count} Sighting${ent.sightings_count > 1 ? 's' : ''}</span>
                                </div>
                                <div class="sighting-time">🕒 Last Seen: ${ent.last_seen}</div>
                                
                                <div style="display: flex; gap: 0.4rem; margin-top: 0.5rem; flex-wrap: wrap;">
                                    <button class="btn btn-sm" onclick="toggleEntityHistory('ent-${ent.id}')">
                                        📂 History (${sightingsList.length})
                                    </button>
                                    <button class="btn btn-sm" style="background: rgba(0, 242, 254, 0.12); border-color: var(--accent-cyan); color: var(--accent-cyan);" onclick="autoMatchEntity(${ent.id}, '${ent.name}')">
                                        ⚡ Auto-Match
                                    </button>
                                    <button class="btn btn-sm" id="btn-merge-${ent.id}" style="${isSelected ? 'background: #a855f7; border-color: #c084fc; color: #fff; font-weight: 600;' : 'background: rgba(168, 85, 247, 0.15); border-color: #a855f7; color: #c084fc;'}" onclick="toggleMergeSelect(${ent.id})">
                                        ${isSelected ? '✓ Selected' : '🔗 Merge'}
                                    </button>
                                    <button class="btn btn-sm" style="background: rgba(255, 68, 68, 0.12); border-color: #ff4444; color: #ff6b6b; padding: 0.35rem 0.6rem;" title="Delete False Alarm" onclick="deleteEntityPrompt(${ent.id}, '${ent.name}', ${ent.sightings_count})">
                                        🗑️
                                    </button>
                                </div>

                                <div id="ent-${ent.id}" style="display: none; margin-top: 0.75rem; border-top: 1px solid var(--border-color); padding-top: 0.5rem;">
                                    ${sightingsList.map((s, idx) => `
                                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem;">
                                            <span>#${idx+1} ${s.timestamp.slice(11, 19)} (${Math.round(s.confidence*100)}%)</span>
                                            <div style="display: flex; gap: 0.3rem;">
                                                ${s.clip_filename ? `<button class="btn btn-sm" style="padding: 0.2rem 0.5rem;" onclick="playClip('${s.clip_filename}', '${ent.name} - ${s.timestamp}')">▶ Play</button>` : ''}
                                                <a class="btn btn-sm" style="padding: 0.2rem 0.5rem;" href="/api/media/snapshots/${s.snapshot_filename}" target="_blank">📸</a>
                                            </div>
                                            ${s.metadata && s.metadata.server_summary ? `<div style="font-size: 0.72rem; color: var(--accent-cyan); margin-top: 0.3rem; background: rgba(0,255,200,0.06); padding: 0.25rem 0.4rem; border-radius: 4px; border: 1px solid rgba(0,255,200,0.15); line-height: 1.2;">🤖 <em>${s.metadata.server_summary}</em></div>` : ''}
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    `;
                });

                // Render Untagged / Unassigned Sightings
                if (untagged.length > 0) {
                    html += `
                        <div style="grid-column: 1 / -1; margin-top: 1rem;">
                            <h3 style="font-size: 1rem; color: var(--text-muted); margin-bottom: 0.5rem;">Untagged Sightings (${untagged.length})</h3>
                        </div>
                    `;

                    untagged.forEach(s => {
                        html += `
                            <div class="sighting-card">
                                <img class="sighting-thumb" src="/api/media/snapshots/${s.snapshot_filename}" alt="${s.label}" loading="lazy">
                                <div class="sighting-body">
                                    <div class="sighting-title">
                                        <span class="sighting-label">${s.label}</span>
                                        <span class="sighting-conf">${Math.round(s.confidence * 100)}%</span>
                                    </div>
                                    <div class="sighting-time">📅 ${s.timestamp}</div>
                                    
                                    <div class="sighting-actions">
                                        ${s.clip_filename ? `<button class="btn btn-sm" onclick="playClip('${s.clip_filename}', '${s.label} - ${s.timestamp}')">▶ Play Clip</button>` : ''}
                                        <a class="btn btn-sm" href="/api/media/snapshots/${s.snapshot_filename}" target="_blank">📸 1080p</a>
                                    </div>

                                    <div class="tag-input-row">
                                        <input type="text" class="tag-input" id="tag-${s.id}" placeholder="Tag (e.g. stefan)">
                                        <button class="btn btn-sm" onclick="saveTag(${s.id})">Tag</button>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                }

                container.innerHTML = html;
                updateMergeBar();
                if (savedScrollY > 0) {
                    window.scrollTo({ top: savedScrollY, behavior: 'instant' });
                }
            } catch(e) {
                container.innerHTML = '<p style="color: var(--alert-red); padding: 2rem;">Error loading entities.</p>';
            }
        }

        function toggleEntityHistory(elementId) {
            const el = document.getElementById(elementId);
            if (el.style.display === 'none') {
                el.style.display = 'block';
            } else {
                el.style.display = 'none';
            }
        }

        // 2. All Sightings Timeline View
        async function loadAllSightings() {
            const container = document.getElementById('sightings-container');
            container.innerHTML = '<p style="color: var(--text-muted); padding: 2rem;">Loading chronological feed...</p>';
            try {
                const res = await fetch('/api/events?limit=40');
                const sightings = await res.json();
                document.getElementById('sightings-count').innerText = sightings.length;
                
                if (sightings.length === 0) {
                    container.innerHTML = '<p style="color: var(--text-muted); padding: 2rem;">No events recorded yet.</p>';
                    return;
                }

                container.innerHTML = sightings.map(s => `
                    <div class="sighting-card">
                        <img class="sighting-thumb" src="/api/media/snapshots/${s.snapshot_filename}" alt="${s.label}" loading="lazy">
                        <div class="sighting-body">
                            <div class="sighting-title">
                                <span class="sighting-label">${s.label}</span>
                                <span class="sighting-conf">${Math.round(s.confidence * 100)}%</span>
                            </div>
                            <div class="sighting-time">📅 ${s.timestamp}</div>
                            ${s.entity_name ? `<div class="entity-pill">🏷️ ${s.entity_name}</div>` : ''}
                            ${s.metadata && s.metadata.server_summary ? `<div style="font-size: 0.78rem; color: var(--accent-cyan); margin-top: 0.35rem; background: rgba(0,255,200,0.06); padding: 0.35rem 0.5rem; border-radius: 4px; border: 1px solid rgba(0,255,200,0.18); line-height: 1.3;">🤖 <em>${s.metadata.server_summary}</em></div>` : ''}
                            
                            <div class="sighting-actions">
                                ${s.clip_filename ? `<button class="btn btn-sm" onclick="playClip('${s.clip_filename}', '${s.label} - ${s.timestamp}')">▶ Play Clip</button>` : ''}
                                <a class="btn btn-sm" href="/api/media/snapshots/${s.snapshot_filename}" target="_blank">📸 1080p Frame</a>
                            </div>

                            <div class="tag-input-row">
                                <input type="text" class="tag-input" id="tag-${s.id}" placeholder="Tag identity (e.g. stefan)">
                                <button class="btn btn-sm" onclick="saveTag(${s.id})">Tag</button>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch(e) {}
        }

        async function saveTag(id) {
            const input = document.getElementById('tag-' + id);
            const name = input.value.trim();
            if (!name) return;

            try {
                await fetch('/api/events/' + id + '/tag', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ name })
                });
                refreshCurrentSightingsView();
            } catch(e) {
                alert('Failed to save tag');
            }
        }

        let currentResolution = '1080p';
        function setResolution(res) {
            currentResolution = res;
            document.querySelectorAll('.res-btn').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById('btn-res-' + res);
            if (btn) btn.classList.add('active');
            
            const feed = document.querySelector('.video-feed');
            if (feed) {
                feed.src = '/api/stream?res=' + res + '&t=' + Date.now();
            }
        }

        async function autoMatchEntity(id, name) {
            try {
                const res = await fetch('/api/entities/' + id + '/auto-match', { method: 'POST' });
                const data = await res.json();
                alert(`⚡ Matched and grouped ${data.matched_count || 0} sightings into '${name}'!`);
                refreshCurrentSightingsView();
            } catch(e) {
                alert('Auto-matching failed: ' + e);
            }
        }

        function toggleMergeSelect(entityId) {
            if (selectedMergeSet.has(entityId)) {
                selectedMergeSet.delete(entityId);
            } else {
                selectedMergeSet.add(entityId);
            }

            const isSel = selectedMergeSet.has(entityId);
            const card = document.getElementById('card-ent-' + entityId);
            const btn = document.getElementById('btn-merge-' + entityId);

            if (card) {
                card.style.border = isSel ? '2px solid #a855f7' : '1px solid var(--accent-cyan)';
                card.style.boxShadow = isSel ? '0 0 15px rgba(168, 85, 247, 0.35)' : 'none';
            }
            if (btn) {
                btn.style.background = isSel ? '#a855f7' : 'rgba(168, 85, 247, 0.15)';
                btn.style.borderColor = isSel ? '#c084fc' : '#a855f7';
                btn.style.color = isSel ? '#fff' : '#c084fc';
                btn.style.fontWeight = isSel ? '600' : 'normal';
                btn.innerHTML = isSel ? '✓ Selected' : '🔗 Merge';
            }

            updateMergeBar();
        }

        function updateMergeBar() {
            const bar = document.getElementById('merge-action-bar');
            if (!bar) return;
            if (selectedMergeSet.size === 0) {
                bar.style.display = 'none';
                return;
            }
            bar.style.display = 'block';
            const selectedList = allEntitiesCache.filter(e => selectedMergeSet.has(e.id));
            const selectedNames = selectedList.map(e => e.name).join(', ');
            const count = selectedMergeSet.size;

            bar.innerHTML = `
                <div style="background: rgba(168, 85, 247, 0.12); border: 1px solid #a855f7; border-radius: 8px; padding: 0.8rem 1.2rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.8rem;">
                    <div>
                        <strong style="color: #c084fc;">🔗 ${count} Group${count > 1 ? 's' : ''} Selected:</strong>
                        <span style="color: var(--text-muted); font-size: 0.85rem; margin-left: 0.4rem;">${selectedNames}</span>
                    </div>
                    <div style="display: flex; gap: 0.6rem; align-items: center;">
                        ${count >= 2 ? `
                            <button class="btn btn-sm" style="background: linear-gradient(135deg, #a855f7, #6366f1); color: #fff; font-weight: 600; padding: 0.45rem 1.1rem; box-shadow: 0 0 12px rgba(168, 85, 247, 0.4);" onclick="executeBatchMerge()">
                                ✨ Merge all into same entity
                            </button>
                        ` : `
                            <span style="color: var(--text-muted); font-size: 0.8rem;">Select at least 1 more group to merge</span>
                        `}
                        <button class="btn btn-sm" style="background: rgba(255, 255, 255, 0.08);" onclick="clearMergeSelection()">
                            ✕ Clear
                        </button>
                    </div>
                </div>
            `;
        }

        function clearMergeSelection() {
            selectedMergeSet.clear();
            allEntitiesCache.forEach(ent => {
                const card = document.getElementById('card-ent-' + ent.id);
                const btn = document.getElementById('btn-merge-' + ent.id);
                if (card) {
                    card.style.border = '1px solid var(--accent-cyan)';
                    card.style.boxShadow = 'none';
                }
                if (btn) {
                    btn.style.background = 'rgba(168, 85, 247, 0.15)';
                    btn.style.borderColor = '#a855f7';
                    btn.style.color = '#c084fc';
                    btn.style.fontWeight = 'normal';
                    btn.innerHTML = '🔗 Merge';
                }
            });
            updateMergeBar();
        }

        async function deleteEntityPrompt(id, name, count) {
            const nl = String.fromCharCode(10);
            if (!confirm(`Delete entity '${name}' and remove its ${count} sightings?` + nl + nl +
                         `Use this to remove false alarm detections (e.g. ceiling lamps, chairs).`)) {
                return;
            }

            try {
                const res = await fetch('/api/entities/' + id + '?delete_sightings=true', { method: 'DELETE' });
                const data = await res.json();
                if (data.status === 'ok') {
                    selectedMergeSet.delete(id);
                    refreshCurrentSightingsView();
                } else {
                    alert('Delete failed: ' + (data.detail || data.message));
                }
            } catch(e) {
                alert('Delete error: ' + e);
            }
        }

        async function executeBatchMerge() {
            if (selectedMergeSet.size < 2) {
                alert('Please select at least 2 entities to merge.');
                return;
            }

            const selectedEntities = allEntitiesCache.filter(e => selectedMergeSet.has(e.id));
            const defaultTarget = selectedEntities.reduce((max, e) => ((e.sightings_count || 0) > (max.sightings_count || 0) ? e : max), selectedEntities[0]);
            const namesList = selectedEntities.map(e => e.name).join(', ');

            const nl = String.fromCharCode(10);
            const promptMsg = `Merging ${selectedEntities.length} groups: [${namesList}]` + nl + nl +
                              `Enter the target identity name (e.g. 'Stefan'):`;
            const chosenName = prompt(promptMsg, defaultTarget.name);
            if (chosenName === null) return;

            const targetName = chosenName.trim() || defaultTarget.name;

            if (!confirm(`Confirm Merge:` + nl +
                         `Merge [${namesList}] into '${targetName}'?` + nl +
                         `All sightings will be united and their visual profiles combined.`)) {
                return;
            }

            try {
                const res = await fetch('/api/entities/merge-multiple', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        entity_ids: Array.from(selectedMergeSet),
                        target_name: targetName
                    })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    alert(`✨ Successfully merged into '${data.target_name}' (${data.total_sightings} total sightings)!`);
                    selectedMergeSet.clear();
                    refreshCurrentSightingsView();
                } else {
                    alert(`Merge failed: ${data.detail || data.message}`);
                }
            } catch(e) {
                alert('Merge error: ' + e);
            }
        }

        async function triggerReindex() {
            try {
                const res = await fetch('/api/re-index', { method: 'POST' });
                const data = await res.json();
                alert(`Extracted embeddings for ${data.backfilled_count || 0} snapshots.`);
                refreshCurrentSightingsView();
            } catch(e) {
                alert('Re-indexing failed: ' + e);
            }
        }

        function openTailscaleModal() {
            document.getElementById('tailscale-modal').classList.add('open');
        }

        function closeTailscaleModal() {
            document.getElementById('tailscale-modal').classList.remove('open');
        }

        function playClip(filename, title) {
            const modal = document.getElementById('video-modal');
            const player = document.getElementById('player');
            document.getElementById('modal-title').innerText = title;
            
            player.pause();
            player.src = '/api/media/clips/' + filename;
            modal.classList.add('open');
            player.load();
            player.play().catch(e => console.log('Autoplay waiting for user gesture:', e));
        }

        function closeModal() {
            const modal = document.getElementById('video-modal');
            const player = document.getElementById('player');
            player.pause();
            player.src = '';
            modal.classList.remove('open');
        }

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

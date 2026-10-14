# Edge Nocturne - Agent Workspace Context

This document maintains the persistent state, architectural decisions, and current progress for **Edge Nocturne**, readable by both Antigravity IDE and Antigravity CLI (`agy`).

---

## 1. Project Overview & Hardware
* **Hardware**:
  * **Raspberry Pi 5 (8GB RAM)** with official active cooler.
  * **Arducam 1080P Day & Night Vision USB Camera Module** (dual IR LED boards, automatic IR-cut filter).
* **Objective**:
  * On-device video recording and real-time detection of people, animals, and vehicles in low-light/night (IR) and daytime conditions.
  * Web interface accessible via local network and later securely over remote VPN (Tailscale).
  * Event logging into a local database with video snippets/thumbnails.

---

## 2. Agreed Technology Stack
* **ML / Computer Vision**:
  * **Python**
  * **TensorFlow Lite (TFLite)** (using `tflite-runtime` or `ai-edge-litert` for lightweight, on-device edge ML)
  * **OpenCV (`opencv-python`)** for frame capture, pre-processing, and drawing overlays
* **Backend & Recording Engine**:
  * **FastAPI** (Python async REST API)
  * **MJPEG HTTP streaming** for low-latency, cross-platform video feed (with real-time detection bounding boxes)
  * **Dual Recording Modes**:
    1. **Event-Triggered Recording**: Automatically saves short snippet clips + thumbnails when an object/animal/person is detected with high confidence.
    2. **Continuous Loop Recording Mode (Dashcam/DVR style)**:
       * Can be toggled **ON / OFF** from the UI/API.
       * Configurable rolling retention window (e.g., **24 hours / 1 day**).
       * Saves continuous video in segmented chunks (e.g., 5-minute `.mp4` segments).
       * Automatic rolling cleanup worker deletes chunks older than the retention threshold or when disk space exceeds limit.
* **Database**:
  * **SQLite** (storing detection events: timestamp, object class, confidence, clip path, thumbnail path)
* **Frontend**:
  * **React + TypeScript (Vite)**
* **Remote Connectivity**:
  * Local IP: `<pi-ip>` (configured via DHCP / local `.env`)
  * Raspberry Pi user: `<pi-user>`
  * Services: SSH, optional RDP/VNC
  * Planned Remote Access: Tailscale VPN

---

## 3. Current State & Setup Progress
1. **Network & Access**:
   * Pi connectivity established on local network.
   * Dedicated non-root user with sudo configured.
   * Passwordless SSH key authentication established.
2. **Camera Verification**:
   * Device: `Arducam 1080P USB Camera Module` (`0c45:6369 Microdia USB 2.0 Camera`).
   * Device node: `/dev/video0`.
   * Supported formats:
     * **MJPG**: Up to **1920x1080 @ 30 FPS**, **1280x720 @ 30 FPS**, **640x480 @ 30 FPS**.
     * **YUYV**: Uncompressed video stream.
   * System resources: Raspberry Pi OS 64-bit (`aarch64`), 7.4 GB RAM free.
3. **Next Milestone (Phase 1 MVP)**:
   * Setup virtual environment on Pi (`python3 -m venv ~/edge-nocturne/venv`).
   * Install `opencv-python-headless`, `tflite-runtime`, and `fastapi` + `uvicorn`.
   * Capture a test snapshot and verify IR illumination in the dark.
   * Launch initial MJPEG FastAPI stream server.

import base64
import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "storage", "nocturne.db")


def get_server_config() -> Dict[str, Any]:
    """Retrieves server verification configuration from environment or defaults."""
    return {
        "enabled": os.environ.get("NOCTURNE_SERVER_ENABLED", "0").lower() in ("1", "true", "yes"),
        "url": os.environ.get("NOCTURNE_SERVER_URL", "http://192.168.0.100:11434"),
        "model": os.environ.get("NOCTURNE_SERVER_MODEL", "llava:latest"),
    }


def dispatch_server_check(
    snapshot_path: str,
    sighting_ids: List[int],
    detections: List[Dict[str, Any]],
):
    """Asynchronously dispatches an event snapshot to a local home server (e.g. Ollama) for VLM verification."""
    config = get_server_config()
    if not config["enabled"] or not os.path.exists(snapshot_path):
        return

    threading.Thread(
        target=_server_worker,
        args=(snapshot_path, sighting_ids, detections, config),
        daemon=True,
    ).start()


def _server_worker(
    snapshot_path: str,
    sighting_ids: List[int],
    detections: List[Dict[str, Any]],
    config: Dict[str, Any],
):
    """Queries self-hosted Ollama / Vision server and annotates sightings in SQLite with AI description."""
    try:
        with open(snapshot_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "Security camera event detected. In 1-2 concise sentences, describe the subjects "
            "(people, vehicles, or animals) visible, their appearance/color, and what they are doing."
        )

        payload = {
            "model": config["model"],
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
        }

        api_url = f"{config['url'].rstrip('/')}/api/generate"
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            summary = data.get("response", "").strip()

            if summary and sighting_ids:
                print(f"[SERVER CHECK] AI Description: {summary}")
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                for sid in sighting_ids:
                    cursor.execute("SELECT metadata_json FROM sightings WHERE id = ?", (sid,))
                    row = cursor.fetchone()
                    meta = json.loads(row["metadata_json"]) if row and row["metadata_json"] else {}
                    meta["server_summary"] = summary
                    meta["server_model"] = config["model"]
                    cursor.execute(
                        "UPDATE sightings SET metadata_json = ? WHERE id = ?",
                        (json.dumps(meta), sid),
                    )

                conn.commit()
                conn.close()

    except Exception as e:
        # Non-blocking, fails gracefully without affecting recording or UI
        print(f"[SERVER CHECK] Optional verification skipped: {e}")

import json
import os
import shutil
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
EVENTS_DIR = os.path.join(STORAGE_DIR, "events")
CONTINUOUS_DIR = os.path.join(STORAGE_DIR, "continuous")
ARCHIVE_DIR = os.path.join(STORAGE_DIR, "archive")
DB_PATH = os.path.join(STORAGE_DIR, "nocturne.db")


def init_storage():
    """Ensure all media and database directories exist."""
    os.makedirs(EVENTS_DIR, exist_ok=True)
    os.makedirs(CONTINUOUS_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the decoupled SQLite schema for entities and sightings."""
    init_storage()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Unique persistent identities (e.g. "My VW Touran", "Neighbor's Cat", "Stefan")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        category TEXT NOT NULL,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_sightings INTEGER DEFAULT 1,
        embedding_json TEXT
    );
    """)

    # Table 2: Individual sightings / occurrences linked to entities
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sightings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        label TEXT NOT NULL,
        category TEXT NOT NULL,
        confidence REAL NOT NULL,
        snapshot_filename TEXT NOT NULL,
        clip_filename TEXT,
        metadata_json TEXT,
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE SET NULL
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_time ON sightings(timestamp DESC);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_entity ON sightings(entity_id);")

    conn.commit()
    conn.close()


def log_sighting(
    label: str,
    category: str,
    confidence: float,
    snapshot_filename: str,
    clip_filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Logs a new detection sighting with snapshot and optional video clip reference."""
    conn = get_db_connection()
    cursor = conn.cursor()

    meta_str = json.dumps(metadata) if metadata else None

    cursor.execute(
        """
        INSERT INTO sightings (label, category, confidence, snapshot_filename, clip_filename, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (label, category, confidence, snapshot_filename, clip_filename, meta_str),
    )
    sighting_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return sighting_id


def get_recent_sightings(limit: int = 50, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns recent sightings ordered chronologically with optional entity metadata."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT s.id, s.timestamp, s.label, s.category, s.confidence, 
               s.snapshot_filename, s.clip_filename, s.metadata_json,
               e.name as entity_name
        FROM sightings s
        LEFT JOIN entities e ON s.entity_id = e.id
    """
    params = []
    if category:
        query += " WHERE s.category = ?"
        params.append(category)

    query += " ORDER BY s.timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results


def tag_sighting_entity(sighting_id: int, entity_name: str, category: str = "other") -> int:
    """Associates a sighting with an entity identity (creating the entity if new)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get or create entity
    cursor.execute("SELECT id, total_sightings FROM entities WHERE name = ?", (entity_name,))
    row = cursor.fetchone()

    if row:
        entity_id = row["id"]
        cursor.execute(
            """
            UPDATE entities 
            SET last_seen = CURRENT_TIMESTAMP, total_sightings = total_sightings + 1
            WHERE id = ?
            """,
            (entity_id,),
        )
    else:
        cursor.execute(
            "INSERT INTO entities (name, category) VALUES (?, ?)",
            (entity_name, category),
        )
        entity_id = cursor.lastrowid

    # Update sighting
    cursor.execute("UPDATE sightings SET entity_id = ? WHERE id = ?", (entity_id, sighting_id))
    conn.commit()
    conn.close()
    return entity_id


def get_grouped_entities() -> Dict[str, Any]:
    """Returns unique identified entities with their sightings count, primary photo, and appearance history."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e.id, e.name, e.category, e.first_seen, e.last_seen,
               COUNT(s.id) as sightings_count,
               (SELECT snapshot_filename FROM sightings WHERE entity_id = e.id ORDER BY timestamp DESC LIMIT 1) as primary_snapshot,
               (SELECT clip_filename FROM sightings WHERE entity_id = e.id ORDER BY timestamp DESC LIMIT 1) as latest_clip
        FROM entities e
        LEFT JOIN sightings s ON s.entity_id = e.id
        GROUP BY e.id
        ORDER BY e.last_seen DESC
    """)
    entities = [dict(r) for r in cursor.fetchall()]

    for ent in entities:
        cursor.execute("""
            SELECT id, timestamp, label, confidence, snapshot_filename, clip_filename
            FROM sightings
            WHERE entity_id = ?
            ORDER BY timestamp DESC
        """, (ent["id"],))
        ent["sightings"] = [dict(s) for s in cursor.fetchall()]

    cursor.execute("""
        SELECT id, timestamp, label, category, confidence, snapshot_filename, clip_filename
        FROM sightings
        WHERE entity_id IS NULL
        ORDER BY timestamp DESC
    """)
    untagged = [dict(s) for s in cursor.fetchall()]

    conn.close()
    return {"entities": entities, "untagged": untagged}


def get_storage_stats() -> Dict[str, Any]:
    """Returns disk usage and free space statistics for the storage filesystem."""
    try:
        total, used, free = shutil.disk_usage(STORAGE_DIR if os.path.exists(STORAGE_DIR) else BASE_DIR)
        free_pct = round((free / total) * 100.0, 1)
        used_pct = round((used / total) * 100.0, 1)
        return {
            "total_gb": round(total / (1024**3), 1),
            "used_gb": round(used / (1024**3), 1),
            "free_gb": round(free / (1024**3), 1),
            "free_percent": free_pct,
            "used_percent": used_pct,
            "min_free_threshold_percent": 15.0,
        }
    except Exception as e:
        return {"error": str(e), "free_percent": 0.0}

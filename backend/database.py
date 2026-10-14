import json
import os
import shutil
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from backend.fingerprint import fingerprint_engine

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
        embedding_json TEXT,
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE SET NULL
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_time ON sightings(timestamp DESC);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_entity ON sightings(entity_id);")

    # Safe schema migration for existing databases
    cursor.execute("PRAGMA table_info(sightings)")
    columns = [col[1] for col in cursor.fetchall()]
    if "embedding_json" not in columns:
        cursor.execute("ALTER TABLE sightings ADD COLUMN embedding_json TEXT;")

    conn.commit()
    conn.close()


def log_sighting(
    label: str,
    category: str,
    confidence: float,
    snapshot_filename: str,
    clip_filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None,
    entity_id: Optional[int] = None,
) -> int:
    """Logs a new detection sighting with snapshot, clip, embedding, and optional entity reference."""
    conn = get_db_connection()
    cursor = conn.cursor()

    meta_str = json.dumps(metadata) if metadata else None
    emb_str = json.dumps(embedding) if embedding else None

    cursor.execute(
        """
        INSERT INTO sightings (entity_id, label, category, confidence, snapshot_filename, clip_filename, metadata_json, embedding_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (entity_id, label, category, confidence, snapshot_filename, clip_filename, meta_str, emb_str),
    )
    sighting_id = cursor.lastrowid

    # If linked to an existing entity, update last seen and sighting count
    if entity_id:
        cursor.execute(
            """
            UPDATE entities 
            SET last_seen = CURRENT_TIMESTAMP, total_sightings = total_sightings + 1
            WHERE id = ?
            """,
            (entity_id,),
        )

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


def find_matching_entity(
    embedding: List[float],
    category: str,
    min_similarity: float = 0.70,
) -> Tuple[Optional[int], Optional[str], float]:
    """Finds best matching known entity using cosine similarity against stored embeddings."""
    if not embedding:
        return None, None, 0.0

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, category, embedding_json FROM entities WHERE category = ? AND embedding_json IS NOT NULL",
        (category,),
    )
    rows = cursor.fetchall()
    conn.close()

    best_match_id = None
    best_match_name = None
    best_similarity = 0.0

    for row in rows:
        try:
            stored_emb = json.loads(row["embedding_json"])
            sim = fingerprint_engine.cosine_similarity(embedding, stored_emb)
            if sim > best_similarity:
                best_similarity = sim
                if sim >= min_similarity:
                    best_match_id = row["id"]
                    best_match_name = row["name"]
        except Exception:
            continue

    return best_match_id, best_match_name, best_similarity


def auto_match_sightings_for_entity(entity_id: int, min_similarity: float = 0.70) -> int:
    """Retroactively scans untagged sightings and links any that match this entity's embedding."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, category, embedding_json FROM entities WHERE id = ?", (entity_id,))
    ent = cursor.fetchone()
    if not ent or not ent["embedding_json"]:
        conn.close()
        return 0

    entity_emb = json.loads(ent["embedding_json"])
    category = ent["category"]

    cursor.execute(
        "SELECT id, snapshot_filename, metadata_json, embedding_json FROM sightings WHERE entity_id IS NULL AND category = ?",
        (category,),
    )
    untagged_rows = cursor.fetchall()

    matched_ids = []
    for row in untagged_rows:
        emb = None
        if row["embedding_json"]:
            try:
                emb = json.loads(row["embedding_json"])
            except Exception:
                pass

        if not emb and row["snapshot_filename"]:
            snap_path = os.path.join(EVENTS_DIR, row["snapshot_filename"])
            if os.path.exists(snap_path):
                import cv2
                img = cv2.imread(snap_path)
                if img is not None:
                    meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                    box = meta.get("box")
                    emb = fingerprint_engine.extract_embedding(img, box)
                    cursor.execute(
                        "UPDATE sightings SET embedding_json = ? WHERE id = ?",
                        (json.dumps(emb), row["id"]),
                    )

        if emb:
            sim = fingerprint_engine.cosine_similarity(entity_emb, emb)
            if sim >= min_similarity:
                matched_ids.append(row["id"])

    for mid in matched_ids:
        cursor.execute("UPDATE sightings SET entity_id = ? WHERE id = ?", (entity_id, mid))

    if matched_ids:
        cursor.execute(
            "UPDATE entities SET total_sightings = total_sightings + ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?",
            (len(matched_ids), entity_id),
        )

    conn.commit()
    conn.close()
    return len(matched_ids)


def tag_sighting_entity(sighting_id: int, entity_name: str, category: str = "other") -> int:
    """Associates a sighting with an entity identity, updates visual fingerprint, and auto-matches similar sightings."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Fetch sighting embedding (or extract from snapshot if missing)
    cursor.execute(
        "SELECT id, snapshot_filename, metadata_json, embedding_json FROM sightings WHERE id = ?",
        (sighting_id,),
    )
    sighting = cursor.fetchone()

    sighting_emb = None
    if sighting:
        if sighting["embedding_json"]:
            try:
                sighting_emb = json.loads(sighting["embedding_json"])
            except Exception:
                pass
        if not sighting_emb and sighting["snapshot_filename"]:
            snap_path = os.path.join(EVENTS_DIR, sighting["snapshot_filename"])
            if os.path.exists(snap_path):
                import cv2
                img = cv2.imread(snap_path)
                if img is not None:
                    meta = json.loads(sighting["metadata_json"]) if sighting["metadata_json"] else {}
                    box = meta.get("box")
                    sighting_emb = fingerprint_engine.extract_embedding(img, box)
                    cursor.execute(
                        "UPDATE sightings SET embedding_json = ? WHERE id = ?",
                        (json.dumps(sighting_emb), sighting_id),
                    )

    # 2. Get or create entity
    cursor.execute("SELECT id, total_sightings, embedding_json FROM entities WHERE name = ?", (entity_name,))
    row = cursor.fetchone()

    if row:
        entity_id = row["id"]
        current_count = row["total_sightings"] or 1
        current_emb = json.loads(row["embedding_json"]) if row["embedding_json"] else None

        new_centroid = (
            fingerprint_engine.merge_embeddings(current_emb, sighting_emb, count=current_count)
            if sighting_emb
            else current_emb
        )
        new_emb_str = json.dumps(new_centroid) if new_centroid else None

        cursor.execute(
            """
            UPDATE entities 
            SET last_seen = CURRENT_TIMESTAMP, total_sightings = total_sightings + 1, embedding_json = ?
            WHERE id = ?
            """,
            (new_emb_str, entity_id),
        )
    else:
        new_emb_str = json.dumps(sighting_emb) if sighting_emb else None
        cursor.execute(
            "INSERT INTO entities (name, category, embedding_json) VALUES (?, ?, ?)",
            (entity_name, category, new_emb_str),
        )
        entity_id = cursor.lastrowid

    # 3. Update sighting
    cursor.execute("UPDATE sightings SET entity_id = ? WHERE id = ?", (entity_id, sighting_id))
    conn.commit()
    conn.close()

    # 4. Auto-match all other past sightings matching this entity!
    auto_match_sightings_for_entity(entity_id, min_similarity=0.70)

    return entity_id


def backfill_sightings_embeddings() -> int:
    """Backfills missing embeddings for existing historical sightings from their snapshots."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, snapshot_filename, metadata_json FROM sightings WHERE embedding_json IS NULL")
    rows = cursor.fetchall()
    updated_count = 0

    for r in rows:
        if r["snapshot_filename"]:
            snap_path = os.path.join(EVENTS_DIR, r["snapshot_filename"])
            if os.path.exists(snap_path):
                import cv2
                img = cv2.imread(snap_path)
                if img is not None:
                    meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                    box = meta.get("box")
                    emb = fingerprint_engine.extract_embedding(img, box)
                    cursor.execute(
                        "UPDATE sightings SET embedding_json = ? WHERE id = ?",
                        (json.dumps(emb), r["id"]),
                    )
                    updated_count += 1

    conn.commit()
    conn.close()
    return updated_count


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

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

    # Ensure existing entities with 'other' category inherit category from their sightings
    cursor.execute("""
        UPDATE entities
        SET category = (
            SELECT s.category FROM sightings s 
            WHERE s.entity_id = entities.id AND s.category != 'other' 
            ORDER BY s.id DESC LIMIT 1
        )
        WHERE (category = 'other' OR category IS NULL) AND EXISTS (
            SELECT 1 FROM sightings s WHERE s.entity_id = entities.id AND s.category != 'other'
        );
    """)

    conn.commit()
    conn.close()


def get_or_create_auto_entity(category: str, embedding: Optional[Any] = None) -> int:
    """Creates a new sequential auto-named entity (e.g. 'Car 1', 'Animal 1', 'Person 1') when a novel subject appears."""
    conn = get_db_connection()
    cursor = conn.cursor()

    prefix_map = {
        "vehicle": "Car",
        "animal": "Animal",
        "person": "Person",
    }
    prefix = prefix_map.get(category, "Subject")

    cursor.execute("SELECT name FROM entities WHERE name LIKE ?", (f"{prefix} %",))
    rows = cursor.fetchall()
    existing_nums = []
    for r in rows:
        name = r["name"]
        parts = name.split()
        if len(parts) >= 2 and parts[1].isdigit():
            existing_nums.append(int(parts[1]))

    next_num = max(existing_nums, default=0) + 1
    new_name = f"{prefix} {next_num}"
    emb_str = json.dumps(embedding) if embedding else None

    cursor.execute(
        """
        INSERT INTO entities (name, category, total_sightings, embedding_json)
        VALUES (?, ?, 0, ?)
        """,
        (new_name, category, emb_str),
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"[ENTITY] Auto-provisioned new group '{new_name}' (ID {new_id}) for {category}")
    return new_id


def log_sighting(
    label: str,
    category: str,
    confidence: float,
    snapshot_filename: str,
    clip_filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None,
    entity_id: Optional[int] = None,
    auto_create: bool = True,
) -> int:
    """Logs a new detection sighting with snapshot, clip, embedding, and optional entity reference."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Automatically create a new group if novel subject has an embedding
    if not entity_id and auto_create and embedding:
        entity_id = get_or_create_auto_entity(category, embedding)

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

    # If linked to an existing entity, update last seen, sighting count, and merge centroid embedding
    if entity_id:
        cursor.execute(
            """
            UPDATE entities 
            SET last_seen = CURRENT_TIMESTAMP, total_sightings = total_sightings + 1
            WHERE id = ?
            """,
            (entity_id,),
        )
        if embedding:
            cursor.execute("SELECT total_sightings, embedding_json FROM entities WHERE id = ?", (entity_id,))
            ent_row = cursor.fetchone()
            if ent_row and ent_row["embedding_json"]:
                try:
                    cur_count = ent_row["total_sightings"]
                    cur_emb = json.loads(ent_row["embedding_json"])
                    merged_emb = fingerprint_engine.merge_embeddings(cur_emb, embedding, cur_count)
                    cursor.execute(
                        "UPDATE entities SET embedding_json = ? WHERE id = ?",
                        (json.dumps(merged_emb), entity_id),
                    )
                except Exception:
                    pass

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
    results = []
    for r in rows:
        d = dict(r)
        if d.get("metadata_json"):
            try:
                d["metadata"] = json.loads(d["metadata_json"])
            except Exception:
                d["metadata"] = {}
        else:
            d["metadata"] = {}
        results.append(d)
    conn.close()
    return results


def find_matching_entity(
    embedding: List[float],
    category: str,
    min_similarity: float = 0.88,
    min_margin: float = 0.03,
) -> Tuple[Optional[int], Optional[str], float]:
    """Finds best matching known entity using cosine similarity against stored embeddings.
    Requires meeting minimum similarity threshold and a decisive separation margin against other entities.
    """
    if not embedding:
        return None, None, 0.0

    conn = get_db_connection()
    cursor = conn.cursor()
    # Match on specific category OR fallback if entity was categorized as 'other'
    cursor.execute(
        """
        SELECT id, name, category, embedding_json 
        FROM entities 
        WHERE (category = ? OR category = 'other' OR ? = 'other') AND embedding_json IS NOT NULL
        """,
        (category, category),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, None, 0.0

    matches = []
    for row in rows:
        try:
            stored_emb = json.loads(row["embedding_json"])
            sim = fingerprint_engine.cosine_similarity(embedding, stored_emb)
            matches.append((row["id"], row["name"], sim))
        except Exception:
            continue

    if not matches:
        return None, None, 0.0

    matches.sort(key=lambda m: m[2], reverse=True)
    best_id, best_name, best_sim = matches[0]

    if best_sim < min_similarity:
        return None, None, best_sim

    if len(matches) > 1:
        second_sim = matches[1][2]
        if (best_sim - second_sim) < min_margin:
            # Borderline tie / ambiguous match: avoid false attribution
            return None, None, best_sim

    return best_id, best_name, best_sim


def auto_match_sightings_for_entity(
    entity_id: int,
    min_similarity: float = 0.88,
    min_margin: float = 0.03,
) -> int:
    """Retroactively scans untagged sightings and links any that match this entity's embedding using competitive matching."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, category, embedding_json FROM entities WHERE id = ?", (entity_id,))
    ent = cursor.fetchone()
    if not ent:
        conn.close()
        return 0

    entity_emb = None
    if ent["embedding_json"]:
        try:
            entity_emb = json.loads(ent["embedding_json"])
        except Exception:
            pass

    # If entity has no embedding, compute centroid from its existing tagged sightings
    if not entity_emb:
        cursor.execute(
            "SELECT id, snapshot_filename, metadata_json, embedding_json, category FROM sightings WHERE entity_id = ? ORDER BY id ASC",
            (entity_id,),
        )
        existing_sightings = cursor.fetchall()
        for idx, es in enumerate(existing_sightings):
            emb = None
            if es["embedding_json"]:
                emb = json.loads(es["embedding_json"])
            elif es["snapshot_filename"]:
                snap_path = os.path.join(EVENTS_DIR, es["snapshot_filename"])
                if os.path.exists(snap_path):
                    import cv2
                    img = cv2.imread(snap_path)
                    if img is not None:
                        meta = json.loads(es["metadata_json"]) if es["metadata_json"] else {}
                        box = meta.get("box")
                        emb = fingerprint_engine.extract_embedding(img, box, category=es["category"])
                        cursor.execute(
                            "UPDATE sightings SET embedding_json = ? WHERE id = ?",
                            (json.dumps(emb), es["id"]),
                        )
            if emb:
                entity_emb = fingerprint_engine.merge_embeddings(entity_emb, emb, count=idx)
                if ent["category"] == "other" and es["category"] != "other":
                    cursor.execute("UPDATE entities SET category = ? WHERE id = ?", (es["category"], entity_id))

        if entity_emb:
            cursor.execute("UPDATE entities SET embedding_json = ? WHERE id = ?", (json.dumps(entity_emb), entity_id))
            conn.commit()

    if not entity_emb:
        conn.close()
        return 0

    category = ent["category"]

    # Preload all other entities for competitive separation
    cursor.execute(
        """
        SELECT id, embedding_json 
        FROM entities 
        WHERE id != ? AND (category = ? OR category = 'other' OR ? = 'other') AND embedding_json IS NOT NULL
        """,
        (entity_id, category, category),
    )
    other_entities = []
    for o_row in cursor.fetchall():
        try:
            other_entities.append((o_row["id"], json.loads(o_row["embedding_json"])))
        except Exception:
            pass

    # Query untagged sightings matching category (or any if entity is 'other')
    cursor.execute(
        "SELECT id, snapshot_filename, metadata_json, embedding_json FROM sightings WHERE entity_id IS NULL AND (category = ? OR ? = 'other')",
        (category, category),
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
                    emb = fingerprint_engine.extract_embedding(img, box, category=category)
                    cursor.execute(
                        "UPDATE sightings SET embedding_json = ? WHERE id = ?",
                        (json.dumps(emb), row["id"]),
                    )

        if emb:
            sim = fingerprint_engine.cosine_similarity(entity_emb, emb)
            if sim >= min_similarity:
                # Competitive matching: ensure target entity is the highest-scoring entity by a decisive margin
                is_best = True
                for _, other_emb in other_entities:
                    other_sim = fingerprint_engine.cosine_similarity(other_emb, emb)
                    if other_sim >= sim or (sim - other_sim) < min_margin:
                        is_best = False
                        break
                if is_best:
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


def tag_sighting_entity(sighting_id: int, entity_name: str, category: Optional[str] = None) -> int:
    """Associates a sighting with an entity identity, updates visual fingerprint, and auto-matches similar sightings."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Fetch sighting embedding (or extract from snapshot if missing) and sighting's actual category
    cursor.execute(
        "SELECT id, category, snapshot_filename, metadata_json, embedding_json FROM sightings WHERE id = ?",
        (sighting_id,),
    )
    sighting = cursor.fetchone()
    if not sighting:
        conn.close()
        return 0

    sighting_cat = sighting["category"] or "other"
    if not category or category == "other":
        category = sighting_cat

    sighting_emb = None
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
                sighting_emb = fingerprint_engine.extract_embedding(img, box, category=category)
                cursor.execute(
                    "UPDATE sightings SET embedding_json = ? WHERE id = ?",
                    (json.dumps(sighting_emb), sighting_id),
                )

    # 2. Get or create entity (case-insensitive lookup)
    cursor.execute(
        "SELECT id, category, total_sightings, embedding_json FROM entities WHERE LOWER(name) = LOWER(?)",
        (entity_name.strip(),),
    )
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

        # Upgrade category if entity was 'other'
        ent_cat = category if (row["category"] == "other" and category != "other") else row["category"]

        cursor.execute(
            """
            UPDATE entities 
            SET last_seen = CURRENT_TIMESTAMP, total_sightings = total_sightings + 1, embedding_json = ?, category = ?
            WHERE id = ?
            """,
            (new_emb_str, ent_cat, entity_id),
        )
    else:
        new_emb_str = json.dumps(sighting_emb) if sighting_emb else None
        cursor.execute(
            "INSERT INTO entities (name, category, embedding_json) VALUES (?, ?, ?)",
            (entity_name.strip(), category, new_emb_str),
        )
        entity_id = cursor.lastrowid

    # 3. Update sighting
    cursor.execute("UPDATE sightings SET entity_id = ? WHERE id = ?", (entity_id, sighting_id))
    conn.commit()
    conn.close()

    # 4. Auto-match all other past sightings matching this entity!
    auto_match_sightings_for_entity(entity_id, min_similarity=0.88, min_margin=0.03)

    return entity_id


def merge_entities(source_id: int, target_id: int) -> Dict[str, Any]:
    """Merges all sightings and visual embedding prototypes from source_id into target_id, then deletes source_id."""
    if source_id == target_id:
        return {"status": "error", "message": "Cannot merge an entity into itself"}

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, category, total_sightings, first_seen, last_seen, embedding_json FROM entities WHERE id = ?",
        (source_id,),
    )
    source = cursor.fetchone()

    cursor.execute(
        "SELECT id, name, category, total_sightings, first_seen, last_seen, embedding_json FROM entities WHERE id = ?",
        (target_id,),
    )
    target = cursor.fetchone()

    if not source or not target:
        conn.close()
        return {"status": "error", "message": "Source or target entity not found"}

    # 1. Move all sightings from source to target
    cursor.execute("UPDATE sightings SET entity_id = ? WHERE entity_id = ?", (target_id, source_id))
    moved_count = cursor.rowcount

    # 2. Combine metrics
    target_count = target["total_sightings"] or 0
    source_count = source["total_sightings"] or 0
    total_count = target_count + source_count

    dates = [d for d in [target["first_seen"], source["first_seen"]] if d]
    first_seen = min(dates) if dates else None

    dates_last = [d for d in [target["last_seen"], source["last_seen"]] if d]
    last_seen = max(dates_last) if dates_last else None

    # 3. Merge embeddings
    target_emb = json.loads(target["embedding_json"]) if target["embedding_json"] else None
    source_emb = json.loads(source["embedding_json"]) if source["embedding_json"] else None
    merged_emb = fingerprint_engine.merge_embeddings(target_emb, source_emb, count=target_count)
    merged_emb_str = json.dumps(merged_emb) if merged_emb else None

    # Upgrade category if needed
    cat = target["category"]
    if cat == "other" and source["category"] != "other":
        cat = source["category"]

    cursor.execute(
        """
        UPDATE entities
        SET total_sightings = ?, first_seen = ?, last_seen = ?, embedding_json = ?, category = ?
        WHERE id = ?
        """,
        (total_count, first_seen, last_seen, merged_emb_str, cat, target_id),
    )

    # 4. Delete source entity
    cursor.execute("DELETE FROM entities WHERE id = ?", (source_id,))

    conn.commit()
    conn.close()

    print(f"[ENTITY MERGE] Merged '{source['name']}' (ID {source_id}) into '{target['name']}' (ID {target_id}) with {moved_count} sightings.")

    return {
        "status": "ok",
        "source_id": source_id,
        "source_name": source["name"],
        "target_id": target_id,
        "target_name": target["name"],
        "moved_sightings": moved_count,
        "total_sightings": total_count,
    }


def merge_multiple_entities(entity_ids: List[int], target_name: Optional[str] = None) -> Dict[str, Any]:
    """Merges a list of entity IDs into a single unified entity."""
    unique_ids = []
    for eid in entity_ids:
        if eid not in unique_ids:
            unique_ids.append(eid)

    if len(unique_ids) < 2:
        return {"status": "error", "message": "At least 2 unique entities required for merging"}

    conn = get_db_connection()
    cursor = conn.cursor()

    placeholders = ",".join("?" for _ in unique_ids)
    cursor.execute(
        f"SELECT id, name, category, total_sightings, first_seen, last_seen, embedding_json FROM entities WHERE id IN ({placeholders})",
        unique_ids,
    )
    rows = cursor.fetchall()

    if len(rows) < 2:
        conn.close()
        return {"status": "error", "message": f"Only found {len(rows)} matching entities in database"}

    target_ent = None
    if target_name:
        for r in rows:
            if r["name"].strip().lower() == target_name.strip().lower():
                target_ent = r
                break

    if not target_ent:
        target_ent = max(rows, key=lambda r: (r["total_sightings"] or 0))

    target_id = target_ent["id"]
    sources = [r for r in rows if r["id"] != target_id]
    final_name = target_name.strip() if target_name and target_name.strip() else target_ent["name"]

    total_moved = 0
    current_count = target_ent["total_sightings"] or 0
    first_seen = target_ent["first_seen"]
    last_seen = target_ent["last_seen"]
    cat = target_ent["category"]
    current_emb = json.loads(target_ent["embedding_json"]) if target_ent["embedding_json"] else None
    merged_source_names = []

    for src in sources:
        sid = src["id"]
        s_count = src["total_sightings"] or 0
        merged_source_names.append(src["name"])

        # Move sightings
        cursor.execute("UPDATE sightings SET entity_id = ? WHERE entity_id = ?", (target_id, sid))
        total_moved += cursor.rowcount

        # Metrics
        current_count += s_count
        if src["first_seen"]:
            first_seen = min(first_seen, src["first_seen"]) if first_seen else src["first_seen"]
        if src["last_seen"]:
            last_seen = max(last_seen, src["last_seen"]) if last_seen else src["last_seen"]
        if cat == "other" and src["category"] != "other":
            cat = src["category"]

        s_emb = json.loads(src["embedding_json"]) if src["embedding_json"] else None
        current_emb = fingerprint_engine.merge_embeddings(current_emb, s_emb, count=current_count)

        # Delete source entity
        cursor.execute("DELETE FROM entities WHERE id = ?", (sid,))

    final_emb_str = json.dumps(current_emb) if current_emb else None
    cursor.execute(
        """
        UPDATE entities
        SET name = ?, total_sightings = ?, first_seen = ?, last_seen = ?, embedding_json = ?, category = ?
        WHERE id = ?
        """,
        (final_name, current_count, first_seen, last_seen, final_emb_str, cat, target_id),
    )

    conn.commit()
    conn.close()

    print(f"[ENTITY BATCH MERGE] Merged {len(sources)} entities ({merged_source_names}) into '{final_name}' (ID {target_id}) with {total_moved} sightings.")

    return {
        "status": "ok",
        "target_id": target_id,
        "target_name": final_name,
        "merged_sources": merged_source_names,
        "moved_sightings": total_moved,
        "total_sightings": current_count,
    }


def delete_entity(entity_id: int, delete_sightings: bool = True) -> Dict[str, Any]:
    """Deletes an entity (e.g. false alarm). If delete_sightings is True, also removes its associated sightings and media files."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM entities WHERE id = ?", (entity_id,))
    ent = cursor.fetchone()
    if not ent:
        conn.close()
        return {"status": "error", "message": "Entity not found"}

    name = ent["name"]
    removed_sightings = 0

    if delete_sightings:
        cursor.execute("SELECT snapshot_filename, clip_filename FROM sightings WHERE entity_id = ?", (entity_id,))
        sightings = cursor.fetchall()
        for s in sightings:
            if s["snapshot_filename"]:
                sp = os.path.join(EVENTS_DIR, s["snapshot_filename"])
                if os.path.exists(sp):
                    try:
                        os.remove(sp)
                    except Exception:
                        pass
            if s["clip_filename"]:
                cp = os.path.join(EVENTS_DIR, s["clip_filename"])
                if os.path.exists(cp):
                    try:
                        os.remove(cp)
                    except Exception:
                        pass

        cursor.execute("DELETE FROM sightings WHERE entity_id = ?", (entity_id,))
        removed_sightings = cursor.rowcount
    else:
        cursor.execute("UPDATE sightings SET entity_id = NULL WHERE entity_id = ?", (entity_id,))
        removed_sightings = cursor.rowcount

    cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
    conn.commit()
    conn.close()

    print(f"[ENTITY DELETED] Removed entity '{name}' (ID {entity_id}) and {removed_sightings} sightings.")
    return {
        "status": "ok",
        "deleted_entity_id": entity_id,
        "deleted_name": name,
        "removed_sightings": removed_sightings,
    }


def backfill_sightings_embeddings(force_recompute: bool = False) -> int:
    """Backfills missing or upgrades embeddings for existing historical sightings and repairs entity centroid embeddings."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if force_recompute:
        cursor.execute("SELECT id, snapshot_filename, metadata_json, category FROM sightings WHERE snapshot_filename IS NOT NULL")
    else:
        cursor.execute("SELECT id, snapshot_filename, metadata_json, category FROM sightings WHERE embedding_json IS NULL")
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
                    emb = fingerprint_engine.extract_embedding(img, box, category=r["category"])
                    cursor.execute(
                        "UPDATE sightings SET embedding_json = ? WHERE id = ?",
                        (json.dumps(emb), r["id"]),
                    )
                    updated_count += 1

    # Repair any entities with missing embeddings or 'other' category
    cursor.execute("SELECT id, name, category, embedding_json FROM entities")
    entities = cursor.fetchall()
    for ent in entities:
        ent_id = ent["id"]
        cursor.execute(
            "SELECT embedding_json, category FROM sightings WHERE entity_id = ? AND embedding_json IS NOT NULL",
            (ent_id,),
        )
        ent_sightings = cursor.fetchall()
        if ent_sightings:
            centroid = None
            detected_cat = ent["category"]
            for idx, s in enumerate(ent_sightings):
                s_emb = json.loads(s[0])
                centroid = fingerprint_engine.merge_embeddings(centroid, s_emb, count=idx)
                if detected_cat == "other" and s[1] != "other":
                    detected_cat = s[1]
            if centroid:
                cursor.execute(
                    "UPDATE entities SET embedding_json = ?, category = ? WHERE id = ?",
                    (json.dumps(centroid), detected_cat, ent_id),
                )

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
               (SELECT snapshot_filename FROM sightings WHERE entity_id = e.id ORDER BY id ASC LIMIT 1) as primary_snapshot,
               (SELECT clip_filename FROM sightings WHERE entity_id = e.id ORDER BY timestamp DESC LIMIT 1) as latest_clip
        FROM entities e
        LEFT JOIN sightings s ON s.entity_id = e.id
        GROUP BY e.id
        ORDER BY e.last_seen DESC
    """)
    entities = [dict(r) for r in cursor.fetchall()]

    for ent in entities:
        cursor.execute("""
            SELECT id, timestamp, label, confidence, snapshot_filename, clip_filename, metadata_json
            FROM sightings
            WHERE entity_id = ?
            ORDER BY timestamp DESC
        """, (ent["id"],))
        s_rows = []
        for s in cursor.fetchall():
            sd = dict(s)
            if sd.get("metadata_json"):
                try:
                    sd["metadata"] = json.loads(sd["metadata_json"])
                except Exception:
                    sd["metadata"] = {}
            else:
                sd["metadata"] = {}
            s_rows.append(sd)
        ent["sightings"] = s_rows

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

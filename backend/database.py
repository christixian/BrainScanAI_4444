import sqlite3
import json
from datetime import datetime

DB_NAME = "brainscan_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Check if table exists to migrate schema if needed (simple check)
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if c.fetchone():
        # Check if image_url column exists
        c.execute("PRAGMA table_info(predictions)")
        columns = [info[1] for info in c.fetchall()]
        if "image_url" not in columns:
            print("Migrating database: Adding image_url and heatmap_base64 columns...")
            c.execute("ALTER TABLE predictions ADD COLUMN image_url TEXT")
            c.execute("ALTER TABLE predictions ADD COLUMN heatmap_base64 TEXT")
    else:
        c.execute('''CREATE TABLE predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      prediction_4class TEXT,
                      prediction_binary TEXT,
                      confidence_scores TEXT,
                      binary_confidence REAL,
                      heatmap_base64 TEXT,
                      image_url TEXT)''')
    conn.commit()
    conn.close()

def add_prediction(prediction_4class, prediction_binary, confidence_scores, binary_confidence, heatmap_base64=None, image_url=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO predictions (timestamp, prediction_4class, prediction_binary, confidence_scores, binary_confidence, heatmap_base64, image_url) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (timestamp, prediction_4class, prediction_binary, json.dumps(confidence_scores), binary_confidence, heatmap_base64, image_url))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "prediction_4class": row["prediction_4class"],
            "prediction_binary": row["prediction_binary"],
            "confidence_scores": json.loads(row["confidence_scores"]),
            "binary_confidence": row["binary_confidence"],
            "heatmap_base64": row["heatmap_base64"],
            "image_url": row["image_url"]
        })
    conn.close()
    return history

def clear_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

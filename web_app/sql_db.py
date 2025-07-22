import sqlite3

conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    label TEXT,
    confidence REAL,
    timestamp TEXT,
    gcs_url TEXT
)
""")

conn.commit()
conn.close()

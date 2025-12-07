import sqlite3

conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    hour_of_day TEXT,
    image_path TEXT,
    detection_count INTEGER,
    max_detection_area REAL,
    detection_data TEXT
)
""")

conn.commit()
conn.close()
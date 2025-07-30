from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sqlite3
from datetime import datetime, timedelta
from google.cloud import storage


def upload_to_gcs(bucket_name: str, local_path: str, destination_blob: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(local_path,
			      content_type='image/jpg')

    expiration_minutes = 60
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET",
    )

    return url

app = FastAPI()
gcs_bucket = "home_wildlife_tracker"

@app.post("/upload")
async def upload_detection(
    image: UploadFile = File(...),
    label: str = Form(...),
    totalDetections: float = Form(...),
    timestamp: str = Form(...)
):
    # Step 1: Save image locally
    safe_filename = f"{timestamp.replace(' ', '_')}_{image.filename}"
    local_path = os.path.join("images", safe_filename)
    os.makedirs("images", exist_ok=True)
    
    with open(local_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Step 2: Upload image to GCS
    gcs_url = upload_to_gcs(gcs_bucket, local_path, f"detections/{safe_filename}")
    # Step 3: Write metadata to SQLite
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (filename, label, totalDetections, timestamp, gcs_url)
        VALUES (?, ?, ?, ?, ?)
    """, (image.filename, label, totalDetections, timestamp, gcs_url) )
    conn.commit()
    conn.close()

    # Step 4: Return response
    metadata = {
        "filename": image.filename,
        "label": label,
        "totalDetections": totalDetections,
        "timestamp": timestamp,
        "gcs_url": gcs_url
    }

    return JSONResponse(content=metadata)


@app.get("/detections")
def get_detections():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("SELECT label, totalDetections, timestamp, gcs_url FROM detections ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "label": row[0],
            "totalDetections": row[1],
            "timestamp": row[2],
            "gcs_url": row[3]
        }
        for row in rows
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_methods=["*"],
    allow_headers=["*"],
)

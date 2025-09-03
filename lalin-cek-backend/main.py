import json
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException, Depends, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from typing import List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import cv2
import numpy as np

# Mengimpor dictionary `processed_frames` dari cv_worker
from cv_worker import start_cv_processing, processed_frames
from models import Stream, Event, StreamCreate
import database as db

app = FastAPI(
    title="LALINCEK API",
    description="API untuk Sistem Deteksi Anomali Lalu Lintas",
    version="1.0.0"
)

# ... (Middleware, WebSocket Manager, Event Handlers, dan endpoint lain tidak berubah) ...
# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, stream_id: int):
        await websocket.accept()
        if stream_id not in self.active_connections:
            self.active_connections[stream_id] = []
        self.active_connections[stream_id].append(websocket)

    def disconnect(self, websocket: WebSocket, stream_id: int):
        if stream_id in self.active_connections and websocket in self.active_connections[stream_id]:
            self.active_connections[stream_id].remove(websocket)

    async def broadcast(self, stream_id: int, message: dict):
        if stream_id in self.active_connections:
            message_str = json.dumps(message, default=str)
            for connection in self.active_connections[stream_id]:
                await connection.send_text(message_str)

manager = ConnectionManager()

# --- Event Handlers ---
@app.on_event("startup")
async def startup():
    await db.init_db()

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {"message": "LALINCEK API is running"}

@app.post("/add_stream", response_model=Stream, tags=["Stream Management"])
async def add_stream_endpoint(stream_data: StreamCreate, db_session: AsyncSession = Depends(db.get_db)):
    """Menambahkan stream baru ke database."""
    return await db.create_stream(db_session, url=stream_data.url, roi_points=stream_data.roi_points, real_width_m=stream_data.real_width_m, real_height_m=stream_data.real_height_m)

@app.post("/start_processing/{stream_id}", tags=["Stream Management"])
async def start_processing_endpoint(stream_id: int, background_tasks: BackgroundTasks, db_session: AsyncSession = Depends(db.get_db)):
    """Memulai proses analisis video untuk stream_id tertentu."""
    stream = await db.get_stream(db_session, stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    await db.update_stream_status(db_session, stream_id, 'processing')
    background_tasks.add_task(start_cv_processing, stream.id, stream.url, stream.roi_points, stream.real_width_m, stream.real_height_m)
    
    return {"message": "Processing started", "stream_id": stream_id}

@app.get("/get_stream_info/{stream_id}", response_model=Stream, tags=["Data Retrieval"])
async def get_stream_info_endpoint(stream_id: int, db_session: AsyncSession = Depends(db.get_db)):
    """Mengambil informasi detail tentang sebuah stream."""
    stream = await db.get_stream(db_session, stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    return stream

@app.get("/get_events/{stream_id}", response_model=List[Event], tags=["Data Retrieval"])
async def get_events_endpoint(stream_id: int, db_session: AsyncSession = Depends(db.get_db)):
    """Mengambil histori event anomali untuk sebuah stream."""
    return await db.get_events_for_stream(db_session, stream_id)

@app.get("/get_stream_frame", tags=["Utilities"])
async def get_stream_frame(url: str = Query(..., min_length=10, description="URL dari video stream")):
    """Mengambil satu frame dari URL stream dan mengembalikannya sebagai gambar JPEG."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video stream from URL")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read a frame from the stream")

    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame to JPEG")

    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

# --- NEW: Video Streaming Endpoint ---
async def frame_generator(stream_id: int):
    """Generator yang mengambil frame dari dictionary dan mengirimkannya."""
    while True:
        if stream_id in processed_frames:
            frame = processed_frames[stream_id]
            # Kirim frame sebagai bagian dari stream MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        await asyncio.sleep(0.03) # Sesuaikan delay sesuai FPS

@app.get("/video_feed/{stream_id}", tags=["Streaming"])
async def video_feed(stream_id: int):
    """Endpoint untuk streaming video yang sudah diproses (MJPEG)."""
    return StreamingResponse(
        frame_generator(stream_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# --- WebSocket Endpoint ---
@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: int):
    """Endpoint WebSocket untuk update data anomali secara real-time ke dashboard."""
    await manager.connect(websocket, stream_id)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket, stream_id)

async def broadcast_event(stream_id: int, event_data: dict):
    await manager.broadcast(stream_id, event_data)

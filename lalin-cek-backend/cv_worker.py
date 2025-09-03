import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import ByteTrack, BotSort
import asyncio
from datetime import datetime
from typing import Dict
import time
import torch

from database import create_event, AsyncSessionLocal
from twilio_client import send_critical_alert
from utils import transform_pixel_to_real, calculate_speed_and_direction
import main

# --- Konfigurasi Optimasi ---
MODEL_PATH = 'accident.pt'  # Gunakan model terkecil untuk FPS maksimal
CONFIDENCE_THRESHOLD = 0.6  # Tingkatkan threshold untuk lebih cepat
DANGEROUS_PROXIMITY_THRESHOLD_M = 4.0
SUDDEN_STOP_THRESHOLD_KMH = 20.0

# Pilihan tracker: 'bytetrack' (CPU, cepat) atau 'botsort' (bisa GPU)
TRACKER_TYPE = 'bytetrack'

# CUDA Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
print(f"🚀 Using device: {DEVICE}")
if USE_CUDA:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA Memory: {props.total_memory / 1024**3:.1f} GB")
    # Optimasi backend untuk input video berukuran tetap (640)
    torch.backends.cudnn.benchmark = True

# --- Shared Memory for Frames ---
processed_frames: Dict[int, bytes] = {}

class OptimizedFPSCounter:
    def __init__(self):
        self.frame_times = []
        self.fps = 0
        self.last_update = time.time()
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 30 frames for smooth FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Update FPS every 0.5 seconds
        if current_time - self.last_update > 0.5 and len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff
            self.last_update = current_time
        
        return self.fps

class FastTrackingLine:
    def __init__(self):
        self.track_colors = {}
        self.track_trails = {}
        self.max_trail_length = 25  # Reduced untuk performa
        # Pre-generate colors untuk menghindari komputasi berulang
        self.color_palette = [
            (255, 61, 99),   # Pink terang
            (61, 255, 99),   # Hijau terang  
            (99, 61, 255),   # Ungu terang
            (255, 199, 61),  # Orange terang
            (61, 199, 255),  # Biru terang
            (199, 255, 61),  # Lime terang
            (255, 61, 199),  # Magenta terang
            (61, 255, 199),  # Cyan terang
            (199, 61, 255),  # Violet terang
            (255, 127, 61),  # Jingga terang
        ]
    
    def get_color(self, track_id):
        """Get unique color for track ID - optimized"""
        if track_id not in self.track_colors:
            self.track_colors[track_id] = self.color_palette[track_id % len(self.color_palette)]
        return self.track_colors[track_id]
    
    def add_point(self, track_id, center_point):
        """Add point to trail - optimized"""
        if track_id not in self.track_trails:
            self.track_trails[track_id] = []
        
        # Convert to int to avoid float operations
        point = (int(center_point[0]), int(center_point[1]))
        self.track_trails[track_id].append(point)
        
        # Limit trail length
        if len(self.track_trails[track_id]) > self.max_trail_length:
            self.track_trails[track_id].pop(0)
    
    def draw_trails(self, frame):
        """Draw trails with optimized rendering"""
        for track_id, trail in self.track_trails.items():
            if len(trail) < 2:
                continue
            
            color = self.get_color(track_id)
            
            # Simplified trail drawing - no alpha blending untuk performa
            for i in range(1, len(trail)):
                # Progressive thickness
                thickness = max(2, int(6 * (i / len(trail))))
                cv2.line(frame, trail[i-1], trail[i], color, thickness)
    
    def cleanup_old_tracks(self, active_track_ids):
        """Cleanup inactive tracks"""
        if len(self.track_trails) > 50:  # Only cleanup when too many tracks
            inactive_tracks = set(self.track_trails.keys()) - set(active_track_ids)
            for track_id in list(inactive_tracks)[:10]:  # Cleanup max 10 per frame
                self.track_trails.pop(track_id, None)
                self.track_colors.pop(track_id, None)

async def start_cv_processing(stream_id: int, stream_url: str, roi_points: dict, real_width_m: float, real_height_m: float):
    print(f"🎬 Starting CV processing for stream {stream_id}")
    
    # Optimized video capture
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video stream for stream ID {stream_id}")
        return
    
    # Aggressive video optimization
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Reduce resolution if too high
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Video: {width}x{height} @ {actual_fps} FPS")
    
    # Load and optimize model (GPU + FP16)
    print("🧠 Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    if USE_CUDA:
        model.to('cuda')
        # Aktifkan half precision (FP16) untuk percepatan di GPU Ampere
        try:
            model.model.half()
            print("✅ YOLO model on CUDA with FP16")
        except Exception as e:
            print(f"⚠️ Could not enable FP16: {e}")
        # Warm-up dengan tensor di GPU (FP16) agar kernel terinisialisasi
        dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float16, device='cuda')
        with torch.inference_mode():
            model.predict(dummy, verbose=False)
        print("✅ YOLO warm-up complete")
    else:
        print("⚠️ Running YOLO on CPU (expect lower FPS)")
    
    # Initialize optimized tracker
    print("🔍 Initializing tracker...")
    try:
        if TRACKER_TYPE == 'bytetrack':
            tracker = ByteTrack(
                track_thresh=0.3,    # Slightly higher for performance
                track_buffer=20,     # Reduced buffer
                match_thresh=0.8,
                frame_rate=30
            )
            print("✅ ByteTrack initialized (CPU)")
        else:
            tracker = BotSort(
                reid_weights=None,
                device=DEVICE if USE_CUDA else 'cpu',  # gunakan CUDA jika ada
                half=USE_CUDA,
                track_high_thresh=0.5,
                track_low_thresh=0.2,
                new_track_thresh=0.6,
                track_buffer=20,
                match_thresh=0.8,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,
                frame_rate=30
            )
            print(f"✅ BotSORT initialized ({'CUDA' if USE_CUDA else 'CPU'})")
    except Exception as e:
        print(f"⚠️ Tracker error: {e}, falling back to ByteTrack")
        tracker = ByteTrack(track_thresh=0.3, track_buffer=20, match_thresh=0.8, frame_rate=30)
    
    # Initialize optimized components
    fps_counter = OptimizedFPSCounter()
    tracking_lines = FastTrackingLine()
    
    # ROI setup
    pts_src = np.array([roi_points['bottom_left'], roi_points['bottom_right'], 
                       roi_points['top_right'], roi_points['top_left']], dtype=np.int32)
    pts_dst = np.array([[0, real_height_m], [real_width_m, real_height_m], 
                       [real_width_m, 0], [0, 0]], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)
    
    track_history = {}
    frame_count = 0
    
    print("🚀 Starting main processing loop...")
    
    async with AsyncSessionLocal() as db_session:
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print(f"🔄 Stream {stream_id} reconnecting...")
                await asyncio.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            frame_count += 1
            current_fps = fps_counter.update()
            
            # Fixed frame-skipping untuk stabilitas (proses deteksi tiap 2 frame)
            process_detection = (frame_count % 2 == 0)
            
            # ROI visualization
            cv2.polylines(frame, [pts_src], isClosed=True, color=(0, 255, 255), thickness=2)
            
            current_objects = []
            active_track_ids = []
            
            if process_detection:
                # YOLO inference on GPU (model sudah di-CUDA). Jangan override device di sini.
                with torch.inference_mode():
                    results = model.predict(
                        frame,
                        verbose=False,
                        conf=CONFIDENCE_THRESHOLD,
                        imgsz=640,
                        max_det=50
                    )[0]
                
                # Fast detection conversion
                detections = []
                if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy()
                    detections = np.column_stack([boxes, scores.reshape(-1, 1), classes.reshape(-1, 1)])
                else:
                    detections = np.empty((0, 6))
                
                # Update tracker
                tracks = tracker.update(detections, frame)
                
                # Process tracks
                if len(tracks) > 0:
                    for track in tracks:
                        try:
                            x1, y1, x2, y2, track_id, _, cls_id, _ = track
                            track_id, cls_id = int(track_id), int(cls_id)
                            active_track_ids.append(track_id)

                            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
                            
                            # ROI check
                            if cv2.pointPolygonTest(pts_src, center_point, False) < 0:
                                continue

                            # Add to tracking line
                            tracking_lines.add_point(track_id, center_point)

                            # Speed calculation
                            bottom_center_pixel = np.array([(x1 + x2) / 2, y2])
                            real_world_pos = transform_pixel_to_real(bottom_center_pixel, homography_matrix)
                            speed_mps, direction = calculate_speed_and_direction(track_id, real_world_pos, track_history)
                            speed_kmh = speed_mps * 3.6
                            
                            current_objects.append({
                                'id': track_id, 
                                'pos': real_world_pos, 
                                'type': model.names[cls_id],
                                'speed_kmh': speed_kmh,
                            })

                            # Drawing with track color
                            color = tracking_lines.get_color(track_id)
                            label = f"ID:{track_id} {model.names[cls_id]}"
                            speed_label = f"{speed_kmh:.1f} km/h"
                            
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, label, (int(x1), int(y1) - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, speed_label, (int(x1), int(y1) - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                            # Anomaly detection (simplified)
                            if frame_count % 10 == 0:  # Check every 10 frames to save CPU
                                history = track_history.get(track_id, {})
                                prev_speed = history.get('last_speed_kmh_avg', speed_kmh)
                                speed_drop = prev_speed - speed_kmh
                                
                                if speed_drop > SUDDEN_STOP_THRESHOLD_KMH and prev_speed > 5.0:
                                    event_data = {
                                        "stream_id": stream_id, "timestamp": datetime.utcnow(), 
                                        "event_type": "Sudden Stop", "object_type": model.names[cls_id], 
                                        "severity": "medium", "details": {"speed_drop_kmh": round(speed_drop, 2), "object_id": track_id}
                                    }
                                    # Non-blocking DB & broadcast
                                    asyncio.create_task(create_event(db_session, event_data))
                                    asyncio.create_task(main.broadcast_event(stream_id, event_data))
                                
                                history['last_speed_kmh_avg'] = (prev_speed * 0.9) + (speed_kmh * 0.1)
                                track_history[track_id] = history

                        except Exception as e:
                            print(f"⚠️ Track processing error: {e}")
                            continue

            # Draw tracking lines
            tracking_lines.draw_trails(frame)
            
            # Cleanup old tracks periodically
            if frame_count % 60 == 0:  # Every 60 frames
                tracking_lines.cleanup_old_tracks(active_track_ids)

            # Proximity detection (simplified, every 30 frames)
            if frame_count % 30 == 0 and len(current_objects) > 1:
                for i in range(len(current_objects)):
                    for j in range(i + 1, len(current_objects)):
                        obj1, obj2 = current_objects[i], current_objects[j]
                        distance = np.linalg.norm(obj1['pos'] - obj2['pos'])
                        
                        if distance < DANGEROUS_PROXIMITY_THRESHOLD_M:
                            is_critical = distance < 1.0
                            event_data = {
                                "stream_id": stream_id, "timestamp": datetime.utcnow(), 
                                "event_type": "Dangerous Proximity", "object_type": f"{obj1['type']} & {obj2['type']}", 
                                "severity": "critical" if is_critical else "medium",
                                "details": {"distance_m": round(distance, 2), "object_ids": [obj1['id'], obj2['id']]}
                            }
                            asyncio.create_task(create_event(db_session, event_data))
                            asyncio.create_task(main.broadcast_event(stream_id, event_data))
                            if is_critical:
                                # Twilio boleh tetap sinkron karena jarang
                                send_critical_alert(event_data)

            # Status display
            fps_color = (0, 255, 0) if current_fps > 20 else (0, 165, 255) if current_fps > 10 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
            cv2.putText(frame, f"CUDA: {'✅' if USE_CUDA else '❌'}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if USE_CUDA else (0, 0, 255), 2)
            cv2.putText(frame, f"Objects: {len(current_objects)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Tracks: {len(tracking_lines.track_trails)}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Optimized encoding (tetap di CPU, ini normal)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                processed_frames[stream_id] = buffer.tobytes()

            # Adaptive sleep based on performance
            processing_time = time.time() - start_time
            target_frame_time = 1.0 / 30.0  # Target 30 FPS
            sleep_time = max(0.001, target_frame_time - processing_time)
            await asyncio.sleep(sleep_time)

    cap.release()
    if stream_id in processed_frames:
        del processed_frames[stream_id]
    print(f"🛑 CV processing stopped for stream {stream_id}")

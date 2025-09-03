import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta

# FPS asumsi untuk kalkulasi kecepatan, bisa disesuaikan
VIDEO_FPS = 30.0

def transform_pixel_to_real(pixel_coords: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    """Mengubah koordinat piksel ke koordinat dunia nyata (meter)."""
    pixel_coords_homogeneous = np.array([*pixel_coords, 1])
    real_world_coords_homogeneous = homography_matrix @ pixel_coords_homogeneous
    # Normalisasi
    if real_world_coords_homogeneous[2] != 0:
        real_world_coords = real_world_coords_homogeneous[:2] / real_world_coords_homogeneous[2]
        return real_world_coords
    return np.array([0, 0])

def calculate_speed_and_direction(track_id: int, current_pos: np.ndarray, track_history: Dict) -> Tuple[float, np.ndarray]:
    """Menghitung kecepatan (m/s) dan vektor arah."""
    speed_mps = 0.0
    direction_vector = np.array([0, 0])
    
    current_time = datetime.utcnow()

    if track_id not in track_history:
        track_history[track_id] = {'positions': [], 'timestamps': []}
    
    history = track_history[track_id]
    history['positions'].append(current_pos)
    history['timestamps'].append(current_time)

    # Hanya simpan beberapa posisi terakhir untuk efisiensi
    if len(history['positions']) > 10:
        history['positions'].pop(0)
        history['timestamps'].pop(0)
    
    if len(history['positions']) > 1:
        prev_pos = history['positions'][-2]
        prev_time = history['timestamps'][-2]
        
        delta_pos = current_pos - prev_pos
        delta_time = (current_time - prev_time).total_seconds()
        
        if delta_time > 0:
            speed_mps = np.linalg.norm(delta_pos) / delta_time
            if np.linalg.norm(delta_pos) > 0:
                direction_vector = delta_pos / np.linalg.norm(delta_pos)

    # Simpan data terbaru
    history['last_speed_mps'] = speed_mps
    history['last_direction'] = direction_vector

    return speed_mps, direction_vector
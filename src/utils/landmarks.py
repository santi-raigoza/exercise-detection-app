# src/utils/landmarks.py
import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def normalize(landmarks):
    """
    landmarks: list of 33 objects with .x, .y, .z (MediaPipe NormalizedLandmark)
    Returns: np.ndarray of shape (99,) — translated to hip midpoint, scaled by torso height
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    hip_mid = (coords[LEFT_HIP] + coords[RIGHT_HIP]) / 2.0
    shoulder_mid = (coords[LEFT_SHOULDER] + coords[RIGHT_SHOULDER]) / 2.0
    torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_height < 1e-6:
        torso_height = 1.0
    return ((coords - hip_mid) / torso_height).flatten()


def build_window(frame_buffer):
    """
    frame_buffer: list or deque of 30 np.ndarray(99,)
    Returns: np.ndarray of shape (2970,) — frames concatenated in order
    """
    return np.concatenate(list(frame_buffer))

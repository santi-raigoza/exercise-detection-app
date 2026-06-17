import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from utils.landmarks import normalize, build_window, POSE_CONNECTIONS
from utils.rep_counter import RepCounter

WINDOW_SIZE = 30  # frames per classification window — must match training
MODEL_PATH  = Path(__file__).parent.parent / 'models' / 'model.pkl'
SCALER_PATH = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
TASK_PATH   = str(Path(__file__).parent.parent / 'pose_landmarker_lite.task')


def _draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)


def _draw_overlay(frame, exercise, confidence, count, fps):
    h, w, _ = frame.shape
    # Exercise label + confidence — top left
    cv2.putText(frame, f'{exercise}  {confidence:.0%}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    # Rep count — top right
    label = f'Reps: {count}'
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(frame, label, (w - tw - 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    # FPS — bottom left
    cv2.putText(frame, f'FPS {fps:.0f}', (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=RunningMode.IMAGE,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    rep_counter  = RepCounter()
    exercise, confidence = 'rest', 0.0
    prev_time = time.time()

    print('Press Q to quit.')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            _draw_skeleton(frame, landmarks)

            # Add normalized frame to buffer; classify once we have a full window
            frame_buffer.append(normalize(landmarks))
            if len(frame_buffer) == WINDOW_SIZE:
                window_s = scaler.transform([build_window(frame_buffer)])
                probs    = model.predict_proba(window_s)[0]
                idx      = int(np.argmax(probs))
                exercise   = model.classes_[idx]
                confidence = float(probs[idx])

            rep_counter.update(exercise, confidence, landmarks)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        _draw_overlay(frame, exercise, confidence, rep_counter.count, fps)
        cv2.imshow('Exercise Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == '__main__':
    main()

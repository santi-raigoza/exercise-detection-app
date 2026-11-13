import shutil
import subprocess
import tempfile
import urllib.request
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
PAUSE_GAP   = 2.0 # seconds between frames treated as a pause in recording
ROOT        = Path(__file__).parent.parent
MODEL_PATH  = ROOT / 'models' / 'model.pkl'
SCALER_PATH = ROOT / 'models' / 'scaler.pkl'
TASK_PATH   = ROOT / 'pose_landmarker_lite.task'
POSE_TASK_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
                 'pose_landmarker_lite/float16/latest/pose_landmarker_lite.task')


def ensure_pose_model():
    """Download the MediaPipe pose model if it isn't in the project root yet."""
    if not TASK_PATH.exists():
        print(f'Downloading pose model → {TASK_PATH}')
        tmp = TASK_PATH.with_suffix('.part')
        urllib.request.urlretrieve(POSE_TASK_URL, tmp)
        tmp.rename(TASK_PATH)
    return TASK_PATH


def load_classifier():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def create_landmarker():
    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(ensure_pose_model())),
        running_mode=RunningMode.IMAGE,
    )
    return PoseLandmarker.create_from_options(options)


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    s = max(w, h) / 640  # thicker dots and lines on high-resolution (incl. portrait) frames
    radius, thickness = max(4, round(4 * s)), max(2, round(2 * s))
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for pt in pts:
        cv2.circle(frame, pt, radius, (0, 255, 0), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), thickness)


def _put_label(frame, text, org, scale, color, thickness):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _format_elapsed(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f'{minutes:02d}:{secs:02d}'


def draw_overlay(frame, exercise, confidence, count, fps=None, elapsed=None):
    h, w, _ = frame.shape
    s = w / 640  # scale text with frame size so it looks the same at any resolution
    thickness = max(2, round(2 * s))

    # Exercise, confidence, and reps — top left, below the web UI's "Detection" label.
    # Shrink the font only if the line would run off the right edge.
    label = f'{exercise}  {confidence:.0%}  |  Reps: {count}'
    scale = 1.1 * s
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    scale *= min(1.0, (w - 30 * s) / tw)
    _put_label(frame, label, (int(15 * s), int(75 * s)), scale, (0, 255, 0), thickness)

    small = 0.7 * s
    # FPS — bottom left
    if fps is not None:
        _put_label(frame, f'FPS {fps:.0f}', (int(15 * s), h - int(15 * s)),
                   small, (200, 200, 200), max(1, thickness - 1))
    # Elapsed recording time — bottom right
    if elapsed is not None:
        text = _format_elapsed(elapsed)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, small, thickness)
        _put_label(frame, text, (w - tw - int(15 * s), h - int(15 * s)),
                   small, (255, 255, 255), thickness)


class ExerciseSession:
    """Per-user inference state: landmark buffer, rep counter, and a log of exercise segments."""

    def __init__(self, model, scaler):
        self.model  = model
        self.scaler = scaler
        self.frame_buffer = deque(maxlen=WINDOW_SIZE)
        self.rep_counter  = RepCounter()
        self.exercise, self.confidence = 'rest', 0.0
        self.segments = []      # closed segments: (exercise, reps, start_t, end_t)
        self._seg_start = None  # start time of the segment the rep counter is on
        self.elapsed = 0.0      # seconds of active recording
        self._last_t = None

    @property
    def count(self):
        return self.rep_counter.count

    def step(self, landmarks, t=0.0):
        """Feed one frame's landmarks; updates the prediction and rep count."""
        # Add normalized frame to buffer; classify once we have a full window
        self.frame_buffer.append(normalize(landmarks))
        if len(self.frame_buffer) == WINDOW_SIZE:
            window_s = self.scaler.transform([build_window(self.frame_buffer)])
            probs    = self.model.predict_proba(window_s)[0]
            idx      = int(np.argmax(probs))
            self.exercise   = str(self.model.classes_[idx])
            self.confidence = float(probs[idx])

        prev_exercise, prev_count = self.rep_counter.exercise, self.rep_counter.count
        self.rep_counter.update(self.exercise, self.confidence, landmarks)

        # RepCounter resets when the exercise changes — log the finished segment
        if self.rep_counter.exercise != prev_exercise:
            if prev_exercise is not None:
                self.segments.append((prev_exercise, prev_count, self._seg_start, t))
            self._seg_start = t

    def process(self, frame_bgr, landmarker, t=0.0, fps=None, detect_frame=None):
        """
        Detect pose on a BGR frame, update state, and draw the overlay in place.
        detect_frame: optional smaller copy of frame_bgr to run pose detection on (faster);
        landmarks are normalized, so they still line up when drawn on the full frame.
        """
        # Accumulate time between consecutive frames; a long gap means recording was
        # paused (e.g. webcam stopped and restarted), so it doesn't count toward elapsed.
        if self._last_t is not None and 0 < t - self._last_t < PAUSE_GAP:
            self.elapsed += t - self._last_t
        self._last_t = t

        rgb = cv2.cvtColor(frame_bgr if detect_frame is None else detect_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame_bgr, landmarks)
            self.step(landmarks, t)

        draw_overlay(frame_bgr, self.exercise, self.confidence, self.count, fps, self.elapsed)
        return frame_bgr

    def summary(self, end_t):
        """All segments, including the one still in progress."""
        segments = list(self.segments)
        if self.rep_counter.exercise is not None:
            segments.append((self.rep_counter.exercise, self.count, self._seg_start, end_t))
        return segments


def _fit(w, h, max_side, min_side=0):
    """
    Scale (w, h) so the longest side is within [min_side, max_side] (never upscaling
    unless below min_side); keep dimensions even for H.264.
    """
    k = min(max(1.0, min_side / max(w, h)), max_side / max(w, h))
    return int(w * k) // 2 * 2, int(h * k) // 2 * 2


class _VideoWriter:
    """
    Writes browser-playable H.264 in a single high-quality encode via ffmpeg, copying the
    audio from audio_src. Falls back to OpenCV's mp4v (no audio) if ffmpeg isn't installed.
    """

    def __init__(self, out_path, size, fps, audio_src, max_seconds):
        w, h = size
        self._cv_writer = self._proc = None
        if shutil.which('ffmpeg') is None:
            self._cv_writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            return
        self._proc = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
             '-t', str(max_seconds), '-i', str(audio_src),  # audio trimmed to the processed length
             '-map', '0:v', '-map', '1:a?',                 # '?': clips without audio still work
             '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
             '-c:a', 'aac', '-movflags', '+faststart', str(out_path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        if self._cv_writer is not None:
            self._cv_writer.write(frame)
        else:
            self._proc.stdin.write(frame.tobytes())

    def close(self):
        if self._cv_writer is not None:
            self._cv_writer.release()
            return
        _, err = self._proc.communicate()
        if self._proc.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {err.decode(errors="replace").strip()}')


def process_video(path, landmarker, model, scaler, max_seconds=30,
                  max_side=1920, min_side=1280, detect_side=640):
    """
    Annotate a video file, keeping its resolution and audio. The output's longest side is kept
    within [min_side, max_side]: low-resolution clips are upscaled first so the skeleton and
    text are drawn sharp instead of being blown up blurry by the player.
    Pose detection runs on the original frame downscaled to detect_side for speed.
    Processes at most max_seconds.
    Returns (annotated_mp4_path, segments) where segments are (exercise, reps, start_s, end_s).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    session = ExerciseSession(model, scaler)
    out_path = Path(tempfile.mkdtemp(prefix='exercise_')) / 'annotated.mp4'
    writer = None
    frame_idx = 0

    try:
        while frame_idx < max_seconds * fps:
            success, frame = cap.read()
            if not success:
                break
            h, w, _ = frame.shape
            out_size, det_size = _fit(w, h, max_side, min_side), _fit(w, h, detect_side)
            detect_frame = frame if det_size == (w, h) else \
                cv2.resize(frame, det_size, interpolation=cv2.INTER_AREA)
            if out_size != (w, h):
                upscaling = out_size[0] > w
                frame = cv2.resize(frame, out_size,
                                   interpolation=cv2.INTER_CUBIC if upscaling else cv2.INTER_AREA)
            if writer is None:
                writer = _VideoWriter(out_path, out_size, fps, path, max_seconds)

            session.process(frame, landmarker, t=frame_idx / fps, detect_frame=detect_frame)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError('The video contains no readable frames.')
    return out_path, session.summary(frame_idx / fps)

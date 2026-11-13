import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipeline import WINDOW_SIZE, ExerciseSession, create_landmarker, load_classifier, process_video


def _pose(knee_bend=0.0):
    """33 synthetic landmarks standing upright; knee_bend moves the knees forward."""
    lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]
    for i, (x, y) in {11: (0.45, 0.3), 12: (0.55, 0.3), 23: (0.45, 0.55), 24: (0.55, 0.55),
                      25: (0.45 + knee_bend, 0.7), 26: (0.55 + knee_bend, 0.7),
                      27: (0.45, 0.9), 28: (0.55, 0.9)}.items():
        lms[i].x, lms[i].y = x, y
    return lms


class _FakeModel:
    """Predicts whatever class `current` is set to, with full confidence."""
    classes_ = np.array(['pushup', 'rest', 'squat'])

    def __init__(self):
        self.current = 'squat'

    def predict_proba(self, X):
        return np.array([[float(c == self.current) for c in self.classes_]])


class _IdentityScaler:
    def transform(self, X):
        return np.asarray(X)


def test_session_predicts_after_full_window():
    model, scaler = load_classifier()
    session = ExerciseSession(model, scaler)
    for _ in range(WINDOW_SIZE - 1):
        session.step(_pose())
    assert session.confidence == 0.0  # no prediction until the buffer is full

    session.step(_pose())
    assert len(session.frame_buffer) == WINDOW_SIZE
    assert session.exercise in model.classes_
    assert 0.0 < session.confidence <= 1.0


def test_session_counts_reps_and_logs_segments():
    model = _FakeModel()
    session = ExerciseSession(model, _IdentityScaler())
    t = 0
    for _ in range(WINDOW_SIZE):
        t += 1
        session.step(_pose(), t)
    for _ in range(2):  # two squats: bend knees deep, then stand up
        for bend in [0.4] * 10 + [0.0] * 10:
            t += 1
            session.step(_pose(bend), t)
    assert session.count == 2

    model.current = 'pushup'
    t += 1
    session.step(_pose(), t)
    assert session.segments == [('squat', 2, WINDOW_SIZE, t)]
    assert session.summary(t + 5)[-1] == ('pushup', 0, t, t + 5)


def _probe(path, stream, field):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', stream,
                          '-show_entries', f'stream={field}', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='needs ffmpeg')
@pytest.mark.parametrize('in_size, out_size', [
    ('1280x720', '1280,720'),  # already high-res: kept as-is
    ('320x180', '1280,720'),   # low-res: upscaled so the overlay is drawn sharp
    ('180x320', '720,1280'),   # low-res portrait
])
def test_process_video_output_size_and_audio(tmp_path, in_size, out_size):
    # 3 s of black video with a tone — no person, so no segments
    src = tmp_path / 'blank.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                    '-f', 'lavfi', '-i', f'color=c=black:s={in_size}:r=10:d=3',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(src)],
                   check=True)

    model, scaler = load_classifier()
    landmarker = create_landmarker()
    out_path, segments = process_video(src, landmarker, model, scaler, max_seconds=2)
    landmarker.close()

    assert segments == []
    assert _probe(out_path, 'v:0', 'codec_name,width,height') == f'h264,{out_size}'
    assert _probe(out_path, 'a:0', 'codec_name') == 'aac'
    duration = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                     '-of', 'csv=p=0', str(out_path)],
                                    capture_output=True, text=True).stdout)
    assert duration == pytest.approx(2, abs=0.2)  # capped at max_seconds, audio trimmed to match

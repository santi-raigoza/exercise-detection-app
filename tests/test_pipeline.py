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


def test_rest_closes_segment_and_resume_counts_only_new_reps():
    model = _FakeModel()
    session = ExerciseSession(model, _IdentityScaler())
    for t in range(WINDOW_SIZE):
        session.step(_pose(), t)
    for t, bend in enumerate([0.4] * 10 + [0.0] * 10, WINDOW_SIZE):
        session.step(_pose(bend), t)
    model.current = 'rest'
    session.step(_pose(), 50)
    assert session.summary(70) == [('squat', 1, 29, 50)]
    assert session.count == 1  # retained counter remains compatible with live overlay
    model.current = 'squat'
    session.step(_pose(), 70)
    for t, bend in enumerate([0.4] * 10 + [0.0] * 10, 71):
        session.step(_pose(bend), t)
    assert session.summary(91) == [('squat', 1, 29, 50), ('squat', 1, 70, 91)]
    session.exercise, session.confidence = 'rest', 0.5
    session.frame_buffer.clear()  # keep the low confidence prediction for this step
    session.step(_pose(), 92)
    assert session.summary(93)[-1] == ('squat', 1, 70, 93)


def test_missing_ffmpeg_fails_before_opening_video(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline.shutil, 'which', lambda _: None)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: pytest.fail('opened video'))
    with pytest.raises(RuntimeError, match='requires ffmpeg'):
        process_video('unused', None, None, None)


@pytest.mark.parametrize('failure', ['process', 'close', 'empty'])
def test_failed_processing_removes_output(tmp_path, monkeypatch, failure):
    import pipeline
    output = tmp_path / 'exercise_test'
    output.mkdir()
    (output / 'annotated.mp4').write_bytes(b'partial')
    monkeypatch.setattr(pipeline, '_require_ffmpeg', lambda: None)
    monkeypatch.setattr(pipeline.tempfile, 'mkdtemp', lambda **_: str(output))
    cap = SimpleNamespace(isOpened=lambda: True, get=lambda _: 10,
                          read=lambda: (False, None) if failure == 'empty' else
                          (True, np.zeros((10, 10, 3), dtype=np.uint8)),
                          release=lambda: None)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: cap)
    def fail(*args, **kwargs):
        raise RuntimeError('failure')
    monkeypatch.setattr(ExerciseSession, 'process', fail if failure == 'process' else lambda *a, **k: None)
    monkeypatch.setattr(pipeline, '_VideoWriter', lambda *a: SimpleNamespace(
        write=lambda _: None, close=fail if failure == 'close' else lambda: None))
    with pytest.raises((RuntimeError, ValueError)):
        process_video('unused', None, None, None, max_seconds=0.1, min_side=0)
    assert not output.exists()

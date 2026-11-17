import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def web(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, 'load_classifier', lambda: (None, None))
    monkeypatch.setattr(pipeline, 'create_landmarker', lambda: None)
    return importlib.import_module('web_app')


def test_upload_copies_before_removing_source(web, tmp_path, monkeypatch):
    source = tmp_path / 'exercise_test'
    source.mkdir()
    video = source / 'annotated.mp4'
    video.write_bytes(b'synthetic video')
    monkeypatch.setattr(web, 'process_video', lambda *a, **kw: (video, []))
    cached, rows = web.analyze_video('input.mp4')
    assert Path(cached).read_bytes() == b'synthetic video'
    assert rows == []
    assert not source.exists()


def test_upload_copy_failure_removes_source(web, tmp_path, monkeypatch):
    source = tmp_path / 'exercise_test'
    source.mkdir()
    monkeypatch.setattr(web, 'process_video', lambda *a, **kw: (source / 'annotated.mp4', []))
    def fail(*a):
        raise OSError('copy failed')
    monkeypatch.setattr(web.video_out, 'move_resource_to_block_cache', fail)
    with pytest.raises(OSError, match='copy failed'):
        web.analyze_video('input.mp4')
    assert not source.exists()


def test_reset_discards_inflight_frame_and_preserves_holder(web, monkeypatch):
    entered, release, resetting = threading.Event(), threading.Event(), threading.Event()
    holder = web._SessionHolder()
    other = web._SessionHolder()
    def process(*a, **kw):
        entered.set()
        assert release.wait(5)
    holder.session = SimpleNamespace(process=process, count=4, frame_buffer=[])
    with ThreadPoolExecutor(max_workers=2) as pool:
        frame = pool.submit(web.stream_frame, np.zeros((10, 10, 3), dtype=np.uint8), holder)
        assert entered.wait(5)
        def reset():
            resetting.set()
            return web.reset_session(holder)
        reset_result = pool.submit(reset)
        assert resetting.wait(5)
        assert '>0<' in reset_result.result(timeout=5)
        assert holder.session is None
        release.set()
        assert frame.result(timeout=5) == (web.gr.skip(), web.gr.skip())
    assert holder.session is None
    assert other.session is None
    # Frame completions never write Gradio State, so they cannot restore it.
    event = next(fn for fn in web.demo.fns.values() if fn.fn is web.stream_frame)
    assert web.session_state not in event.outputs
    assert event.concurrency_limit == 4

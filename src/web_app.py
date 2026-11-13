import base64
import html
import threading
import time
from pathlib import Path

import cv2
import gradio as gr

from pipeline import WINDOW_SIZE, ExerciseSession, create_landmarker, load_classifier, process_video

MAX_UPLOAD_SECONDS = 30
GUIDE_DIR = Path(__file__).parent.parent / 'assets' / 'guide'
CSS = (Path(__file__).parent / 'web_app.css').read_text()
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&display=swap">')

EXERCISE_NAMES = {
    'pushup': 'Pushup',
    'squat': 'Squat',
    'pullup': 'Pullup',
    'jumping_jack': 'Jumping jack',
    'rest': 'Resting',
}

# (exercise, where to stand, alt text for the example photo)
GUIDE = [
    ('pushup', 'Side on to the camera', 'Person holding a pushup plank, filmed from the side'),
    ('squat', 'Side on to the camera', 'Person at the bottom of a squat, filmed from the side'),
    ('pullup', 'Facing toward or away from the camera', 'Person hanging from a pullup bar, facing away from the camera'),
    ('jumping_jack', 'Facing the camera', 'Person mid jumping jack with arms raised, facing the camera'),
]


def _pretty(exercise):
    return EXERCISE_NAMES.get(exercise, exercise.replace('_', ' ').capitalize())


def _image_data_uri(path):
    return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()


def _intro_html():
    cards = ''.join(
        f'<figure class="ed-guide-card">'
        f'<img src="{_image_data_uri(GUIDE_DIR / f"{exercise}.jpg")}" alt="{html.escape(alt)}" loading="lazy">'
        f'<figcaption><strong>{_pretty(exercise)}</strong><span>{html.escape(position)}</span></figcaption>'
        f'</figure>'
        for exercise, position, alt in GUIDE
    )
    return f'''
    <header class="ed-hero">
      <h1>Exercise Detection</h1>
      <p>Work out in front of your camera or upload a clip. It recognizes pushups, squats,
      pullups and jumping jacks, and counts your reps as you go.</p>
    </header>
    <section class="ed-guide" aria-labelledby="ed-guide-title">
      <div class="ed-guide-head">
        <h2 id="ed-guide-title">Set up your camera</h2>
        <p>Keep your whole body in frame. Counting starts about a second after you appear.</p>
      </div>
      <div class="ed-guide-grid">{cards}</div>
    </section>
    '''


class _LockedLandmarker:
    """One shared MediaPipe landmarker, serialized across concurrent users."""

    def __init__(self, landmarker):
        self._landmarker = landmarker
        self._lock = threading.Lock()

    def detect(self, mp_image):
        with self._lock:
            return self._landmarker.detect(mp_image)


MODEL, SCALER = load_classifier()
LANDMARKER = _LockedLandmarker(create_landmarker())


def _stats(session):
    if session is None:
        exercise, confidence, reps = 'Not started', '–', 0
    elif len(session.frame_buffer) < WINDOW_SIZE:
        exercise, confidence, reps = 'Getting ready', '–', session.count
    else:
        exercise, confidence, reps = _pretty(session.exercise), f'{session.confidence:.0%}', session.count
    return f'''
    <div class="ed-stats">
      <div class="ed-stat"><p class="ed-stat-label">Exercise</p><p class="ed-stat-value">{exercise}</p></div>
      <div class="ed-stat"><p class="ed-stat-label">Confidence</p><p class="ed-stat-value">{confidence}</p></div>
      <div class="ed-stat ed-stat-reps"><p class="ed-stat-label">Reps</p><p class="ed-stat-value">{reps}</p></div>
    </div>
    '''


def stream_frame(frame_rgb, session):
    if frame_rgb is None:
        return None, session, _stats(session)
    if session is None:
        session = ExerciseSession(MODEL, SCALER)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    session.process(frame_bgr, LANDMARKER, t=time.time())
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), session, _stats(session)


def reset_session():
    return None, _stats(None)


def analyze_video(video_path):
    if not video_path:
        raise gr.Error('Choose a video to analyze first.')
    try:
        out_path, segments = process_video(video_path, LANDMARKER, MODEL, SCALER,
                                           max_seconds=MAX_UPLOAD_SECONDS)
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e))
    rows = [[_pretty(exercise), reps, round(start, 1), round(end, 1)]
            for exercise, reps, start, end in segments]
    return str(out_path), rows


# Pine (primary) and sage (neutral) scales replace Gradio's default blue and grey, so built-in
# pieces like the webcam record button, progress bars, and loaders match the palette.
PINE = gr.themes.Color(
    name='pine', c50='#EEF3F0', c100='#D9E5DE', c200='#B6CCBF', c300='#8FAF9D', c400='#6A907D',
    c500='#4B7362', c600='#365B4E', c700='#2B4C42', c800='#22413A', c900='#1A332D', c950='#11231F',
)
SAGE = gr.themes.Color(
    name='sage', c50='#F8F9F5', c100='#EFF2EB', c200='#E1E8DC', c300='#D5DDD0', c400='#A9B6AB',
    c500='#7C8D82', c600='#62786D', c700='#4A5D54', c800='#2A3B35', c900='#1A2925', c950='#121D1A',
)

THEME = gr.themes.Base(
    primary_hue=PINE,
    secondary_hue=PINE,
    neutral_hue=SAGE,
    font=[gr.themes.GoogleFont('Figtree'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    body_background_fill='#ECEFE7',
    body_background_fill_dark='#121D1A',
    body_text_color='#22413A',
    body_text_color_dark='#E3EBE5',
    body_text_color_subdued='#62786D',
    body_text_color_subdued_dark='#93A79C',
    background_fill_primary='#F8F9F5',
    background_fill_primary_dark='#1A2925',
    background_fill_secondary='#EFF2EB',
    background_fill_secondary_dark='#15231F',
    border_color_primary='#D5DDD0',
    border_color_primary_dark='#2A3B35',
    block_background_fill='#F8F9F5',
    block_background_fill_dark='#1A2925',
    block_border_color='#D5DDD0',
    block_border_color_dark='#2A3B35',
    block_radius='22px',
    block_shadow='0 1px 2px rgba(34, 65, 58, 0.05), 0 12px 32px -20px rgba(34, 65, 58, 0.35)',
    block_shadow_dark='none',
    block_label_background_fill='#F8F9F5',
    block_label_background_fill_dark='#1A2925',
    block_label_text_color='#62786D',
    block_label_text_color_dark='#93A79C',
    block_label_border_color='#D5DDD0',
    block_label_border_color_dark='#2A3B35',
    button_primary_background_fill='#22413A',
    button_primary_background_fill_hover='#2E554B',
    button_primary_text_color='#F4F2E8',
    button_primary_background_fill_dark='#CFE0D5',
    button_primary_background_fill_hover_dark='#E1ECE5',
    button_primary_text_color_dark='#13201C',
    button_primary_border_color='transparent',
    button_primary_border_color_dark='transparent',
    button_secondary_background_fill='#E1E8DC',
    button_secondary_background_fill_hover='#D5DFCF',
    button_secondary_text_color='#22413A',
    button_secondary_background_fill_dark='#233631',
    button_secondary_background_fill_hover_dark='#2C433C',
    button_secondary_text_color_dark='#E3EBE5',
    button_secondary_border_color='transparent',
    button_secondary_border_color_dark='transparent',
    button_large_radius='999px',
    button_medium_radius='999px',
    button_small_radius='999px',
    color_accent='#22413A',
    color_accent_soft='#E1E8DC',
    color_accent_soft_dark='#233631',
    loader_color='#22413A',
    loader_color_dark='#CFE0D5',
    input_background_fill='#F8F9F5',
    input_background_fill_dark='#1A2925',
    table_border_color='#D5DDD0',
    table_border_color_dark='#2A3B35',
    table_even_background_fill='#F8F9F5',
    table_even_background_fill_dark='#1A2925',
    table_odd_background_fill='#EFF2EB',
    table_odd_background_fill_dark='#15231F',
)


with gr.Blocks(title='Exercise Detection') as demo:
    gr.HTML(_intro_html())

    with gr.Tabs(elem_classes='ed-tabs'):
        with gr.Tab('Live webcam'):
            session_state = gr.State(None)
            with gr.Row(equal_height=True):
                cam = gr.Image(sources=['webcam'], streaming=True, label='Camera')
                annotated = gr.Image(label='Detection', interactive=False)
            stats = gr.HTML(_stats(None))
            reset_btn = gr.Button('Reset reps', variant='secondary', elem_classes='ed-inline-btn')

            # show_progress='hidden' stops Gradio's per-frame loading timer/fade from
            # flickering over the outputs on every streamed frame
            cam.stream(stream_frame, inputs=[cam, session_state],
                       outputs=[annotated, session_state, stats],
                       stream_every=0.1, time_limit=600, concurrency_limit=4,
                       show_progress='hidden')
            reset_btn.click(reset_session, outputs=[session_state, stats])

        with gr.Tab('Upload video'):
            gr.HTML(f'<p class="ed-note">Only the first {MAX_UPLOAD_SECONDS} seconds of a video '
                    f'are analyzed.</p>')
            with gr.Row(equal_height=True):
                video_in = gr.Video(sources=['upload'], label='Your video')
                video_out = gr.Video(label='Annotated')
            analyze_btn = gr.Button('Analyze video', variant='primary', elem_classes='ed-inline-btn')
            gr.HTML('<h3 class="ed-subhead">Exercises found</h3>')
            table = gr.Dataframe(headers=['Exercise', 'Reps', 'Start (s)', 'End (s)'],
                                 show_label=False, interactive=False, elem_classes='ed-table')

            analyze_btn.click(analyze_video, inputs=video_in, outputs=[video_out, table])


if __name__ == '__main__':
    demo.launch(theme=THEME, css=CSS, head=HEAD)

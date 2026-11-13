# ExerciseDetection

Real-time exercise classification and rep counting from webcam video using pose estimation and a trained neural network.

## What it does

- Detects your body pose every frame using MediaPipe
- Classifies what exercise you're doing (pushup, squat, pullup, jumping jack, or rest) in real time
- Counts reps automatically using a per-exercise angle-based state machine
- Displays exercise label, confidence %, rep count, and FPS as an overlay on the webcam feed
- Runs as a web app (Gradio) with a live webcam mode and a video-upload mode

## How it works

The pipeline has three phases:

```
collect.py  →  train.py  →  app.py (CLI)  /  web_app.py (browser)
```

**collect.py** — records labeled training data. Opens your webcam (or a video file), runs pose detection on each frame, and saves 30-frame sliding windows of normalized landmarks to CSV files.

**train.py** — trains the classifier offline. Loads all CSVs, fits a `StandardScaler` + `MLPClassifier` (256→128 hidden layers), evaluates on a held-out 20% test set, and saves `model.pkl` + `scaler.pkl` to `models/`.

**pipeline.py** — shared inference core used by both apps: loads the model, runs pose detection, classifies 30-frame windows, counts reps, and draws the overlay.

**web_app.py** — Gradio web app. Live webcam tab streams frames from the browser; Upload tab annotates a video and returns a per-exercise rep table.

**app.py** — runs live inference. Opens your webcam, fills a 30-frame buffer of landmarks, classifies the buffer every frame, and feeds the prediction into the rep counter.

## Exercises supported

| Exercise | Camera orientation |
|---|---|
| Pushup | Side profile |
| Squat | Side profile |
| Pullup | Front or back facing |
| Jumping jack | Facing forward |
| Rest | Any |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: install `ffmpeg` (e.g. `brew install ffmpeg`) so annotated videos from the web app's Upload tab keep their audio and play in every browser. Without it, videos are saved without audio.

The MediaPipe pose model (`pose_landmarker_lite.task`) is downloaded to the project root automatically the first time any script runs.

## Usage

### 1. Collect training data

Run once per exercise label. Press **SPACE** to start/stop recording, **Q** to quit and save.

```bash
python src/collect.py --exercise pushup
python src/collect.py --exercise squat
python src/collect.py --exercise pullup
python src/collect.py --exercise jumping_jack
python src/collect.py --exercise rest
```

Or collect from a video file:

```bash
python src/collect.py --exercise pushup --video data/videos/pushup.mp4
python src/collect.py --exercise pullup --video data/videos/pullup.mp4 --start 60 --end 90
```

Collected windows are saved to `data/raw/` as timestamped CSVs.

### 2. Train the model

```bash
python src/train.py
```

Prints a per-class classification report and confusion matrix. Results are also saved to `models/report.txt`. Target accuracy: >85%.

### 3. Run the live app

```bash
python src/app.py
```

Stand in front of your webcam. The overlay shows:
- **Top left** — predicted exercise + confidence score (how certain the model is)
- **Top right** — rep count
- **Bottom left** — FPS

Press **Q** to quit.

### 4. Run the web app

```bash
python src/web_app.py
```

Open http://127.0.0.1:7860. Two tabs:
- **Live webcam** — allow camera access; the annotated feed, exercise, confidence, and rep count update live. **Reset reps** clears the count.
- **Upload video** — upload a clip (first 30 s are analyzed) to get an annotated video and a table of exercise segments with rep counts.

Each browser session has its own rep counter, so several people can use it at once.

## Retrain

The CLI tools and the web app share the same model files (`models/model.pkl`, `models/scaler.pkl`), so retraining updates both:

1. Collect data: `python src/collect.py --exercise <name>`
2. Retrain: `python src/train.py`
3. Check it: `python src/web_app.py`

New exercise classes will be recognized after retraining, but rep counting also needs a rule in `src/utils/rep_counter.py`.

## Tests

```bash
pytest
```

## Project structure

```
ExerciseDetectionProgram/
├── src/
│   ├── utils/
│   │   ├── landmarks.py      # Pose normalization and window building
│   │   └── rep_counter.py    # Per-exercise rep counting state machine
│   ├── collect.py            # Data collection script
│   ├── train.py              # Offline training pipeline
│   ├── pipeline.py           # Shared inference core (pose → classify → reps → overlay)
│   ├── app.py                # Live inference app (OpenCV window)
│   ├── web_app.py            # Gradio web app (webcam + upload)
│   └── web_app.css           # Web app styling (palette, type, layout)
├── assets/guide/             # Camera-setup example photos shown in the web app
├── tests/                    # pytest suite
├── data/raw/                 # Training CSVs
├── data/videos/              # Source videos used for data collection
├── models/                   # Saved model + scaler
├── requirements.txt          # Python dependencies
└── README.md
```

## Tech stack

| Component | Library |
|---|---|
| Pose detection | MediaPipe Tasks API |
| Computer vision | OpenCV |
| Web UI | Gradio |
| Classifier | scikit-learn MLPClassifier |
| Feature scaling | scikit-learn StandardScaler |
| Numerical ops | NumPy |
| Data handling | pandas |
| Model persistence | joblib |
| Language | Python 3.13 |

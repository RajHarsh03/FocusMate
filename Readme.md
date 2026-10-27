# FocusMate

**AI-Powered Focus & Productivity Monitor**

FocusMate is a webcam-based study companion that watches for signs of fatigue and lost attention, then adapts your study session instead of forcing a fixed Pomodoro timer. All processing runs locally on your machine, so no video ever leaves your laptop.

> Final Year Project for **Term Paper and Seminar (CBS3293)**, Department of Computer Science and Business Systems, Heritage Institute of Technology, Kolkata (HITK). Academic year 2025-26.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Database Design](#database-design)
- [Project Status and Roadmap](#project-status-and-roadmap)
- [Known Limitations](#known-limitations)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Overview

Most study timers are blind: they ring after 25 minutes whether you are fully focused or already nodding off. FocusMate looks at your face through a normal webcam, estimates how alert you are in real time, and uses that to decide when you should keep going, get a warning, or take a break.

**The idea in one line:** a study timer that responds to *you*, not to the clock.

---

## Key Features

- **Real-time face tracking** using MediaPipe Face Landmarker (Tasks API) and OpenCV
- **Eye Aspect Ratio (EAR)** calculation with blink detection to spot drowsiness
- **Head-pose and gaze deviation** estimation to detect when you look away from the screen
- **Focus Score** that combines EAR and head pose, smoothed with an exponential moving average so it does not jump around
- **Session state machine:** Active, Warning, Break Prompt, Ended
- **Adaptive session length** with break reminders based on your actual focus, not a fixed timer
- **Streamlit dashboard** with a live timer, focus meter, and start/stop controls
- **SQLite session logging** for sessions, focus readings, and break events
- **Analytics:** end-of-session summary, focus-over-time chart, focus heatmap, and session history
- **Privacy first:** everything is processed locally, nothing is uploaded

---

## How It Works

```
Webcam  ->  OpenCV frame capture  ->  MediaPipe Face Landmarker
                                              |
                                              v
                              EAR + Head-pose feature pipeline
                                              |
                                              v
                          Focus Score (smoothed with EMA)
                                              |
                                              v
                   Session state machine (Active / Warning / Break Prompt / Ended)
                                              |
                          +-------------------+-------------------+
                          v                                       v
                 Streamlit dashboard                      SQLite session logs
                 (timer, live meter, alerts)       (analytics, history, heatmap)
```

### Eye Aspect Ratio (EAR)

EAR compares the height of the eye to its width using six landmark points around each eye. When your eyes are open the value stays fairly steady, and it drops sharply when they close. A run of low values means a blink; a long run means drowsiness.

### Focus Score

The Focus Score blends two signals: how open your eyes are (EAR) and how far your head has turned away from the screen. An exponential moving average smooths the result so a single blink or quick glance does not trigger a false alarm.

### Session States

| State | Meaning |
| --- | --- |
| **Active** | You are focused, session continues normally |
| **Warning** | Focus has dropped below the warning threshold |
| **Break Prompt** | Focus stayed low, so a break is suggested |
| **Ended** | Session finished and a summary is saved |

---

## Tech Stack

| Area | Tools |
|:---|:---|
| Language | Python 3.10+ |
| Computer Vision | OpenCV, MediaPipe Face Landmarker (Tasks API) |
| Dashboard | Streamlit |
| Storage | SQLite |
| Analytics | Matplotlib |

---

## Project Structure

> Adjust this to match your actual folders as the project evolves.

```text
FocusMate/
├── app.py                  # Streamlit entry point
├── vision/
│   ├── capture.py          # Webcam frame capture (OpenCV)
│   ├── landmarks.py        # MediaPipe Face Landmarker wrapper
│   ├── ear.py              # EAR calculation and blink detection
│   ├── head_pose.py        # Head direction / gaze deviation
│   └── pipeline.py         # Combines features into one clean output
├── scoring/
│   ├── focus_score.py      # Focus Score formula with EMA smoothing
│   └── session_state.py    # Session state machine and timer
├── storage/
│   ├── db.py               # SQLite schema and read/write functions
│   └── focusmate.db        # Local database (created at runtime)
├── analytics/
│   ├── summary.py          # End-of-session summary
│   └── charts.py           # Focus-over-time chart, heatmap
├── models/
│   └── face_landmarker.task  # MediaPipe model file
├── tests/
├── docs/                   # Report, diagrams, screenshots
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or newer (tested with Python 3.11.9). Check with `python --version`
- A working webcam
- Good, even lighting for best results

### Installation

Run every command in PowerShell from the project root folder (the folder that contains `requirements.txt`).

```powershell
# 1. Clone the repository
git clone https://github.com/RajHarsh03/FocusMate.git
cd FocusMate

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

- If activation says "running scripts is disabled", run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window and activate again. This only affects the current window.
- Next time you only need `.venv\Scripts\activate`.
- Only Harsh edits `requirements.txt`. If you need a new library, ask Harsh. Everyone else pulls and runs `pip install -r requirements.txt` again.

### `requirements.txt`

The five libraries are pinned to versions that were installed and tested together (Python 3.11.9, Windows):

```text
opencv-python==5.0.0.93
mediapipe==1.1.1
streamlit==1.66.1
numpy==2.4.6
matplotlib==3.11.2
```

### MediaPipe model file

The model is not installed by pip. Download `face_landmarker.task` from the [MediaPipe Face Landmarker page](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) and save it as `models/face_landmarker.task`. Or run:

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile models\face_landmarker.task
```

Use only the MediaPipe Tasks API (`FaceLandmarker`). The old Face Mesh API is deprecated and must not be used.

### Always start from the project root

The database path is relative (`data/focusmate.db`). If you start a script from another folder, a second, empty database is created there.

### Storage demo data and tests

```powershell
# Fill data/focusmate.db with 6 demo sessions (user name "demo") from the last 6 days.
# Running it again replaces them. Real sessions are never touched.
python src/storage/seed_demo_data.py

# Storage tests (they use a temporary database, your real data is safe)
python src/storage/test_database.py
python src/storage/test_storage_extras.py
```

## Usage

```powershell
streamlit run app.py
```

1. Allow webcam access when prompted.
2. Click **Start** to begin a study session.
3. Watch the live focus meter and timer on the dashboard.
4. Take a break when FocusMate suggests one.
5. Click **Stop** to end the session and view your summary.

---

## Database Design

FocusMate stores everything in a local SQLite database with three main tables:

| Table | Purpose |
|:---|:---|
| `sessions` | One row per study session (start time, end time, summary) |
| `focus_readings` | Time-stamped Focus Score readings logged during a session |
| `break_events` | Breaks that were prompted, taken, or skipped |

---

## Project Status and Roadmap

FocusMate is under active development. The original eight-week plan covers the foundation, and development continues until April 2027.

- [x] Environment, repository, and folder structure
- [ ] Webcam capture and MediaPipe landmark extraction
- [ ] EAR calculation and blink detection
- [ ] Head-pose / gaze deviation estimation
- [ ] Streamlit dashboard shell with timer and controls
- [x] SQLite schema with basic read/write functions
- [ ] Focus Score formula with EMA smoothing
- [ ] Full session state machine (Warning, Break Prompt)
- [ ] Adaptive session length and break reminders
- [ ] Focus-over-time graphs and focus heatmap
- [ ] Session history view
- [ ] Testing across users, lighting conditions, and devices
- [ ] Final report and documentation

> Update the checkboxes as features are completed.

---

## Known Limitations

- Accuracy drops in poor lighting or when the face is partly covered
- Glasses can interfere with eye landmark detection
- Thresholds are being tuned and may need per-user calibration
- Works with a single face in frame

---

## Team

| Name | Focus Area |
|:---|:---|
| Qaem Hussain Rizvi | Vision module: capture, landmarks, head pose, pipeline |
| Kumari Muskan | EAR and blink detection, Focus Score, session state machine |
| Ayush Kumar | Streamlit dashboard and UI/UX |
| Harsh Raj | SQLite storage, session summary, analytics |

---

## Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe) by Google for the Face Landmarker
- [OpenCV](https://opencv.org/) for video capture and image processing
- [Streamlit](https://streamlit.io/) for the dashboard framework
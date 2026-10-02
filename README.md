# AI Exam Proctor

A laptop-only AI system that watches a student through the webcam during an online test and flags suspicious behaviour. Every violation is logged with a timestamp and a snapshot as evidence.

## Features
- **No face detected:** student left the frame or covered the camera
- **Multiple faces:** another person is present
- **Looking away:** head turned away from the screen for more than 2 seconds
- **Phone detected:** a mobile phone is visible (YOLOv8)
- **Evidence log:** `events.csv` plus snapshots in the `evidence/` folder
- **Auto calibration:** learns each student's normal head position first

## How it works
```
Webcam -> MediaPipe Face Mesh (faces + head direction)
       -> YOLOv8 (phone detection)
       -> Rule engine (behaviour must last 2 s to count)
       -> Log + snapshot + on-screen status
```

## Tech stack
Python, OpenCV, MediaPipe, Ultralytics YOLOv8, NumPy

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
python proctor.py
```
Look straight at the screen for about 2 seconds when it starts (calibration). Press `q` to quit, `c` to recalibrate.

## Tuning
Edit the settings at the top of `proctor.py` (`YAW_LIMIT`, `PITCH_LIMIT`, `HOLD_SECONDS`, etc.) to make detection stricter or more tolerant.

## Limitations
- Head direction is estimated from face landmarks, not true eye gaze
- Works best with good lighting and a front-facing camera
- Phone detection can miss phones that are small, covered, or far away

## Future work
- Eye gaze tracking
- Audio detection (talking or whispering)
- Tab-switch detection
- Web dashboard for exam reports

## Note
Use only with the consent of the person being monitored.

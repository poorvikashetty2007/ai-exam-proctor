"""
AI Exam Proctor v3 - strict mode with sound alerts.

Detects:
  NO_FACE          - student not visible
  MULTIPLE_PEOPLE  - more than one person (extra face, or extra body in frame)
  PHONE_DETECTED   - a mobile phone is visible
  BOOK_DETECTED    - a book is visible
  HEAD_TURNED      - head turned left/right or tilted up/down
  GAZE_AWAY        - eyes looking left/right/up/down (iris tracking)

When a violation lasts HOLD_SECONDS, it beeps, saves a snapshot to evidence/
and writes a row to events.csv. The screen border also turns red.

The numbers at the bottom-left show how far you are from your calibrated
position. Use them to tune the LIMIT values below.

Controls: press 'q' to quit, 'c' to re-calibrate.
"""
import csv
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

try:
    import winsound          # built into Windows, nothing to install
except ImportError:
    winsound = None

# ---------------- Settings (you can tune these) ----------------
HOLD_SECONDS = 1.0         # how long a behaviour must last before the alert (smaller = stricter)
COOLDOWN_SECONDS = 3.0     # minimum gap between two alerts of the same type
YAW_LIMIT = 0.06           # head left/right allowed (smaller = stricter)
PITCH_LIMIT = 0.06         # head up/down allowed
GAZE_H_LIMIT = 0.06        # eyes left/right allowed
GAZE_V_LIMIT = 0.10        # eyes up/down allowed
EAR_CLOSED = 0.18          # below this the eyes are treated as closed (blink)
CALIB_FRAMES = 50          # frames used to learn your normal position
SMOOTH_FRAMES = 3          # average over N frames to reduce jitter
OBJECT_CONF = 0.35         # YOLO confidence for phone / book
PERSON_CONF = 0.5          # YOLO confidence for a person
YOLO_EVERY_N_FRAMES = 3    # run object detection every N frames
SOUND_ON = True            # set to False to mute
EVIDENCE_DIR = "evidence"
LOG_FILE = "events.csv"
# COCO class ids: 0 = person, 67 = cell phone, 73 = book
YOLO_CLASSES = [0, 67, 73]
# ---------------------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh
_alert_busy = False


def play_alert():
    """Play a two-tone beep without freezing the video."""
    global _alert_busy
    if not SOUND_ON or _alert_busy:
        return

    def _run():
        global _alert_busy
        _alert_busy = True
        try:
            if winsound:
                winsound.Beep(1400, 250)
                winsound.Beep(1000, 350)
            else:
                print("\a", end="", flush=True)
        finally:
            _alert_busy = False

    threading.Thread(target=_run, daemon=True).start()


def head_ratios(face):
    """(yaw, pitch) ratios from face landmarks. ~0.5 when facing the screen."""
    def pt(i):
        return face.landmark[i].x, face.landmark[i].y

    nose, left, right, top, chin = pt(1), pt(234), pt(454), pt(10), pt(152)
    yaw = (nose[0] - left[0]) / (right[0] - left[0] + 1e-6)
    pitch = (nose[1] - top[1]) / (chin[1] - top[1] + 1e-6)
    return yaw, pitch


def eye_metrics(face):
    """Return (eye openness, horizontal gaze, vertical gaze) from eye + iris landmarks."""
    lm = face.landmark

    def p(i):
        return np.array([lm[i].x, lm[i].y])

    def ear(a, b, c, d, e, f):
        vertical = np.linalg.norm(p(b) - p(f)) + np.linalg.norm(p(c) - p(e))
        horizontal = 2 * np.linalg.norm(p(a) - p(d)) + 1e-6
        return vertical / horizontal

    def iris_h(a, b, iris):      # iris position between the two eye corners
        return (lm[iris].x - lm[a].x) / (lm[b].x - lm[a].x + 1e-6)

    def iris_v(top, bottom, iris):   # iris position between top and bottom eyelid
        return (lm[iris].y - lm[top].y) / (lm[bottom].y - lm[top].y + 1e-6)

    ear_val = (ear(33, 160, 158, 133, 153, 144) + ear(362, 385, 387, 263, 373, 380)) / 2
    gaze_h = (iris_h(33, 133, 468) + iris_h(362, 263, 473)) / 2
    gaze_v = (iris_v(159, 145, 468) + iris_v(386, 374, 473)) / 2
    return ear_val, gaze_h, gaze_v


def main():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    is_new_log = not os.path.exists(LOG_FILE)
    log_file = open(LOG_FILE, "a", newline="")
    writer = csv.writer(log_file)
    if is_new_log:
        writer.writerow(["time", "event", "snapshot"])

    print("Loading object detection model (first run downloads it)...")
    yolo = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    start_times = {}
    last_logged = {}
    totals = Counter()
    calib = []
    base = None                       # calibrated (yaw, pitch, gaze_h, gaze_v)
    smooth = [deque(maxlen=SMOOTH_FRAMES) for _ in range(4)]
    detections = []                   # (class_id, confidence, box)
    dev = (0.0, 0.0, 0.0, 0.0)        # deviations shown on screen
    frame_count = 0

    def check(event, active, now, raw_frame):
        """Alert + log a violation if `active` has stayed true for HOLD_SECONDS."""
        if not active:
            start_times.pop(event, None)
            return
        start_times.setdefault(event, now)
        held = now - start_times[event]
        if held >= HOLD_SECONDS and now - last_logged.get(event, 0) >= COOLDOWN_SECONDS:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap = os.path.join(EVIDENCE_DIR, f"{stamp}_{event}.jpg")
            cv2.imwrite(snap, raw_frame)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, snap])
            log_file.flush()
            last_logged[event] = now
            totals[event] += 1
            play_alert()
            print(f"[VIOLATION] {event} -> {snap}")

    with mp_face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            raw = frame.copy()
            h, w = frame.shape[:2]
            now = time.time()
            frame_count += 1

            # ---- Faces ----
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = result.multi_face_landmarks or []
            n_faces = len(faces)

            # ---- Head + eye direction ----
            head_away = gaze_away = False
            if n_faces == 1:
                yaw, pitch = head_ratios(faces[0])
                ear_val, gaze_h, gaze_v = eye_metrics(faces[0])
                for buf, val in zip(smooth, (yaw, pitch, gaze_h, gaze_v)):
                    buf.append(val)
                if len(smooth[0]) == SMOOTH_FRAMES:
                    m = [float(np.mean(buf)) for buf in smooth]
                    if base is None:
                        if ear_val > EAR_CLOSED:   # ignore blinks while calibrating
                            calib.append(m)
                        if len(calib) >= CALIB_FRAMES:
                            base = np.mean(calib, axis=0)
                            print("Calibration done. Monitoring started.")
                    else:
                        dev = tuple(m[i] - base[i] for i in range(4))
                        head_away = abs(dev[0]) > YAW_LIMIT or abs(dev[1]) > PITCH_LIMIT
                        gaze_away = ear_val > EAR_CLOSED and (
                            abs(dev[2]) > GAZE_H_LIMIT or abs(dev[3]) > GAZE_V_LIMIT)
            else:
                for buf in smooth:
                    buf.clear()

            # ---- Objects: person / phone / book ----
            if frame_count % YOLO_EVERY_N_FRAMES == 0:
                det = yolo(frame, verbose=False, conf=OBJECT_CONF, classes=YOLO_CLASSES)[0]
                detections = []
                for b in det.boxes:
                    cls, conf = int(b.cls[0]), float(b.conf[0])
                    if cls == 0 and conf < PERSON_CONF:
                        continue
                    detections.append((cls, conf, tuple(map(int, b.xyxy[0]))))
            persons = sum(1 for d in detections if d[0] == 0)
            phone_found = any(d[0] == 67 for d in detections)
            book_found = any(d[0] == 73 for d in detections)

            # ---- Violations (only after calibration) ----
            calibrated = base is not None
            if calibrated:
                check("NO_FACE", n_faces == 0, now, raw)
                check("MULTIPLE_PEOPLE", n_faces > 1 or persons > 1, now, raw)
                check("PHONE_DETECTED", phone_found, now, raw)
                check("BOOK_DETECTED", book_found, now, raw)
                check("HEAD_TURNED", head_away, now, raw)
                check("GAZE_AWAY", gaze_away, now, raw)

            # ---- Draw ----
            labels = {0: "PERSON", 67: "PHONE", 73: "BOOK"}
            for cls, conf, (x1, y1, x2, y2) in detections:
                col = (255, 120, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{labels[cls]} {conf:.2f}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

            if not calibrated:
                status, color = "Calibrating... look at the screen", (0, 255, 255)
            elif n_faces == 0:
                status, color = "No face detected", (0, 0, 255)
            elif n_faces > 1 or persons > 1:
                status, color = "Multiple people!", (0, 0, 255)
            elif phone_found:
                status, color = "Phone detected!", (0, 0, 255)
            elif book_found:
                status, color = "Book detected!", (0, 0, 255)
            elif head_away:
                status, color = "Head turned away", (0, 165, 255)
            elif gaze_away:
                status, color = "Eyes looking away", (0, 165, 255)
            else:
                status, color = "OK", (0, 200, 0)

            if calibrated and status != "OK":
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)   # warning border

            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Violations logged: {sum(totals.values())}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame,
                        f"head L/R {dev[0]:+.2f} (lim {YAW_LIMIT})  U/D {dev[1]:+.2f} (lim {PITCH_LIMIT})",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame,
                        f"eyes L/R {dev[2]:+.2f} (lim {GAZE_H_LIMIT})  U/D {dev[3]:+.2f} (lim {GAZE_V_LIMIT})",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("AI Exam Proctor (q = quit, c = recalibrate)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calib, base = [], None
                for buf in smooth:
                    buf.clear()

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

    print("\n===== Session summary =====")
    if totals:
        for event, count in totals.items():
            print(f"{event}: {count}")
    else:
        print("No violations. Clean session.")


if __name__ == "__main__":
    main()

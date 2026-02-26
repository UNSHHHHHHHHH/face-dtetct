'''
Real-time face recognition using dlib + face_recognition library.
Uses pre-computed embeddings from dlib_face_embeddings.py.

Controls:
  Q - Quit
  S - Save current frame snapshot to reports/
  R - Toggle recording on/off
'''

import cv2
import face_recognition
import pickle
import os
import time
import numpy as np
from datetime import datetime

from parameters import (
    DLIB_FACE_ENCODING_PATH,
    FACE_MATCHING_TOLERANCE,
    FACE_RECOGNITION_MODEL,
    NUMBER_OF_TIMES_TO_UPSAMPLE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    REPORT_PATH,
)

# ── Colours and font ──────────────────────────────────────────────────────────
KNOWN_COLOUR   = (0, 200, 0)    # green  – recognised face
UNKNOWN_COLOUR = (0, 0, 220)    # red    – unknown face
TEXT_COLOUR    = (255, 255, 255)
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# ── Process every N-th frame for speed (1 = every frame) ─────────────────────
PROCESS_EVERY_N_FRAMES = 2


def load_encodings(path: str):
    '''Load pre-computed face encodings from a pickle file.'''
    print(f'[INFO] Loading encodings from {path} ...')
    with open(path, 'rb') as f:
        data = pickle.loads(f.read())
    print(f'[INFO] Loaded {len(data["names"])} face encoding(s).')
    return data['encodings'], data['names']


def open_camera():
    '''Try to open the default webcam (index 0).'''
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open camera. '
                           'Make sure a webcam is connected.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f'[INFO] Camera opened at {FRAME_WIDTH}x{FRAME_HEIGHT}')
    return cap


def draw_box(frame, top, right, bottom, left, name, colour):
    '''Draw a labelled bounding box around a detected face.'''
    cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
    cv2.rectangle(frame, (left, bottom - 28), (right, bottom), colour, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 8),
                FONT, 0.65, TEXT_COLOUR, 1)


def run_recognition():
    # ── Load embeddings ───────────────────────────────────────────────────────
    known_encodings, known_names = load_encodings(DLIB_FACE_ENCODING_PATH)

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = open_camera()

    # ── Prepare output directories ────────────────────────────────────────────
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    snapshots_dir = os.path.join(os.path.dirname(REPORT_PATH), 'snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)

    # ── Video writer (optional recording) ─────────────────────────────────────
    recording   = False
    video_writer = None

    frame_count    = 0
    face_locations = []
    face_names     = []

    print('[INFO] Starting face recognition. Press Q to quit, S to snapshot, R to record.')

    # ── Recognition CSV log ───────────────────────────────────────────────────
    seen_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[WARN] Failed to grab frame. Retrying...')
            time.sleep(0.1)
            continue

        frame_count += 1
        display = frame.copy()

        # ── Only process every N-th frame ─────────────────────────────────────
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Shrink frame to ¼ size for faster face detection
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(
                rgb_small,
                number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                model=FACE_RECOGNITION_MODEL,
            )
            face_encodings = face_recognition.face_encodings(
                rgb_small,
                known_face_locations=face_locations,
                num_jitters=1,
                model='small',
            )

            face_names = []
            for encoding in face_encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx  = int(np.argmin(distances)) if len(distances) else -1
                name      = 'Unknown'
                if best_idx >= 0 and distances[best_idx] <= FACE_MATCHING_TOLERANCE:
                    name = known_names[best_idx]
                    seen_names.add(name)
                face_names.append(name)

        # ── Draw boxes (scale coords back up ×4) ──────────────────────────────
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top    *= 4;  right  *= 4
            bottom *= 4;  left   *= 4
            colour  = KNOWN_COLOUR if name != 'Unknown' else UNKNOWN_COLOUR
            draw_box(display, top, right, bottom, left, name, colour)

        # ── HUD overlay ───────────────────────────────────────────────────────
        ts  = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        fps_text = f'Frame: {frame_count}  |  {ts}'
        rec_text = '⏺ REC' if recording else ''
        cv2.putText(display, fps_text, (10, 25), FONT, 0.55, (200, 200, 200), 1)
        if rec_text:
            cv2.putText(display, rec_text, (FRAME_WIDTH - 80, 25),
                        FONT, 0.65, (0, 0, 255), 2)

        # ── Optional recording ─────────────────────────────────────────────────
        if recording and video_writer:
            video_writer.write(display)

        cv2.imshow('GRIL Team Face Recognition  [Q=quit  S=snapshot  R=record]', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC → quit
            break

        elif key == ord('s'):              # S → save snapshot
            snap_path = os.path.join(
                snapshots_dir,
                f'snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            )
            cv2.imwrite(snap_path, display)
            print(f'[INFO] Snapshot saved → {snap_path}')

        elif key == ord('r'):              # R → toggle recording
            recording = not recording
            if recording:
                vid_path = os.path.join(
                    snapshots_dir,
                    f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
                )
                fourcc       = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(vid_path, fourcc, 20.0,
                                               (FRAME_WIDTH, FRAME_HEIGHT))
                print(f'[INFO] Recording started → {vid_path}')
            else:
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print('[INFO] Recording stopped.')

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # ── Save attendance-style CSV ──────────────────────────────────────────────
    if seen_names:
        ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_lines = [f'name,seen_at\n']
        for n in sorted(seen_names):
            report_lines.append(f'{n},{ts_str}\n')
        with open(REPORT_PATH, 'w') as f:
            f.writelines(report_lines)
        print(f'[INFO] Attendance report saved → {REPORT_PATH}')

    print(f'[INFO] Recognised faces: {sorted(seen_names) if seen_names else "none"}')
    print('[INFO] Done.')


if __name__ == '__main__':
    run_recognition()

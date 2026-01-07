import cv2
import pandas as pd
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

# --- Paths ---
VIDEO_PATH = "/home/she/Desktop/cctv_transaction_project/videos/chunks/chunk_1.mp4"
OUTPUT_CSV = "/home/she/Desktop/Watershopsystem/outputs/person_tracks_chunk1.csv"

# ✅ Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# --- Load YOLO model for person detection ---
model = YOLO("yolov8n.pt")

# --- Initialize DeepSort tracker ---
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)

# --- Open video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

rows = []
frame_id = 0

# --- Tracking loop ---
with tqdm(total=total_frames, desc="Person tracking (chunk 1)") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        timestamp = round(frame_id / fps, 2)

        results = model(frame, classes=[0], conf=0.4, verbose=False)

        detections = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            l, t_, r, b = map(int, t.to_ltrb())
            rows.append([frame_id, timestamp, t.track_id, l, t_, r, b])

        pbar.update(1)

cap.release()

# --- Save CSV ---
df = pd.DataFrame(rows, columns=["frame", "timestamp", "customer_id", "x1", "y1", "x2", "y2"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"Person tracks saved to: {OUTPUT_CSV}")

import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

# ================= PATHS =================
VIDEO_PATH = "/home/she/Desktop/cctv_transaction_project/videos/chunks/chunk_1.mp4"
CAN_MODEL_PATH = "/home/she/Desktop/models/water_can.pt"
OUTPUT_CSV = "/home/she/Desktop/cctv_transaction_project/outputs/can_tracks_chunk1.csv"

# ================= LOAD MODEL =================
print("🔄 Loading can detection model...")
can_model = YOLO(CAN_MODEL_PATH)

# DeepSORT for CAN TRACKING
can_tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)

CAN_CONF = 0.4

# ================= VIDEO INIT =================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_id = 0
rows = []

print("▶️ Starting CAN tracking...")

# ================= PROCESS VIDEO =================
with tqdm(total=total_frames, desc="📦 Tracking Cans", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        timestamp = frame_id / fps

        detections = []

        # -------- CAN DETECTION --------
        results = can_model(frame, conf=CAN_CONF, verbose=False)

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(b.conf[0])

                # DeepSORT expects [x, y, w, h]
                detections.append(([x1, y1, w, h], conf, "can"))

        # -------- TRACK CANS --------
        tracks = can_tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            can_id = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())

            rows.append([
                frame_id,
                round(timestamp, 2),
                can_id,
                x1, y1, x2, y2
            ])

        pbar.update(1)

cap.release()

# ================= SAVE CSV =================
df = pd.DataFrame(
    rows,
    columns=["frame", "timestamp", "can_track_id", "x1", "y1", "x2", "y2"]
)

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Can tracking CSV saved: {OUTPUT_CSV}")

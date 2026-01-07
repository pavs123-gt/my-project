import cv2
import os
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from tqdm import tqdm

# ================= CONFIG =================
FRAME_DIR = "/home/she/Desktop/cctv_transaction_project/outputs/frames"
CAN_MODEL_PATH = "/home/she/Desktop/models/water_can.pt"
OUTPUT_CSV = "phase2_transaction_events_final.csv"

# ✅ COUNTER ROI (FIXED FROM CCTV VIEW)
# (x1, y1, x2, y2)
COUNTER_ROI = (850, 200, 1200, 650)

# ================= INIT MODELS =================
person_model = YOLO("yolov8n.pt")      # Person detection
can_model = YOLO(CAN_MODEL_PATH)       # Water can detection
tracker = DeepSort(max_age=30, n_init=3)

# ================= HELPERS =================
def overlap(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    return (xb - xa) > 0 and (yb - ya) > 0

# ✅ FEET-BASED ROI CHECK (CRITICAL)
def inside_roi_feet(bbox, roi):
    x1, y1, x2, y2 = bbox
    fx = (x1 + x2) // 2   # feet X
    fy = y2               # feet Y
    rx1, ry1, rx2, ry2 = roi
    return rx1 < fx < rx2 and ry1 < fy < ry2

# ================= MEMORY =================
customer_state = defaultdict(lambda: {
    "enter_frame": None,
    "exit_frame": None,
    "took_can": False,
    "counter_wait_frames": 0,
    "active": False
})

event_rows = []
active_ids = set()

# ================= LOAD FRAMES =================
frame_files = sorted([
    os.path.join(FRAME_DIR, f)
    for f in os.listdir(FRAME_DIR)
    if f.endswith(".jpg")
])

print(f"✅ Total frames found: {len(frame_files)}")

# ================= PROCESS FRAMES =================
for frame_id, frame_file in enumerate(tqdm(frame_files, desc="Phase-2 Processing")):
    frame = cv2.imread(frame_file)
    if frame is None:
        continue

    # ---- PERSON DETECTION ----
    results = person_model.predict(frame, classes=[0], conf=0.4, verbose=False)[0]
    detections = []

    for r in results.boxes:
        x, y, w, h = r.xywh.cpu().numpy()[0]
        conf = float(r.conf.cpu().numpy()[0])
        detections.append(([x - w/2, y - h/2, w, h], conf, 0))

    # ---- CAN DETECTION ----
    can_results = can_model.predict(frame, conf=0.4, verbose=False)[0]
    can_boxes = [tuple(map(int, box.xyxy[0])) for box in can_results.boxes]

    # ---- TRACKING ----
    tracks = tracker.update_tracks(detections, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        cid = f"CUST_{int(track.track_id):04d}"
        current_ids.add(cid)

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        bbox = (x1, y1, x2, y2)
        state = customer_state[cid]

        # ---- ENTER ----
        if not state["active"]:
            state["enter_frame"] = frame_id
            state["active"] = True

        # ---- TAKE CAN ----
        for can in can_boxes:
            if overlap(bbox, can):
                state["took_can"] = True

        # ---- COUNTER WAIT (FEET BASED) ----
        if inside_roi_feet(bbox, COUNTER_ROI):
            state["counter_wait_frames"] += 1

    # ---- EXIT ----
    disappeared = active_ids - current_ids
    for cid in disappeared:
        st = customer_state[cid]
        st["exit_frame"] = frame_id
        st["active"] = False

        event_rows.append({
            "customer_id": cid,
            "enter_frame": st["enter_frame"],
            "exit_frame": st["exit_frame"],
            "took_can": st["took_can"],
            "counter_wait_frames": st["counter_wait_frames"]
        })

    active_ids = current_ids

# ================= SAVE =================
df = pd.DataFrame(event_rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ PHASE-2 COMPLETE (FINAL)")
print("Saved CSV →", OUTPUT_CSV)
print(df.head())

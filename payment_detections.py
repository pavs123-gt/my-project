import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from tqdm import tqdm
import glob, os

# ================== PATHS ==================
FRAME_DIR = "/home/she/Desktop/cctv_transaction_project/frames_chunk1/"
CAN_MODEL_PATH = "/home/she/Desktop/models/water_can.pt"
OUTPUT_CSV = "transactiong_chunk1.csv"

# ================== LOAD MODELS ==================
coco_model = YOLO("yolov8n.pt")          # person + cell phone
can_model = YOLO(CAN_MODEL_PATH)         # water can
tracker = DeepSort(max_age=30)

# ================== ROI / SETTINGS ==================
COUNTER_ROI = (500, 300, 900, 650)
FPS = 25

# ================== HELPERS ==================
def inside_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi
    return x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2

def overlap(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    return (xb - xa) > 0 and (yb - ya) > 0

def detect_hand_exchange(phone_boxes, person_bbox):
    """
    Heuristic:
    If NO phone detected but person stays at counter long → assume cash/coins
    """
    for phone in phone_boxes:
        if overlap(person_bbox, phone):
            return False
    return True

# ================== MEMORY ==================
person_state = defaultdict(lambda: {
    "has_item": False,
    "counter_frames": 0,
    "phone_detected": False,
    "hand_exchange": False,
    "exited": False
})

transactions = []
active_tracks = set()

# ================== PROCESS FRAMES ==================
frame_files = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
if not frame_files:
    raise ValueError("No frames found")

for frame_file in tqdm(frame_files):
    frame = cv2.imread(frame_file)
    if frame is None:
        continue

    # ---- COCO DETECTIONS ----
    coco_results = coco_model(frame, conf=0.4, iou=0.5, verbose=False)[0]
    person_dets, phone_boxes = [], []

    for box in coco_results.boxes:
        cls = coco_model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == "person":
            person_dets.append(([x1, y1, x2-x1, y2-y1], box.conf[0], "person"))
        elif cls == "cell phone":
            phone_boxes.append((x1, y1, x2, y2))

    # ---- CAN DETECTIONS ----
    can_results = can_model(frame, conf=0.4, iou=0.5, verbose=False)[0]
    can_boxes = [tuple(map(int, box.xyxy[0])) for box in can_results.boxes]

    # ---- TRACK ----
    tracks = tracker.update_tracks(person_dets, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        pid = f"CUST_{int(track.track_id):04d}"

        current_ids.add(pid)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        bbox = (x1, y1, x2, y2)
        state = person_state[pid]

        for can in can_boxes:
            if overlap(bbox, can):
                state["has_item"] = True

        if inside_roi(bbox, COUNTER_ROI):
            state["counter_frames"] += 1
            if detect_hand_exchange(phone_boxes, bbox):
                state["hand_exchange"] = True
        else:
            state["counter_frames"] = max(0, state["counter_frames"] - 1)

        for phone in phone_boxes:
            if overlap(bbox, phone):
                state["phone_detected"] = True

    # ---- EXITED ----
    disappeared = active_tracks - current_ids
    for pid in disappeared:
        state = person_state[pid]
        if state["exited"]:
            continue

        state["exited"] = True
        paid = state["has_item"] and state["counter_frames"] > FPS * 3
        payment_status = "PAID" if paid else "UNPAID"

        if not paid:
            mode = "NONE"
        elif state["phone_detected"]:
            mode = "UPI"
        elif state["hand_exchange"]:
            mode = "CASH"
        else:
            mode = "COINS"

        transactions.append({
            "customer_id": pid,
            "payment_status": payment_status,
            "payment_mode": mode
        })

    active_tracks = current_ids

# ================== SAVE ==================
df = pd.DataFrame(transactions)
df.to_csv(OUTPUT_CSV, index=False)
print("✅ Done")
print(df.head())

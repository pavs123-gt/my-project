import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import glob

# ================= CONFIG =================
FRAME_DIR = "/home/she/Desktop/cctv_transaction_project/outputs/frames"
OUTPUT_DIR = "/home/she/Desktop/cctv_transaction_project/person_clips"
YOLO_MODEL = "yolov8n.pt"
FPS = 6
CLIP_SECONDS = 2
CLIP_LENGTH = FPS * CLIP_SECONDS
RESIZE_DIM = (224, 224) # Standard for Video Transformers

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= INIT MODELS =================
yolo_model = YOLO(YOLO_MODEL)
# Re-ID model is built-in to handle person consistency
tracker = DeepSort(max_age=30, n_init=3) 

# ================= GET FRAMES =================
frame_files = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
total_frames = len(frame_files)
print(f"✅ Found {total_frames} frames. Starting processing...")

person_buffers = {}
person_frame_counts = {}

pbar = tqdm(total=total_frames, desc="Processing")

for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    if frame is None: continue

    pbar.update(1)

    # 1. YOLO Detection (Person only, class 0)
    results = yolo_model.predict(frame, classes=[0], conf=0.4, verbose=False)
    
    detections = []
    # 2. Correct Formatting for DeepSORT
    for r in results[0].boxes:
        # Get coordinates in [x_center, y_center, w, h]
        row = r.xywh.cpu().numpy()[0] 
        conf = float(r.conf.cpu().numpy()[0])
        
        # Convert to [left, top, w, h]
        left = float(row[0] - row[2]/2)
        top = float(row[1] - row[3]/2)
        w = float(row[2])
        h = float(row[3])
        
        # This nested list structure is what DeepSORT requires
        detections.append(([left, top, w, h], conf, 0))

    # 3. Update Tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb() # Left, Top, Right, Bottom
        
        # Crop and Resize
        x1, y1, x2, y2 = map(int, ltrb)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        # Resize to fixed dimension for the Transformer later
        crop_resized = cv2.resize(crop, RESIZE_DIM)

        if track_id not in person_buffers:
            person_buffers[track_id] = []
            
        person_buffers[track_id].append(crop_resized)

        # 4. Save Clip when buffer is full
        if len(person_buffers[track_id]) >= CLIP_LENGTH:
            clip_name = f"person_{track_id}_f{pbar.n}.mp4"
            clip_path = os.path.join(OUTPUT_DIR, clip_name)
            
            out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RESIZE_DIM)
            for f in person_buffers[track_id]:
                out.write(f)
            out.release()
            
            # Clear buffer for this person to start next 2-second segment
            person_buffers[track_id] = []

pbar.close()
print(f"✅ Finished! Clips saved to: {OUTPUT_DIR}")
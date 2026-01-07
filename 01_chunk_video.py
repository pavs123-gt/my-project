import cv2
import os

# Paths
INPUT_VIDEO = "/home/she/Desktop/cctv_transaction_project/videos/cctv.mp4"
OUTPUT_DIR = "videos/chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

chunk_minutes = 20
frames_per_chunk = int(fps * 60 * chunk_minutes)

chunk_id = 1
frame_count = 0
writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frames_per_chunk == 0:
        if writer:
            writer.release()

        out_path = f"{OUTPUT_DIR}/chunk_{chunk_id}.mp4"
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        print(f"Creating {out_path}")
        chunk_id += 1

    writer.write(frame)
    frame_count += 1

if writer:
    writer.release()

cap.release()
print(" Video chunking completed")

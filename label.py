import cv2
import os
import shutil

# ====== CONFIG ======
CLIP_DIR = "/home/she/Desktop/project/person_clips"
OUTPUT_DIR = "/home/she/Desktop/cctv_transaction_project/action_dataset"

ACTIONS = {
    ord('1'): "TAKE_CAN",
    ord('2'): "WAIT",
    ord('3'): "HAND",
    ord('4'): "WALK",
    ord('5'): "OTHER"
}

# Create folders
for action in ACTIONS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, action), exist_ok=True)

video_files = sorted([f for f in os.listdir(CLIP_DIR) if f.endswith(".mp4")])

print("🎯 Labeling started")
print("1=TAKE_CAN | 2=WAIT | 3=HAND | 4=WALK | 5=OTHER | q=QUIT")

for video in video_files:
    video_path = os.path.join(CLIP_DIR, video)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.putText(
            frame,
            "1:TAKE  2:WAIT  3:HAND  4:WALK  5:OTHER  q:QUIT",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Label Clip", frame)
        key = cv2.waitKey(30)

        if key in ACTIONS:
            label = ACTIONS[key]
            dst = os.path.join(OUTPUT_DIR, label, video)
            shutil.move(video_path, dst)
            print(f"✔ {video} → {label}")
            break

        elif key == ord('q'):
            print("🛑 Labeling stopped")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

cv2.destroyAllWindows()
print("✅ Dataset labeling complete")

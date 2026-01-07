import pandas as pd
from collections import defaultdict

# ================= LOAD CSVs =================
person_df = pd.read_csv("/home/she/Desktop/cctv_transaction_project/outputs/person_tracks_chunk1.csv")
can_df = pd.read_csv("/home/she/Desktop/cctv_transaction_project/outputs/can_tracks_chunk1.csv")

# ================= HELPERS =================
def inside(px1, py1, px2, py2, cx, cy):
    return px1 <= cx <= px2 and py1 <= cy <= py2

# ================= PREP =================
# frame → list of persons
person_by_frame = defaultdict(list)
for _, r in person_df.iterrows():
    person_by_frame[r["frame"]].append(
        (r["customer_id"], r["x1"], r["y1"], r["x2"], r["y2"])
    )

# ================= CAN → CUSTOMER ASSOCIATION =================
can_customer_frames = defaultdict(lambda: defaultdict(int))

for _, r in can_df.iterrows():
    frame = r["frame"]
    can_id = r["can_track_id"]

    cx = (r["x1"] + r["x2"]) // 2
    cy = (r["y1"] + r["y2"]) // 2

    for cid, px1, py1, px2, py2 in person_by_frame.get(frame, []):
        if inside(px1, py1, px2, py2, cx, cy):
            can_customer_frames[can_id][cid] += 1

# ================= FINAL CAN OWNERSHIP =================
MIN_FRAMES = 15   # 🔴 Tune if needed

customer_cans = defaultdict(set)

for can_id, owners in can_customer_frames.items():
    # choose customer with max overlap
    best_customer, frames = max(owners.items(), key=lambda x: x[1])

    if frames >= MIN_FRAMES:
        customer_cans[best_customer].add(can_id)

# ================= FINAL RESULT =================
rows = []
for cid, cans in customer_cans.items():
    rows.append([cid, len(cans)])

final_df = pd.DataFrame(
    rows,
    columns=["customer_id", "no_of_cans"]
)

final_df.to_csv("customer_can_counts_chunk1.csv", index=False)

print("✅ Correct can counts generated")
print(final_df)

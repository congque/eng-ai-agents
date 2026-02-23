from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from tqdm import tqdm

VIDEO_ID = "YcvECxtXoxQ"
MODEL_PATH = "../runs/segment/weights/carparts_y8n3/weights/best.pt"
FRAME_DIR = "frames"
OUT_PATH = "out/video_detections.parquet"

model = YOLO(MODEL_PATH)

rows = []
frame_paths = sorted(Path(FRAME_DIR).glob("*.jpg"))

for idx, fp in enumerate(tqdm(frame_paths)):
    results = model.predict(source=str(fp), verbose=False)
    r = results[0]

    if r.boxes is None:
        continue

    for box in r.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        rows.append({
            "video_id": VIDEO_ID,
            "frame_index": idx,
            "timestamp_sec": idx * 5,
            "class_label": model.names[cls_id],
            "bounding_box": [x1, y1, x2, y2],
            "confidence_score": conf
        })

df = pd.DataFrame(rows)
Path("out").mkdir(exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print(df.head())
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from ultralytics import YOLO
from datasets import load_from_disk

MODEL_PATH = "./runs/segment/weights/carparts_y8n3/weights/best.pt"
PARQUET_PATH = "./out/video_detections.parquet"
QUERY_DISK_PATH = "./rav4_exterior_images" 
STRIDE_SEC = 5                       
DEVICE = "0" 
CONF = 0.25 
OUT_JSONL = "./out/retrieval_results.jsonl"
TOPK_LABELS = 1 

def must_exist(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required path not found: {p.resolve()}")
    return p


def merge_frames_to_intervals(frames: List[int], stride_sec: int) -> List[Dict[str, Any]]:
    frames_sorted = sorted(set(int(x) for x in frames))
    if not frames_sorted:
        return []

    intervals: List[Tuple[int, int]] = []
    s = prev = frames_sorted[0]
    for f in frames_sorted[1:]:
        if f == prev + 1:
            prev = f
        else:
            intervals.append((s, prev))
            s = prev = f
    intervals.append((s, prev))

    out: List[Dict[str, Any]] = []
    for a, b in intervals:
        out.append(
            {
                "start_timestamp": a * stride_sec,
                "end_timestamp": b * stride_sec + stride_sec,
                "number_of_supporting_detections": (b - a + 1),
            }
        )
    return out


def topk_labels_from_query(model: YOLO, img, topk: int, device: str, conf: float) -> List[Tuple[str, float]]:
    res = model.predict(source=img, verbose=False, device=device, conf=conf)
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    confs = r.boxes.conf.detach().cpu().tolist()
    clses = r.boxes.cls.detach().cpu().tolist()

    pairs = []
    for c, s in zip(clses, confs):
        cls_id = int(c)
        label = model.names.get(cls_id, str(cls_id))
        pairs.append((label, float(s)))

    # keep best confidence per label
    best_by_label: Dict[str, float] = {}
    for label, s in sorted(pairs, key=lambda x: x[1], reverse=True):
        if label not in best_by_label:
            best_by_label[label] = s

    out = list(best_by_label.items())
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:topk]


def main() -> None:
    must_exist(MODEL_PATH)
    must_exist(PARQUET_PATH)
    must_exist(QUERY_DISK_PATH)

    Path("out").mkdir(exist_ok=True)

    print(f"Loading model: {Path(MODEL_PATH).resolve()}")
    model = YOLO(MODEL_PATH)

    print(f"Loading detections parquet: {Path(PARQUET_PATH).resolve()}")
    df = pd.read_parquet(PARQUET_PATH)

    required_cols = {"frame_index", "class_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    print(f"Loading query dataset from disk: {Path(QUERY_DISK_PATH).resolve()}")
    dataset = load_from_disk(QUERY_DISK_PATH)
    print("Query dataset:", dataset)

    if "image" not in dataset.features:
        raise ValueError(f"Query dataset has no 'image' column. Features: {list(dataset.features.keys())}")

    out_path = Path(OUT_JSONL)
    n_with_intervals = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            img = sample["image"]  # PIL.Image
            q_ts = sample.get("timestamp_sec", None)
            q_title = sample.get("video_title", None)
            exterior_score = sample.get("exterior_score", None)

            label_scores = topk_labels_from_query(model, img, TOPK_LABELS, DEVICE, CONF)

            if not label_scores:
                record = {
                    "query_index": i,
                    "query_timestamp_sec": q_ts,
                    "query_video_title": q_title,
                    "query_exterior_score": exterior_score,
                    "predicted_labels": [],
                    "intervals": [],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            labels = [ls[0] for ls in label_scores]

            sub = df[df["class_label"].isin(labels)]
            intervals: List[Dict[str, Any]] = []
            if not sub.empty:
                frames = sub["frame_index"].tolist()
                intervals = merge_frames_to_intervals(frames, STRIDE_SEC)
                for itv in intervals:
                    itv["class_label"] = labels[0] if TOPK_LABELS == 1 else labels

            if intervals:
                n_with_intervals += 1

            record = {
                "query_index": i,
                "query_timestamp_sec": q_ts,
                "query_video_title": q_title,
                "query_exterior_score": exterior_score,
                "predicted_labels": [{"label": lab, "conf": sc} for lab, sc in label_scores],
                "intervals": intervals,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # console preview
            if intervals:
                print(f"\nQuery {i}: ts={q_ts} label(s)={labels} intervals={len(intervals)}")
                for itv in intervals[:10]:
                    print(itv)

    print(f"\nDone. Queries with >=1 interval: {n_with_intervals}/{len(dataset)}")
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
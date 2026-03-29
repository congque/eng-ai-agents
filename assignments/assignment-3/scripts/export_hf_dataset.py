from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def build_records(detections_df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    grouped = detections_df.groupby(
        ["video_name", "frame_index", "saved_frame_path", "timestamp_sec", "sample_fps"],
        sort=True,
    )

    for group_key, group_df in grouped:
        video_name, frame_index, saved_frame_path, timestamp_sec, sample_fps = group_key
        detections = []
        for _, row in group_df.iterrows():
            detections.append(
                {
                    "label": row["class_label"],
                    "score": float(row["confidence_score"]),
                    "bbox": [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ],
                }
            )

        records.append(
            {
                "image": saved_frame_path,
                "video_name": video_name,
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp_sec),
                "sample_fps": float(sample_fps),
                "num_detections": len(detections),
                "detections": detections,
                "detections_json": json.dumps(detections),
            }
        )

    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.parquet_path.parent.mkdir(parents=True, exist_ok=True)

    detections_df = pd.read_parquet(args.detections_parquet)
    records = build_records(detections_df)
    if not records:
        raise RuntimeError("No detection frames found.")

    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("image", Image())
    dataset.save_to_disk(str(args.output_dir))

    try:
        dataset.to_parquet(str(args.parquet_path))
    except AttributeError:
        dataset.to_pandas().to_parquet(args.parquet_path, index=False)

    if args.repo_id:
        token = os.environ.get("HF_TOKEN")
        dataset.push_to_hub(args.repo_id, private=args.private, token=token)


if __name__ == "__main__":
    main()

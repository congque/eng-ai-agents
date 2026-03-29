from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image  # noqa: F401
from ultralytics import YOLO


FRAME_PATTERN = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--class-names", nargs="*", default=None)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def frame_index_from_path(frame_path: Path) -> int:
    match = FRAME_PATTERN.search(frame_path.stem)
    if match is None:
        raise ValueError(f"Could not parse frame number from {frame_path.name}")
    return int(match.group(1)) - 1


def normalize_class_names(class_names: list[str] | None) -> set[str] | None:
    if not class_names:
        return None
    return {name.strip().lower() for name in class_names if name.strip()}


def keep_detection(class_label: str, allowed_names: set[str] | None) -> bool:
    if allowed_names is None:
        return True
    return class_label.strip().lower() in allowed_names


def copy_detection_frame(frame_path: Path, target_dir: Path, overwrite: bool) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / frame_path.name
    if overwrite or not target_path.exists():
        shutil.copy2(frame_path, target_path)
    return target_path.resolve()


def batched(paths: list[Path], batch_size: int) -> Iterable[list[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for index in range(0, len(paths), batch_size):
        yield paths[index : index + batch_size]


def main() -> None:
    args = parse_args()
    args.detections_dir.mkdir(parents=True, exist_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)

    frame_dirs = sorted(path for path in args.frames_dir.iterdir() if path.is_dir())
    if not frame_dirs:
        raise FileNotFoundError(f"No frame folders found in {args.frames_dir}")

    model = YOLO(args.model)
    allowed_names = normalize_class_names(args.class_names)
    rows: list[dict[str, object]] = []

    for frame_dir in frame_dirs:
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        for frame_batch in batched(frame_paths, args.batch_size):
            results = model.predict(
                source=[str(frame_path) for frame_path in frame_batch],
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            for frame_path, result in zip(frame_batch, results):
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                frame_index = frame_index_from_path(frame_path)
                height, width = result.orig_shape
                saved_frame_path: Path | None = None

                for detection_index, box in enumerate(result.boxes):
                    class_id = int(box.cls.item())
                    class_label = str(model.names[class_id])
                    if not keep_detection(class_label, allowed_names):
                        continue

                    if saved_frame_path is None:
                        saved_frame_path = copy_detection_frame(
                            frame_path=frame_path,
                            target_dir=args.detections_dir / frame_dir.name,
                            overwrite=args.overwrite,
                        )

                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    box_width = x2 - x1
                    box_height = y2 - y1
                    center_x = x1 + box_width / 2.0
                    center_y = y1 + box_height / 2.0

                    rows.append(
                        {
                            "video_name": frame_dir.name,
                            "frame_path": str(frame_path.resolve()),
                            "saved_frame_path": str(saved_frame_path),
                            "frame_index": frame_index,
                            "sample_fps": args.sample_fps,
                            "timestamp_sec": frame_index / args.sample_fps,
                            "image_width": width,
                            "image_height": height,
                            "detection_index": detection_index,
                            "class_id": class_id,
                            "class_label": class_label,
                            "confidence_score": float(box.conf.item()),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "bbox_width": box_width,
                            "bbox_height": box_height,
                            "center_x": center_x,
                            "center_y": center_y,
                        }
                    )

    if not rows:
        raise RuntimeError("No detections passed the class filter.")

    detections_df = pd.DataFrame(rows).sort_values(
        by=["video_name", "frame_index", "confidence_score"],
        ascending=[True, True, False],
    )
    detections_df.to_parquet(args.output_parquet, index=False)


if __name__ == "__main__":
    main()

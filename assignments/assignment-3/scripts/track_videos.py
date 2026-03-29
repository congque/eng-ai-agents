from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


FRAME_PATTERN = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--detections-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--max-missed", type=int, default=8)
    parser.add_argument("--association-distance", type=float, default=150.0)
    parser.add_argument("--measurement-var", type=float, default=25.0)
    parser.add_argument("--process-var", type=float, default=5.0)
    return parser.parse_args()


def frame_index_from_path(frame_path: Path) -> int:
    match = FRAME_PATTERN.search(frame_path.stem)
    if match is None:
        raise ValueError(f"Could not parse frame number from {frame_path.name}")
    return int(match.group(1)) - 1


def build_filter(dt: float, cx: float, cy: float, measurement_var: float, process_var: float) -> KalmanFilter:
    kalman = KalmanFilter(dim_x=4, dim_z=2)
    kalman.x = np.array([[cx], [cy], [0.0], [0.0]])
    kalman.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    kalman.P = np.eye(4) * 500.0
    kalman.R = np.eye(2) * measurement_var
    update_motion_model(kalman=kalman, dt=dt, process_var=process_var)
    return kalman


def update_motion_model(kalman: KalmanFilter, dt: float, process_var: float) -> None:
    kalman.F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q = np.array(
        [
            [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
            [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
            [dt**3 / 2.0, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2.0, 0.0, dt**2],
        ]
    )
    kalman.Q = q * process_var


def choose_detection(
    frame_detections: pd.DataFrame,
    predicted_center: tuple[float, float] | None,
    max_distance: float,
) -> pd.Series | None:
    if frame_detections.empty:
        return None

    candidates = frame_detections.copy()
    if predicted_center is None:
        return candidates.sort_values("confidence_score", ascending=False).iloc[0]

    dx = candidates["center_x"] - predicted_center[0]
    dy = candidates["center_y"] - predicted_center[1]
    candidates["distance"] = np.sqrt(dx * dx + dy * dy)
    candidates = candidates.sort_values(
        by=["distance", "confidence_score"],
        ascending=[True, False],
    )
    best = candidates.iloc[0]
    if max_distance > 0.0 and float(best["distance"]) > max_distance:
        return None
    return best


def bbox_from_center(
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    half_width = box_width / 2.0
    half_height = box_height / 2.0
    x1 = max(0.0, center_x - half_width)
    y1 = max(0.0, center_y - half_height)
    x2 = min(float(image_width), center_x + half_width)
    y2 = min(float(image_height), center_y + half_height)
    return x1, y1, x2, y2


def group_detections(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(frame_index): group.copy() for frame_index, group in df.groupby("frame_index")}


def main() -> None:
    args = parse_args()
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    detections_df = pd.read_parquet(args.detections_parquet)
    rows: list[dict[str, object]] = []

    for video_name, video_df in detections_df.groupby("video_name"):
        frame_dir = args.frames_dir / str(video_name)
        if not frame_dir.exists():
            continue

        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            continue

        sample_fps = float(video_df["sample_fps"].dropna().iloc[0])
        dt = 1.0 / sample_fps if sample_fps > 0 else 1.0
        image_width = int(video_df["image_width"].iloc[0])
        image_height = int(video_df["image_height"].iloc[0])
        detections_by_frame = group_detections(video_df)

        kalman: KalmanFilter | None = None
        track_id = 0
        missed_frames = 0
        box_width: float | None = None
        box_height: float | None = None
        last_timestamp: float | None = None

        for frame_path in frame_paths:
            frame_index = frame_index_from_path(frame_path)
            timestamp_sec = frame_index / sample_fps if sample_fps > 0 else float(frame_index)
            frame_detections = detections_by_frame.get(frame_index, pd.DataFrame())

            if kalman is None:
                initial_detection = choose_detection(
                    frame_detections=frame_detections,
                    predicted_center=None,
                    max_distance=args.association_distance,
                )
                if initial_detection is None:
                    continue

                box_width = float(initial_detection["bbox_width"])
                box_height = float(initial_detection["bbox_height"])
                kalman = build_filter(
                    dt=dt,
                    cx=float(initial_detection["center_x"]),
                    cy=float(initial_detection["center_y"]),
                    measurement_var=args.measurement_var,
                    process_var=args.process_var,
                )
                track_id += 1
                missed_frames = 0
                source = "detection"
                confidence_score = float(initial_detection["confidence_score"])
            else:
                current_dt = dt
                if last_timestamp is not None:
                    current_dt = max(timestamp_sec - last_timestamp, 1.0 / sample_fps)
                update_motion_model(kalman=kalman, dt=current_dt, process_var=args.process_var)
                kalman.predict()
                predicted_center = (float(kalman.x[0, 0]), float(kalman.x[1, 0]))

                matched_detection = choose_detection(
                    frame_detections=frame_detections,
                    predicted_center=predicted_center,
                    max_distance=args.association_distance,
                )
                if matched_detection is not None:
                    measurement = np.array(
                        [float(matched_detection["center_x"]), float(matched_detection["center_y"])]
                    )
                    kalman.update(measurement)
                    box_width = float(matched_detection["bbox_width"])
                    box_height = float(matched_detection["bbox_height"])
                    missed_frames = 0
                    source = "detection"
                    confidence_score = float(matched_detection["confidence_score"])
                else:
                    missed_frames += 1
                    if missed_frames > args.max_missed:
                        kalman = None
                        box_width = None
                        box_height = None
                        last_timestamp = timestamp_sec
                        continue
                    source = "prediction"
                    confidence_score = math.nan

            if kalman is None or box_width is None or box_height is None:
                continue

            center_x = float(kalman.x[0, 0])
            center_y = float(kalman.x[1, 0])
            x1, y1, x2, y2 = bbox_from_center(
                center_x=center_x,
                center_y=center_y,
                box_width=box_width,
                box_height=box_height,
                image_width=image_width,
                image_height=image_height,
            )

            rows.append(
                {
                    "video_name": video_name,
                    "frame_path": str(frame_path.resolve()),
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "sample_fps": sample_fps,
                    "track_id": track_id,
                    "source": source,
                    "confidence_score": confidence_score,
                    "missed_frames": missed_frames,
                    "center_x": center_x,
                    "center_y": center_y,
                    "velocity_x": float(kalman.x[2, 0]),
                    "velocity_y": float(kalman.x[3, 0]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "bbox_width": box_width,
                    "bbox_height": box_height,
                }
            )
            last_timestamp = timestamp_sec

    if not rows:
        raise RuntimeError("Tracking produced no rows.")

    output_df = pd.DataFrame(rows).sort_values(
        by=["video_name", "track_id", "frame_index"]
    )
    output_df.to_parquet(args.output_parquet, index=False)


if __name__ == "__main__":
    main()

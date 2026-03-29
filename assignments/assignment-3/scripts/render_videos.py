from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--keep-raw", action="store_true")
    return parser.parse_args()


def draw_overlay(
    frame: np.ndarray,
    frame_row: pd.Series,
    trail: list[tuple[int, int]],
) -> np.ndarray:
    annotated = frame.copy()
    x1 = int(round(float(frame_row["x1"])))
    y1 = int(round(float(frame_row["y1"])))
    x2 = int(round(float(frame_row["x2"])))
    y2 = int(round(float(frame_row["y2"])))
    center = (
        int(round(float(frame_row["center_x"]))),
        int(round(float(frame_row["center_y"]))),
    )

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.circle(annotated, center, radius=3, color=(0, 255, 255), thickness=-1)

    if len(trail) > 1:
        polyline = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [polyline], isClosed=False, color=(255, 200, 0), thickness=2)

    if frame_row["source"] == "detection":
        label = f"drone {float(frame_row['confidence_score']):.2f}"
    else:
        label = "drone predicted"

    cv2.putText(
        annotated,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    return annotated


def export_video_with_ffmpeg(raw_path: Path, final_path: Path, ffmpeg_bin: str) -> None:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(final_path),
    ]

    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        shutil.copy2(raw_path, final_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks_df = pd.read_parquet(args.tracks_parquet)

    for video_name, video_df in tracks_df.groupby("video_name"):
        video_df = video_df.sort_values(["frame_index", "track_id"]).reset_index(drop=True)
        first_frame = cv2.imread(str(video_df.iloc[0]["frame_path"]))
        if first_frame is None:
            continue

        height, width = first_frame.shape[:2]
        fps = float(video_df["sample_fps"].iloc[0])
        raw_path = args.output_dir / f"{video_name}_tracking_raw.mp4"
        final_path = args.output_dir / f"{video_name}_tracking.mp4"

        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        trails: dict[int, list[tuple[int, int]]] = {}

        for _, frame_row in video_df.iterrows():
            frame = cv2.imread(str(frame_row["frame_path"]))
            if frame is None:
                continue

            track_id = int(frame_row["track_id"])
            center = (
                int(round(float(frame_row["center_x"]))),
                int(round(float(frame_row["center_y"]))),
            )
            trails.setdefault(track_id, []).append(center)
            annotated = draw_overlay(frame=frame, frame_row=frame_row, trail=trails[track_id])
            writer.write(annotated)

        writer.release()
        export_video_with_ffmpeg(raw_path=raw_path, final_path=final_path, ffmpeg_bin=args.ffmpeg_bin)

        if not args.keep_raw and raw_path.exists():
            raw_path.unlink()


if __name__ == "__main__":
    main()

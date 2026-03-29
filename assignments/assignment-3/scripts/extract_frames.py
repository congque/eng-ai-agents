from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_videos(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix in VIDEO_EXTENSIONS)


def extract_video(
    video_path: Path,
    output_dir: Path,
    fps: float,
    ffmpeg_bin: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = output_dir / "frame_%06d.jpg"
    overwrite_flag = "-y" if overwrite else "-n"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite_flag,
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(frame_pattern),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = iter_videos(args.input_dir)

    if not videos:
        raise FileNotFoundError(f"No videos found in {args.input_dir}")

    for video_path in videos:
        extract_video(
            video_path=video_path,
            output_dir=args.output_dir / video_path.stem,
            fps=args.fps,
            ffmpeg_bin=args.ffmpeg_bin,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

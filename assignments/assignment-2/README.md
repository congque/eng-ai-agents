# Assignment 2 – Video Detections Index (Parquet)

Each row corresponds to one detection on one sampled video frame.

## Files
- `video_detections.parquet`: detections index (required)
- `retrieval_results.jsonl`: query → predicted label(s) → matched intervals (optional)
- `retrieval_intervals.csv`: flattened intervals table (optional)

## Schema (video_detections.parquet)
- `video_id` (string): YouTube id
- `frame_index` (int): 0-based index over sampled frames
- `timestamp_sec` (int/float): frame timestamp in seconds (frame_index * 5 if sampled every 5s)
- `class_label` (string): predicted part label
- `bounding_box` (list[float]): [x_min, y_min, x_max, y_max]
- `confidence_score` (float): detection confidence

## Notes
- Frames sampled every 5 seconds using ffmpeg (fps=1/5).
- Detector: YOLOv8-seg fine-tuned on carparts-seg.
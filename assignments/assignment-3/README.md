# Assignment 3 - UAV Drone Detection and Tracking

This folder contains the scripts used for Assignment 3:

1. Extract frames from input videos.
2. Run a single-class drone detector.
3. Track the drone center with a Kalman filter.
4. Render an output video for each input video.
5. Export detection frames as a Hugging Face dataset and a Parquet file.

## Setup

Inside the course devcontainer:

```bash
make start
source .venv/bin/activate
uv pip install -r assignments/assignment-3/requirements.txt
```

If you are not using the devcontainer, make sure `ffmpeg` is available and the packages in `requirements.txt` are installed.

## Train a detector

The detector used for the submission was trained on the public Hugging Face dataset `pathikg/drone-detection-dataset`.

Prepare the dataset:

```bash
python assignments/assignment-3/scripts/prepare_hf_drone_dataset.py \
  --dataset-id pathikg/drone-detection-dataset \
  --output-dir assignments/assignment-3/data/pathikg_yolo
```

Train from scratch:

```bash
yolo detect train \
  data=assignments/assignment-3/data/pathikg_yolo/data.yaml \
  model=yolo11n.yaml \
  pretrained=False \
  imgsz=640 \
  epochs=50 \
  batch=64 \
  device=0 \
  workers=8 \
  fraction=0.35 \
  project=assignments/assignment-3/runs \
  name=pathikg_yolo11n_scratch_e50_f35_bs64
```

Copy the best checkpoint:

```bash
cp assignments/assignment-3/runs/pathikg_yolo11n_scratch_e50_f35_bs64/weights/best.pt \
  assignments/assignment-3/models/drone_best.pt
```

## Workflow

Put your `.mp4` files in `assignments/assignment-3/videos/`.

Extract frames:

```bash
python assignments/assignment-3/scripts/extract_frames.py \
  --input-dir assignments/assignment-3/videos \
  --output-dir assignments/assignment-3/frames \
  --fps 5
```

Run the detector:

```bash
python assignments/assignment-3/scripts/detect_videos.py \
  --frames-dir assignments/assignment-3/frames \
  --model assignments/assignment-3/models/drone_best.pt \
  --detections-dir assignments/assignment-3/detections \
  --output-parquet assignments/assignment-3/out/detections.parquet \
  --class-names drone uav \
  --sample-fps 5 \
  --conf 0.25
```

Track the drone:

```bash
python assignments/assignment-3/scripts/track_videos.py \
  --frames-dir assignments/assignment-3/frames \
  --detections-parquet assignments/assignment-3/out/detections.parquet \
  --output-parquet assignments/assignment-3/out/tracks.parquet \
  --max-missed 8 \
  --association-distance 150
```

Render output videos:

```bash
python assignments/assignment-3/scripts/render_videos.py \
  --tracks-parquet assignments/assignment-3/out/tracks.parquet \
  --output-dir assignments/assignment-3/out/videos
```

Export the detection frames as a Hugging Face dataset:

```bash
python assignments/assignment-3/scripts/export_hf_dataset.py \
  --detections-parquet assignments/assignment-3/out/detections.parquet \
  --output-dir assignments/assignment-3/hf_dataset \
  --parquet-path assignments/assignment-3/out/detection_frames.parquet
```

## Notes

- If your detector label is not exactly `drone`, pass the names with `--class-names`.
- The tracker is single-target.
- Short gaps are handled by prediction-only steps. Long gaps start a new track.

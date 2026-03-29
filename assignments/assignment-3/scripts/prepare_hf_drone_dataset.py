from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="pathikg/drone-detection-dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--class-name", default="drone")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_yolo_line(
    bbox: list[float],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> str:
    x, y, width, height = bbox
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    return (
        f"{class_id} "
        f"{center_x / image_width:.6f} "
        f"{center_y / image_height:.6f} "
        f"{width / image_width:.6f} "
        f"{height / image_height:.6f}"
    )


def export_split(
    dataset_id: str,
    split_name: str,
    output_dir: Path,
    limit: int | None,
) -> dict[str, int]:
    image_dir = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name
    ensure_dir(image_dir)
    ensure_dir(label_dir)

    split = load_dataset(dataset_id, split=split_name)
    if limit is not None:
        split = split.select(range(min(limit, len(split))))

    image_count = 0
    box_count = 0

    for row in split:
        image = row["image"]
        image_width = int(row["width"])
        image_height = int(row["height"])
        image_id = int(row["image_id"])
        file_stem = f"{split_name}_{image_id:06d}"

        image_path = image_dir / f"{file_stem}.jpg"
        label_path = label_dir / f"{file_stem}.txt"

        image.save(image_path)

        boxes = row["objects"]["bbox"]
        label_lines = [to_yolo_line(bbox, image_width, image_height) for bbox in boxes]
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        image_count += 1
        box_count += len(boxes)

    return {
        "images": image_count,
        "boxes": box_count,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    train_stats = export_split(
        dataset_id=args.dataset_id,
        split_name=args.train_split,
        output_dir=args.output_dir,
        limit=args.limit_train,
    )
    val_stats = export_split(
        dataset_id=args.dataset_id,
        split_name=args.val_split,
        output_dir=args.output_dir,
        limit=args.limit_val,
    )

    data_yaml = {
        "path": str(args.output_dir.resolve()),
        "train": f"images/{args.train_split}",
        "val": f"images/{args.val_split}",
        "names": {0: args.class_name},
    }
    (args.output_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False),
        encoding="utf-8",
    )

    summary = {
        "dataset_id": args.dataset_id,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "class_name": args.class_name,
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

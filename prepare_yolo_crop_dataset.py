#!/usr/bin/env python3
"""Create a reusable classifier dataset from YOLO bboxes without changing the source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from train_yolo_crop_classifier import (
    IMAGE_SUFFIXES,
    label_path_for,
    load_yaml,
    read_image_list,
    resolve_path,
    split_images,
)


def class_folder(class_id: int, name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or f"class_{class_id}"
    return f"{class_id:03d}_{safe_name}"


def reset_prepared_output(output_root: Path, source_root: Path) -> None:
    """Remove only a validated prepared dataset directory."""
    resolved_output = output_root.resolve()
    resolved_source = source_root.resolve()
    protected = {Path("/"), Path.home().resolve(), resolved_source, resolved_source.parent}
    if resolved_output in protected or len(resolved_output.parts) < 4:
        raise ValueError(f"Refusing to remove unsafe prepared output path: {resolved_output}")
    if resolved_source in resolved_output.parents:
        raise ValueError(f"Prepared output must be outside the source dataset: {resolved_source}")
    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise ValueError(f"Prepared output exists but is not a directory: {resolved_output}")
        print(f"Removing existing prepared dataset: {resolved_output}", flush=True)
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=False)


def prepare_image(task: tuple[Any, ...]) -> list[tuple[str, int, str, str]]:
    image_path_text, split, output_root_text, names, expansion, min_pixels, jpeg_quality = task
    image_path = Path(image_path_text)
    output_root = Path(output_root_text)
    label_path = label_path_for(image_path)
    if not label_path.is_file():
        return []
    labels: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        if 0 <= class_id < len(names):
            x_center, y_center, box_width, box_height = map(float, fields[1:5])
            if box_width > 0 and box_height > 0:
                labels.append((class_id, x_center, y_center, box_width, box_height))
    if not labels:
        return []

    source_key = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    records: list[tuple[str, int, str, str]] = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        for box_index, (class_id, x_center, y_center, box_width, box_height) in enumerate(labels):
            crop_width = max(box_width * width * expansion, min_pixels)
            crop_height = max(box_height * height * expansion, min_pixels)
            center_x, center_y = x_center * width, y_center * height
            left = max(0, math.floor(center_x - crop_width / 2))
            top = max(0, math.floor(center_y - crop_height / 2))
            right = min(width, math.ceil(center_x + crop_width / 2))
            bottom = min(height, math.ceil(center_y + crop_height / 2))
            if right <= left or bottom <= top:
                continue
            destination = (
                output_root
                / split
                / class_folder(class_id, names[class_id])
                / f"{source_key}_{image_path.stem}_box{box_index:03d}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                image.crop((left, top, right, bottom)).save(destination, "JPEG", quality=jpeg_quality)
            records.append((split, class_id, str(image_path), str(destination)))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional prepared-root override; never changes YAML")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-images-per-split", type=int, default=0, help="Smoke-test only; zero means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config.resolve())
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["training"]
    data_yaml_path = resolve_path(data_cfg["yaml"], repository_root, repository_root)
    data_yaml = load_yaml(data_yaml_path)
    source_root = Path(data_cfg["local_test_path"]) if args.local_test else Path(data_yaml["path"])
    prepared_key = "prepared_local_test_path" if args.local_test else "prepared_server_path"
    output_root = (args.output or Path(data_cfg[prepared_key])).resolve()
    if source_root.resolve() == output_root or source_root.resolve() in output_root.parents:
        raise ValueError(f"Prepared output must be outside the source dataset: {source_root}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    reset_prepared_output(output_root, source_root)

    names_value = data_yaml["names"]
    names = list(names_value.values()) if isinstance(names_value, dict) else list(names_value)
    if len(names) != int(model_cfg["num_classes"]):
        raise ValueError("Data YAML names and model.num_classes disagree")
    train_images = read_image_list(data_yaml.get("train"), data_yaml_path.parent, source_root)
    val_images = read_image_list(data_yaml.get("val"), data_yaml_path.parent, source_root)
    if args.local_test and (not train_images or not val_images):
        all_images = sorted(path for path in (source_root / "images").rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        train_images, val_images = split_images(
            all_images, float(data_cfg["val_fraction_if_lists_missing"]), int(train_cfg["seed"])
        )
        print("Local split lists unavailable; using the deterministic image split.", flush=True)
    if not train_images or not val_images:
        raise RuntimeError("No train/validation images resolved from the data YAML")
    if args.max_images_per_split > 0:
        train_images = train_images[: args.max_images_per_split]
        val_images = val_images[: args.max_images_per_split]

    # Create every class in both splits so ImageFolder mappings are stable even
    # when a small test subset contains no instances of a class.
    for split in ("train", "val"):
        for class_id, name in enumerate(names):
            (output_root / split / class_folder(class_id, name)).mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(image_path),
            split,
            str(output_root),
            names,
            float(data_cfg["crop_expansion"]),
            int(data_cfg["min_crop_pixels"]),
            int(args.jpeg_quality),
        )
        for split, images in (("train", train_images), ("val", val_images))
        for image_path in images
    ]
    print(
        f"Preparing crops from {len(train_images)} train and {len(val_images)} val images into {output_root} "
        f"with {args.workers} workers. Source remains read-only.",
        flush=True,
    )
    manifest_path = output_root / "manifest.csv"
    counts = {"train": 0, "val": 0}
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["split", "class_id", "source_image", "crop_image"])
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for completed, records in enumerate(executor.map(prepare_image, tasks, chunksize=16), 1):
                writer.writerows(records)
                for split, _, _, _ in records:
                    counts[split] += 1
                if completed % 1000 == 0 or completed == len(tasks):
                    print(
                        f"Prepared {completed}/{len(tasks)} source images "
                        f"({counts['train']} train, {counts['val']} val crops).",
                        flush=True,
                    )
    print(
        f"Prepared dataset complete: train={counts['train']} crops, val={counts['val']} crops, "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

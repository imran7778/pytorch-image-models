#!/usr/bin/env python3
"""Train a timm classifier from a prepared YOLO bbox crop dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

import timm
from timm.data import create_transform


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_path(value: str, yaml_dir: Path, dataset_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    options = (yaml_dir / candidate, dataset_root / candidate)
    return next((item.resolve() for item in options if item.exists()), options[0].resolve())


def read_image_list(
    split_value: str | list[str] | None,
    yaml_dir: Path,
    dataset_root: Path,
) -> list[Path]:
    values = [split_value] if isinstance(split_value, str) else list(split_value or [])
    images: list[Path] = []
    for value in values:
        source = resolve_path(str(value), yaml_dir, dataset_root)
        if source.is_dir():
            images.extend(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = Path(line)
                if not item.is_absolute():
                    root_candidate = dataset_root / item
                    item = root_candidate if root_candidate.exists() else source.parent / item
                images.append(item.resolve())
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source)
    return sorted(set(images))


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def build_samples(images: list[Path], num_classes: int, min_crop_pixels: int) -> list[tuple[Path, int, tuple[float, ...]]]:
    samples: list[tuple[Path, int, tuple[float, ...]]] = []
    for image_path in images:
        label_path = label_path_for(image_path)
        if not label_path.is_file():
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                class_id = int(float(fields[0]))
                x_center, y_center, width, height = map(float, fields[1:5])
            except ValueError as error:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from error
            if 0 <= class_id < num_classes and width > 0 and height > 0:
                samples.append((image_path, class_id, (x_center, y_center, width, height, float(min_crop_pixels))))
    return samples


def split_images(images: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction_if_lists_missing must be between 0 and 1")
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    boundary = max(1, round(len(shuffled) * (1 - val_fraction)))
    return sorted(shuffled[:boundary]), sorted(shuffled[boundary:])


def limit_samples(samples: list[Any], maximum: int) -> list[Any]:
    return samples if maximum <= 0 else samples[:maximum]


def limit_imagefolder(dataset: ImageFolder, maximum: int, seed: int) -> None:
    """Apply a deterministic, class-order-independent sample limit in place."""
    if maximum <= 0 or maximum >= len(dataset.samples):
        return
    selected = dataset.samples.copy()
    random.Random(seed).shuffle(selected)
    dataset.samples = selected[:maximum]
    dataset.imgs = dataset.samples
    dataset.targets = [target for _, target in dataset.samples]


def parse_sample_limit(value: Any, setting_name: str) -> int:
    """Parse a sample cap; zero and common unlimited aliases mean all samples."""
    if value is None:
        return 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "all", "none", "null", "unlimited"}:
            return 0
    try:
        limit = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{setting_name} must be a positive integer, 0, or 'all'; received {value!r}"
        ) from error
    if limit < 0:
        raise ValueError(f"{setting_name} cannot be negative; received {limit}")
    return limit


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    *,
    epoch: int,
    epochs: int,
    phase: str,
    verbose: bool,
    num_classes: int,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    started = time.perf_counter()
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{epochs} {phase}",
        unit="batch",
        dynamic_ncols=True,
        disable=not verbose,
    )
    for inputs, targets in progress:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if training:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * targets.numel()
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_count += targets.numel()
        encoded = targets.detach().cpu() * num_classes + predictions.detach().cpu()
        confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        elapsed = max(time.perf_counter() - started, 1e-9)
        postfix = {
            "loss": f"{total_loss / total_count:.4f}",
            "acc": f"{total_correct / total_count:.4f}",
            "samples/s": f"{total_count / elapsed:.1f}",
        }
        if training:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        progress.set_postfix(postfix)
    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(support > 0, true_positive / support, 0.0)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "balanced_accuracy": recall[present].mean().item() if present.any() else 0.0,
        "macro_precision": precision[present].mean().item() if present.any() else 0.0,
        "macro_recall": recall[present].mean().item() if present.any() else 0.0,
        "macro_f1": f1[present].mean().item() if present.any() else 0.0,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.to(torch.int64).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def display_class_name(folder_name: str) -> str:
    return folder_name.split("_", 1)[1] if "_" in folder_name else folder_name


def print_epoch_metrics(
    epoch: int,
    epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    class_names: list[str],
) -> None:
    print(f"\nEpoch {epoch}/{epochs} summary", flush=True)
    print(f"{'metric':<22}{'train':>12}{'val':>12}", flush=True)
    print("-" * 46, flush=True)
    for key, label in (
        ("loss", "loss"),
        ("accuracy", "accuracy"),
        ("balanced_accuracy", "balanced accuracy"),
        ("macro_precision", "macro precision"),
        ("macro_recall", "macro recall"),
        ("macro_f1", "macro F1"),
    ):
        print(f"{label:<22}{train_metrics[key]:>12.4f}{val_metrics[key]:>12.4f}", flush=True)
    print("\nPer-class metrics", flush=True)
    print(
        f"{'class':<14}{'split':<8}{'precision':>11}{'recall':>10}{'F1':>10}{'support':>10}",
        flush=True,
    )
    print("-" * 63, flush=True)
    for class_id, class_name in enumerate(class_names):
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            print(
                f"{class_name:<14}{split:<8}{metrics['precision'][class_id]:>11.4f}"
                f"{metrics['recall'][class_id]:>10.4f}{metrics['f1'][class_id]:>10.4f}"
                f"{metrics['support'][class_id]:>10d}",
                flush=True,
            )


def flatten_epoch_metrics(
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"):
            row[f"{split}_{key}"] = metrics[key]
        for class_id, class_name in enumerate(class_names):
            safe_name = class_name.lower().replace(" ", "_")
            for key in ("precision", "recall", "f1", "support"):
                row[f"{split}_{safe_name}_{key}"] = metrics[key][class_id]
        row[f"{split}_confusion_matrix"] = json.dumps(metrics["confusion_matrix"], separators=(",", ":"))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["training"]
    repository_root = Path(__file__).resolve().parent
    prepared_key = "prepared_local_test_path" if args.local_test else "prepared_server_path"
    dataset_root = Path(data_cfg[prepared_key])
    if not (dataset_root / "train").is_dir() or not (dataset_root / "val").is_dir():
        raise FileNotFoundError(
            f"Prepared crop dataset is missing at {dataset_root}. Run prepare_yolo_crop_dataset.py first."
        )

    seed = int(train_cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    num_classes = int(model_cfg["num_classes"])
    input_size = int(model_cfg["input_size"])
    train_transform = create_transform(input_size=input_size, is_training=True)
    val_transform = create_transform(input_size=input_size, is_training=False)
    train_dataset = ImageFolder(dataset_root / "train", transform=train_transform, allow_empty=True)
    val_dataset = ImageFolder(dataset_root / "val", transform=val_transform, allow_empty=True)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise RuntimeError("Prepared train and val class folders do not match")
    if len(train_dataset.classes) != num_classes:
        raise RuntimeError(
            f"Prepared dataset has {len(train_dataset.classes)} classes, but model.num_classes={num_classes}"
        )
    class_names = [display_class_name(name) for name in train_dataset.classes]
    max_train_samples = parse_sample_limit(
        args.max_train_samples
        if args.max_train_samples is not None
        else train_cfg.get("max-train-samples", train_cfg.get("max_train_samples", 0)),
        "training.max-train-samples",
    )
    max_val_samples = parse_sample_limit(
        args.max_val_samples
        if args.max_val_samples is not None
        else train_cfg.get("max-val-samples", train_cfg.get("max_val_samples", 0)),
        "training.max-val-samples",
    )
    limit_imagefolder(train_dataset, max_train_samples, seed)
    limit_imagefolder(val_dataset, max_val_samples, seed + 1)
    workers = int(train_cfg["workers"])
    loader_args = dict(
        batch_size=int(train_cfg["batch_size"]),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    requested_device = str(train_cfg["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    device = torch.device(requested_device)
    model = timm.create_model(
        str(model_cfg["name"]),
        pretrained=bool(model_cfg["pretrained"]),
        num_classes=num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(args.epochs or train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda" and bool(train_cfg["amp"])
    # ConvNeXt V2 trained from scratch can overflow PyTorch's 65536 default
    # scale on its first updates. A conservative initial scale avoids skipped
    # optimizer steps, especially in one-batch smoke tests.
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0) if use_amp else None
    output_dir = (repository_root / train_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    verbose = bool(train_cfg.get("verbose", True))

    print(
        f"Training {model_cfg['name']} on {len(train_dataset)} train and {len(val_dataset)} val prepared bbox crops; "
        f"dataset={dataset_root}, device={device}.",
        flush=True,
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = None
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                epoch=epoch,
                epochs=epochs,
                phase="train",
                verbose=verbose,
                num_classes=num_classes,
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    epoch=epoch,
                    epochs=epochs,
                    phase="val",
                    verbose=verbose,
                    num_classes=num_classes,
                )
            scheduler.step()
            print_epoch_metrics(epoch, epochs, train_metrics, val_metrics, class_names)
            row = flatten_epoch_metrics(epoch, train_metrics, val_metrics, class_names)
            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()
            confusion_payload = {
                "epoch": epoch,
                "classes": class_names,
                "rows": "ground_truth",
                "columns": "prediction",
                "train": train_metrics["confusion_matrix"],
                "val": val_metrics["confusion_matrix"],
            }
            with (confusion_dir / f"epoch-{epoch:03d}.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(confusion_payload, handle, sort_keys=False)
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}
            torch.save(state, output_dir / "last.pt")
            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                torch.save(state, output_dir / "best.pt")
            if epoch % int(train_cfg["save_every"]) == 0:
                torch.save(state, output_dir / f"epoch-{epoch}.pt")
            print(f"Best-checkpoint metric: validation macro F1={val_metrics['macro_f1']:.4f}\n", flush=True)


if __name__ == "__main__":
    main()

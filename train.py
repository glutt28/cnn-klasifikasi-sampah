#!/usr/bin/env python3
"""Fine-tune UniRepLKNet-S on the five Studio 2 waste classes."""
from __future__ import annotations

import argparse
import csv
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from submission.model import CLASS_NAMES, IMAGE_SIZE, get_transform
from submission.unireplknet import unireplknet_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("datasets/split/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("datasets/split/val"))
    parser.add_argument("--output", type=Path, default=Path("submission/model.pth"))
    parser.add_argument("--history", type=Path, default=Path("outputs/training_history.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=6, help="Early-stopping patience; 0 disables it")
    parser.add_argument("--class-weighting", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from random initialization")
    parser.add_argument(
        "--image-size", type=int, default=IMAGE_SIZE,
        help=f"Input resolution for training/validation (default {IMAGE_SIZE}, from submission/model.py). "
             "Evaluation always uses IMAGE_SIZE in submission/model.py, so edit that constant to match.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Enable cudnn/MIOpen benchmark mode. On ROCm this triggers an exhaustive kernel search "
             "for every new tensor shape, which can take hours. Off by default.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Stronger augmentation for improved robustness while preserving material and shape cues."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.80, 1.20),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
            # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            # transforms.RandomGrayscale(p=0.05), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ]
    )


def load_datasets(train_dir: Path, val_dir: Path, image_size: int = IMAGE_SIZE):
    train_dir, val_dir = train_dir.resolve(), val_dir.resolve()
    for name, path in (("training", train_dir), ("validation", val_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"{name.capitalize()} directory not found: {path}\nRun python scripts/setup_dataset.py first.")

    train_dataset = ImageFolder(train_dir, transform=training_transform(image_size))
    val_dataset = ImageFolder(val_dir, transform=get_transform(image_size))
    for name, dataset in (("training", train_dataset), ("validation", val_dataset)):
        if tuple(dataset.classes) != CLASS_NAMES:
            raise ValueError(
                f"Unexpected classes in {name} set.\nExpected: {list(CLASS_NAMES)}\nFound: {dataset.classes}"
            )
        if not dataset:
            raise ValueError(f"No images found in {name} directory: {dataset.root}")
    return train_dataset, val_dataset


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, device: torch.device, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Construct the backbone; pretrained weights are fetched only during training."""
    model = unireplknet_s(
        num_classes=1000 if pretrained else num_classes,
        in_1k_pretrained=pretrained,
        drop_path_rate=0.1,
        attempt_use_lk_impl=False,
    )
    if pretrained:
        model.head = nn.Linear(model.head.in_features, num_classes)
        nn.init.trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)
    return model


def class_weights(dataset: ImageFolder, device: torch.device) -> torch.Tensor:
    counts = np.bincount(dataset.targets, minlength=len(CLASS_NAMES)).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"At least one class has no training images. Counts: {counts.tolist()}")
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def amp_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def run_epoch(
    model, loader, criterion, device, optimizer=None, scaler=None, *, description="Train"
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_seen: list[int] = []
    predictions_seen: list[int] = []

    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.inference_mode()
        with grad_context:
            with amp_context(device):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.detach().item() * labels.size(0)
        labels_seen.extend(labels.detach().cpu().tolist())
        predictions_seen.extend(logits.detach().argmax(1).cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / len(labels_seen):.4f}")

    average_loss = total_loss / len(labels_seen)
    macro_f1 = f1_score(
        labels_seen, predictions_seen, labels=range(len(CLASS_NAMES)),
        average="macro", zero_division=0,
    )
    accuracy = accuracy_score(labels_seen, predictions_seen)
    return average_loss, macro_f1, accuracy


def save_state_dict(model: nn.Module, path: Path) -> None:
    """Save only a CPU state_dict, as required by evaluate.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("--epochs and --batch-size must be positive; --workers cannot be negative")
    if args.patience < 0:
        raise ValueError("--patience cannot be negative")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = args.benchmark and torch.cuda.is_available()

    train_dataset, val_dataset = load_datasets(args.train_dir, args.val_dir, args.image_size)
    size_mismatch = args.image_size != IMAGE_SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(train_dataset, args.batch_size, True, args.workers, device, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.workers, device, args.seed)

    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Train images : {len(train_dataset)}")
    print(f"Val images   : {len(val_dataset)}")
    print(f"Class order  : {list(CLASS_NAMES)}")
    print(f"Pretrained   : {not args.no_pretrained}")
    print(f"Image size   : {args.image_size}")
    print(f"Benchmark    : {torch.backends.cudnn.benchmark}")
    if size_mismatch:
        print(f"WARNING: --image-size {args.image_size} != IMAGE_SIZE={IMAGE_SIZE} in submission/model.py. "
              "Set IMAGE_SIZE in model.py to the same value before evaluating or submitting.")

    model = build_model(len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    weights = class_weights(train_dataset, device) if args.class_weighting == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)

    head_parameter_ids = {id(parameter) for parameter in model.head.parameters()}
    backbone_parameters = [p for p in model.parameters() if id(p) not in head_parameter_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_f1, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler,
            description=f"Train {epoch:02d}/{args.epochs}",
        )
        val_loss, val_f1, val_acc = run_epoch(
            model, val_loader, criterion, device,
            description=f"Val   {epoch:02d}/{args.epochs}",
        )
        scheduler.step()
        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} F1={train_f1:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} F1={val_f1:.4f} acc={val_acc:.4f} | {elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
            "seconds": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1, best_epoch, stale_epochs = val_f1, epoch, 0
            save_state_dict(model, args.output)
            print(f"  Best checkpoint saved: {args.output} (Macro-F1={best_f1:.4f})")
        else:
            stale_epochs += 1

        if args.patience and stale_epochs >= args.patience:
            print(f"Early stopping: validation Macro-F1 did not improve for {args.patience} epochs.")
            break

    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print(f"\nBest epoch                  : {best_epoch}")
    print(f"Best local validation F1    : {best_f1:.4f}")
    print(f"Model state_dict            : {args.output}")
    print(f"Training history            : {args.history}")
    print(f"Trained at image size       : {args.image_size}")
    if size_mismatch:
        print(f"!! Set IMAGE_SIZE = {args.image_size} in submission/model.py, otherwise evaluation uses {IMAGE_SIZE}.")


if __name__ == "__main__":
    main()

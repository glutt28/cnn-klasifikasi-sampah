#!/usr/bin/env python3
"""Balance the TRAIN split by dropping the most suspicious images instead of random ones.

How it works
1. Embed every training image with the ImageNet-pretrained UniRepLKNet-S (no fine-tuning,
   so the scores cannot memorise your labels).
2. Score each image: how many of its k nearest neighbours (cosine) share its label
   ("agreement"), tie-broken by similarity to its class centroid.
3. For every class larger than the target size, drop the lowest-scoring images.

Only train/ is pruned. val/ and test/ are linked unchanged, so evaluation stays honest.
Always read the report CSV: "suspicious" means "unlike its neighbours", which can be a
wrong label (good to drop) or a legitimately hard example (not necessarily good to drop).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Pure numpy part (no torch needed; unit-testable)
# ----------------------------------------------------------------------------
def score_samples(features: np.ndarray, labels: np.ndarray, k: int = 10):
    """Return (agreement, centroid_similarity); LOWER = more suspicious.

    features: (N, D) L2-normalised embeddings. labels: (N,) integer class ids.
    """
    n = len(labels)
    if n < 2:
        raise ValueError("Need at least 2 samples to score.")
    k = max(1, min(k, n - 1))

    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)  # never count an image as its own neighbour
    neighbours = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    agreement = (labels[neighbours] == labels[:, None]).mean(axis=1)

    centroid_similarity = np.zeros(n, dtype=np.float64)
    for cls in np.unique(labels):
        mask = labels == cls
        centroid = features[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        centroid_similarity[mask] = features[mask] @ centroid
    return agreement, centroid_similarity


def select_drops(agreement, centroid_similarity, labels, target: int) -> np.ndarray:
    """Boolean mask of images to drop so that no class exceeds `target`."""
    drop = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        excess = len(indices) - target
        if excess <= 0:
            continue
        # lexsort: last key is primary -> sort by agreement, then centroid similarity (ascending)
        order = np.lexsort((centroid_similarity[indices], agreement[indices]))
        drop[indices[order[:excess]]] = True
    return drop


# ----------------------------------------------------------------------------
# Torch part (imported lazily)
# ----------------------------------------------------------------------------
def extract_features(train_dir: Path, batch_size: int, workers: int):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    sys.path.insert(0, str(PROJECT_ROOT))
    from submission.model import CLASS_NAMES, get_transform  # noqa: E402
    from submission.unireplknet import unireplknet_s  # noqa: E402

    dataset = ImageFolder(train_dir, transform=get_transform())
    if dataset.classes != list(CLASS_NAMES):
        raise ValueError(f"Class folders {dataset.classes} do not match {list(CLASS_NAMES)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unireplknet_s(num_classes=1000, in_1k_pretrained=True, attempt_use_lk_impl=False)
    model.head = nn.Identity()  # forward() then returns the pooled, normalised embedding
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    chunks = []
    with torch.inference_mode():
        for done, (images, _) in enumerate(loader, start=1):
            chunks.append(model(images.to(device)).float().cpu().numpy())
            print(f"\rEmbedding {min(done * batch_size, len(dataset))}/{len(dataset)}", end="", flush=True)
    print()

    features = np.concatenate(chunks)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    paths = [Path(path) for path, _ in dataset.samples]
    labels = np.array([label for _, label in dataset.samples])
    return features, labels, paths, dataset.classes


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source_dir: Path, destination_dir: Path) -> int:
    count = 0
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            link_or_copy(path, destination_dir / path.relative_to(source_dir))
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, required=True, help="Existing split dir containing train/ (and val/, test/)")
    parser.add_argument("--output", type=Path, help="New split dir to create (required unless --dry-run)")
    parser.add_argument("--target", type=int, help="Max train images per class (default: size of the smallest class)")
    parser.add_argument("--k", type=int, default=10, help="Neighbours used for label agreement (default 10)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path("outputs/pruning_report.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Only write the report; create no dataset")
    parser.add_argument("--force", action="store_true", help="Replace --output if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split.resolve()
    train_dir = split / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"Missing {train_dir}")
    if not args.dry_run:
        if args.output is None:
            raise SystemExit("--output is required unless --dry-run")
        output = args.output.resolve()
        if output == split or output.is_relative_to(split) or split.is_relative_to(output):
            raise SystemExit("--output must not overlap --split")
        if output.exists():
            if not args.force:
                raise SystemExit(f"{output} exists; use --force to replace it")
            if output.is_symlink():
                raise SystemExit("Refusing to remove a symlink")
            shutil.rmtree(output)

    features, labels, paths, classes = extract_features(train_dir, args.batch_size, args.workers)
    counts = np.bincount(labels, minlength=len(classes))
    target = args.target if args.target is not None else int(counts.min())
    if target < 1:
        raise SystemExit("--target must be >= 1")

    agreement, centroid_similarity = score_samples(features, labels, args.k)
    drop = select_drops(agreement, centroid_similarity, labels, target)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((centroid_similarity, agreement))  # most suspicious first
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "class", "knn_agreement", "centroid_similarity", "dropped"])
        for i in order:
            writer.writerow([paths[i], classes[labels[i]], f"{agreement[i]:.2f}", f"{centroid_similarity[i]:.4f}", int(drop[i])])

    print(f"{'Class':<18} {'Before':>7} {'After':>7} {'Dropped':>8} {'Mean agree.':>12}")
    for cls_id, name in enumerate(classes):
        mask = labels == cls_id
        print(f"{name:<18} {counts[cls_id]:>7} {int(mask.sum() - drop[mask].sum()):>7} "
              f"{int(drop[mask].sum()):>8} {agreement[mask].mean():>12.3f}")
    low = int((agreement < 0.5).sum())
    print(f"\nImages with <50% neighbour agreement (possible label noise): {low} of {len(labels)}")
    print(f"Report: {args.report}  (open the top rows and check the labels by eye)")

    if args.dry_run:
        print("Dry run: no dataset created.")
        return

    kept = 0
    for i, path in enumerate(paths):
        if not drop[i]:
            link_or_copy(path, output / "train" / path.relative_to(train_dir))
            kept += 1
    linked = sum(link_tree(split / name, output / name) for name in ("val", "test") if (split / name).is_dir())
    manifest = {
        "derived_from": str(split), "target_per_class": target, "k": args.k,
        "train_before": int(len(labels)), "train_after": kept, "val_test_files_linked_unchanged": linked,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {output}: train {len(labels)} -> {kept}; val/test linked unchanged ({linked} files).")


if __name__ == "__main__":
    main()

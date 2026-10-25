#!/usr/bin/env python3
"""Prepare a deterministic train/validation split from the downloaded course dataset.

Expected source layout: one directory containing the five class directories.
The script does not read train_sources.csv or download individual images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

CLASS_NAMES = (
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git", ".venv", ".venv-rocm-official", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", "split", "outputs", "checkpoints",
}


@dataclass(frozen=True)
class ImageRecord:
    label: str
    path: Path
    relative_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("datasets"),
        help="Directory for downloaded/extracted data (default: datasets)",
    )
    parser.add_argument(
        "--source", type=Path,
        help="Directory containing the five class folders; detected automatically if omitted",
    )
    parser.add_argument(
        "--archive", type=Path,
        help="Local ZIP downloaded from course Google Drive; extracted into datasets/raw/extracted",
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/split"))
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation fraction per class (0.05-0.50)")
    parser.add_argument(
        "--test-size", type=float, default=0.0,
        help="Held-out test fraction per class (0 = no test split, otherwise 0.05-0.40). "
             "Carved out of what would otherwise be the training set, so the validation set is unchanged.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Use at most N images per class (random, seeded). Handy for quick experiments.",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Undersample every class to the size of the smallest class (after --dedupe). "
             "Applied before the split, so train/val/test are all balanced. Discards data.",
    )
    parser.add_argument(
        "--dedupe", action="store_true",
        help="Drop byte-identical duplicate images before splitting (prevents train/val/test leakage).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--force", action="store_true", help="Replace only the generated split directory")
    return parser.parse_args()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_class_root(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_dir() for name in CLASS_NAMES)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting paths that escape the destination."""
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a valid ZIP archive: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if not is_within(target, base):
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        zipped.extractall(destination)


def extract_archive(archive: Path, datasets_dir: Path) -> Path:
    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {archive}")

    destination = datasets_dir / "raw" / "extracted"
    if destination.exists() and any(destination.iterdir()):
        if discover_class_roots(destination, destination / "__never_output__"):
            print(f"Using existing extracted dataset: {destination}")
            return destination
        raise FileExistsError(
            f"Extraction directory already exists but no valid class folders were found: {destination}\n"
            "Move/remove that directory yourself, or extract the ZIP to a different location first."
        )

    print(f"Extracting {archive} -> {destination}")
    safe_extract_zip(archive, destination)
    return destination


def discover_class_roots(root: Path, output_dir: Path) -> list[Path]:
    """Find directories whose direct children are exactly the required class folder names."""
    found: list[Path] = []
    if not root.is_dir():
        return found

    for current_text, subdirs, _ in os.walk(root):
        current = Path(current_text)
        subdirs[:] = [
            name for name in subdirs
            if name not in SKIP_DIRS
            and not is_within(current / name, output_dir)
            and not (name.startswith(".") and name != ".")
        ]
        if is_class_root(current) and not is_within(current, output_dir):
            found.append(current.resolve())
            # Class folders themselves are data, not alternate dataset roots.
            subdirs[:] = [name for name in subdirs if name not in CLASS_NAMES]

    # Preserve order while removing duplicate paths.
    return list(dict.fromkeys(found))


def find_class_root(source: Path | None, datasets_dir: Path, output_dir: Path) -> Path:
    if source is not None:
        explicit = source.expanduser().resolve()
        if not is_class_root(explicit):
            raise ValueError(
                f"--source must contain these class folders directly: {explicit}\n"
                f"Expected: {', '.join(CLASS_NAMES)}"
            )
        if is_within(explicit, output_dir) or is_within(output_dir, explicit):
            raise ValueError("Dataset source and split output must not contain/overlap each other.")
        return explicit

    roots = discover_class_roots(datasets_dir, output_dir)
    if not roots:
        raise FileNotFoundError(
            "Could not find a directory containing all five class folders.\n"
            f"Expected direct subfolders: {', '.join(CLASS_NAMES)}\n\n"
            "Download the curated ZIP from the course Google Drive, then run:\n"
            "  python scripts/setup_dataset.py --archive datasets/course_dataset.zip\n"
            "Or extract it manually and pass the directory with --source."
        )
    if len(roots) > 1:
        listed = "\n".join(f"  - {path}" for path in roots)
        raise ValueError(
            "More than one dataset root was found; refusing to choose a dataset silently:\n"
            f"{listed}\n\nUse --source with the intended course Drive dataset."
        )
    return roots[0]


def collect_records(source: Path) -> tuple[list[ImageRecord], dict[str, int], list[str]]:
    records: list[ImageRecord] = []
    counts: dict[str, int] = {}
    invalid: list[str] = []

    for label in CLASS_NAMES:
        class_dir = source / label
        valid_count = 0
        for path in image_files(class_dir):
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                invalid.append(f"{path}: {type(error).__name__}")
                continue
            records.append(ImageRecord(label, path.resolve(), path.relative_to(class_dir)))
            valid_count += 1
        counts[label] = valid_count
        if valid_count < 2:
            raise ValueError(
                f"Class {label!r} contains only {valid_count} valid image(s); at least 2 are required. "
                f"Directory: {class_dir}"
            )

    return records, counts, invalid


def file_digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def drop_duplicates(records: list[ImageRecord]) -> tuple[list[ImageRecord], int]:
    """Remove byte-identical images (first occurrence wins, across all classes)."""
    seen: dict[str, ImageRecord] = {}
    kept: list[ImageRecord] = []
    for record in records:
        key = file_digest(record.path)
        if key in seen:
            continue
        seen[key] = record
        kept.append(record)
    return kept, len(records) - len(kept)


def split_records(
    records: list[ImageRecord], val_fraction: float, test_fraction: float,
    seed: int, max_per_class: int | None = None,
):
    """Stratified split. Order per class: shuffled -> val, then test, then train.

    Validation is taken first, so with the same seed and val fraction it is identical
    whether or not a test split is requested.
    """
    plans: dict[str, dict[str, list[ImageRecord]]] = {}
    for index, label in enumerate(CLASS_NAMES):
        shuffled = [record for record in records if record.label == label]
        random.Random(seed + index).shuffle(shuffled)
        if max_per_class is not None:
            shuffled = shuffled[:max_per_class]
        total = len(shuffled)
        min_needed = 3 if test_fraction > 0 else 2
        if total < min_needed:
            raise ValueError(f"Class {label!r} has {total} image(s) after filtering; need at least {min_needed}.")

        val_count = max(1, round(total * val_fraction))
        test_count = max(1, round(total * test_fraction)) if test_fraction > 0 else 0
        # Always keep at least one training image.
        while val_count + test_count > total - 1:
            if test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break
        plans[label] = {
            "val": shuffled[:val_count],
            "test": shuffled[val_count:val_count + test_count],
            "train": shuffled[val_count + test_count:],
        }
    return plans


def materialize(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite duplicate output file: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "hardlink":
        os.link(source, destination)
        return "hardlink"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_split(
    records: list[ImageRecord], output: Path, fraction: float, test_fraction: float, seed: int,
    mode: str, force: bool, source: Path, max_per_class: int | None = None,
) -> None:
    output = output.expanduser().resolve()
    source = source.expanduser().resolve()
    if is_within(output, source) or is_within(source, output):
        raise ValueError(f"Split output and dataset source cannot overlap. Source={source}; output={output}")

    plans = split_records(records, fraction, test_fraction, seed, max_per_class)
    if output.exists():
        if not force:
            raise FileExistsError(f"Split output already exists: {output}\nUse --force to regenerate only this split.")
        if output.is_symlink():
            raise ValueError(f"Refusing to recursively remove a symlink as split output: {output}")
        shutil.rmtree(output)

    linked = copied = 0
    split_names = ["train", "val"] + (["test"] if test_fraction > 0 else [])
    counts: dict[str, dict[str, int]] = {name: {} for name in split_names}
    for label in CLASS_NAMES:
        for split_name in split_names:
            items = plans[label][split_name]
            counts[split_name][label] = len(items)
            for item in items:
                action = materialize(item.path, output / split_name / label / item.relative_path, mode)
                linked += action == "hardlink"
                copied += action == "copy"

    manifest = {
        "source": str(source),
        "seed": seed,
        "validation_fraction": fraction,
        "test_fraction": test_fraction,
        "max_per_class": max_per_class,
        "class_names": list(CLASS_NAMES),
        "counts": counts,
        "file_materialization": {"hardlinked": linked, "copied": copied},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset source : {source}")
    print(f"Split output   : {output}")
    print(f"Validation     : {fraction:.0%} per class")
    if test_fraction > 0:
        print(f"Test           : {test_fraction:.0%} per class (do NOT tune on this)")
    print(f"Random seed    : {seed}")
    header = f"{'Class':<18} {'Train':>7} {'Val':>7}" + (f" {'Test':>7}" if test_fraction > 0 else "") + f" {'Total':>7}"
    print(header)
    print("-" * len(header))
    for label in CLASS_NAMES:
        row = [counts[name][label] for name in split_names]
        line = f"{label:<18} {row[0]:>7} {row[1]:>7}"
        if test_fraction > 0:
            line += f" {row[2]:>7}"
        print(line + f" {sum(row):>7}")
    print(f"\nCreated {linked} hardlinks and {copied} copies.")
    print("Dataset setup complete. The original dataset was not modified.")


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.val_size <= 0.50:
        raise ValueError("--val-size must be between 0.05 and 0.50")
    if args.test_size != 0 and not 0.05 <= args.test_size <= 0.40:
        raise ValueError("--test-size must be 0 (disabled) or between 0.05 and 0.40")
    if args.val_size + args.test_size > 0.60:
        raise ValueError("--val-size + --test-size must not exceed 0.60 (training set would be too small)")
    if args.max_per_class is not None and args.max_per_class < 3:
        raise ValueError("--max-per-class must be at least 3")
    if args.source is not None and args.archive is not None:
        raise ValueError("Use either --source or --archive, not both.")

    datasets_dir = args.datasets_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == datasets_dir:
        raise ValueError("--output cannot be the same directory as --datasets-dir.")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if args.archive is not None:
        extracted = extract_archive(args.archive, datasets_dir)
        roots = discover_class_roots(extracted, output)
        if len(roots) != 1:
            if not roots:
                raise FileNotFoundError(
                    f"Archive extracted, but no directory containing all five class folders was found under {extracted}. "
                    "Inspect the archive structure and use --source with the correct folder."
                )
            raise ValueError(
                "Archive contains multiple possible class roots:\n"
                + "\n".join(f"  - {path}" for path in roots)
                + "\nUse --source with the intended root."
            )
        source = roots[0]
    else:
        source = find_class_root(args.source, datasets_dir, output)

    records, class_counts, invalid = collect_records(source)
    print("Source image counts: " + ", ".join(f"{name}={class_counts[name]}" for name in CLASS_NAMES))
    if invalid:
        print(f"Skipped invalid/unreadable images: {len(invalid)}")
        for path in invalid[:5]:
            print(f"  - {path}")
    if args.dedupe:
        records, removed = drop_duplicates(records)
        print(f"Removed {removed} byte-identical duplicate image(s)")
    print(f"Total valid images: {len(records)}")

    max_per_class = args.max_per_class
    if args.balance:
        per_class = {name: sum(r.label == name for r in records) for name in CLASS_NAMES}
        smallest = min(per_class.values())
        max_per_class = smallest if max_per_class is None else min(smallest, max_per_class)
        discarded = sum(count - max_per_class for count in per_class.values() if count > max_per_class)
        print(f"Balancing: every class capped at {max_per_class} (smallest class); discarding {discarded} image(s)")

    prepare_split(
        records, output, args.val_size, args.test_size, args.seed,
        args.mode, args.force, source, max_per_class,
    )


if __name__ == "__main__":
    main()

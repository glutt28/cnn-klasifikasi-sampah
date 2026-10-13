#!/usr/bin/env python3
"""Prepare a deterministic train/validation split from the downloaded course dataset.

Expected source layout: one directory containing the five class directories.
The script does not read train_sources.csv or download individual images.
"""
from __future__ import annotations

import argparse
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


def split_records(records: list[ImageRecord], fraction: float, seed: int):
    plans: dict[str, dict[str, list[ImageRecord]]] = {}
    for index, label in enumerate(CLASS_NAMES):
        class_records = [record for record in records if record.label == label]
        shuffled = list(class_records)
        random.Random(seed + index).shuffle(shuffled)
        val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * fraction)))
        plans[label] = {"val": shuffled[:val_count], "train": shuffled[val_count:]}
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
    records: list[ImageRecord], output: Path, fraction: float, seed: int,
    mode: str, force: bool, source: Path,
) -> None:
    output = output.expanduser().resolve()
    source = source.expanduser().resolve()
    if is_within(output, source) or is_within(source, output):
        raise ValueError(f"Split output and dataset source cannot overlap. Source={source}; output={output}")

    plans = split_records(records, fraction, seed)
    if output.exists():
        if not force:
            raise FileExistsError(f"Split output already exists: {output}\nUse --force to regenerate only this split.")
        if output.is_symlink():
            raise ValueError(f"Refusing to recursively remove a symlink as split output: {output}")
        shutil.rmtree(output)

    linked = copied = 0
    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    for label in CLASS_NAMES:
        for split_name, items in plans[label].items():
            counts[split_name][label] = len(items)
            for item in items:
                action = materialize(item.path, output / split_name / label / item.relative_path, mode)
                linked += action == "hardlink"
                copied += action == "copy"

    manifest = {
        "source": str(source),
        "seed": seed,
        "validation_fraction": fraction,
        "class_names": list(CLASS_NAMES),
        "counts": counts,
        "file_materialization": {"hardlinked": linked, "copied": copied},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset source : {source}")
    print(f"Split output   : {output}")
    print(f"Validation     : {fraction:.0%} per class")
    print(f"Random seed    : {seed}")
    print(f"{'Class':<18} {'Train':>7} {'Val':>7} {'Total':>7}")
    print("-" * 42)
    for label in CLASS_NAMES:
        train_count = counts["train"][label]
        val_count = counts["val"][label]
        print(f"{label:<18} {train_count:>7} {val_count:>7} {train_count + val_count:>7}")
    print(f"\nCreated {linked} hardlinks and {copied} copies.")
    print("Dataset setup complete. The original dataset was not modified.")


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.val_size <= 0.50:
        raise ValueError("--val-size must be between 0.05 and 0.50")
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
    print(f"Total valid images: {len(records)}")

    prepare_split(records, output, args.val_size, args.seed, args.mode, args.force, source)


if __name__ == "__main__":
    main()

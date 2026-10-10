from pathlib import Path
from .constants import MAX_MODEL_SIZE_BYTES, REPO_ROOT


def check_files(repo_root: Path = REPO_ROOT) -> None:
    required = [
        repo_root / "README.md",
        repo_root / "evaluate.py",
        repo_root / "submission" / "model.py",
        repo_root / "submission" / "model.pth",
    ]

    for path in required:
        rel = path.relative_to(repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {rel}")
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty (0 bytes): {rel}")

    pth_file = repo_root / "submission" / "model.pth"
    size = pth_file.stat().st_size
    if size > MAX_MODEL_SIZE_BYTES:
        raise ValueError(f"submission/model.pth ({size / 1e6:.1f} MB) exceeds 500 MB limit")

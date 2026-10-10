from pathlib import Path
from .constants import CheckResult, MAX_MODEL_SIZE_BYTES, REPO_ROOT


def check_files(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    results = []
    required = [
        "README.md",
        "evaluate.py",
        "submission/model.py",
        "submission/model.pth",
    ]

    for rel in required:
        path = repo_root / rel
        if not path.is_file():
            results.append(CheckResult(f"File: {rel}", False, "File missing"))
        elif path.stat().st_size == 0:
            results.append(CheckResult(f"File: {rel}", False, "File is empty (0 bytes)"))
        else:
            results.append(CheckResult(f"File: {rel}", True, "Present and non-empty"))

    pth_file = repo_root / "submission" / "model.pth"
    if pth_file.is_file():
        size = pth_file.stat().st_size
        if size > MAX_MODEL_SIZE_BYTES:
            results.append(CheckResult("File size: submission/model.pth", False, f"{size / 1e6:.1f} MB exceeds 500 MB limit"))
        else:
            results.append(CheckResult("File size: submission/model.pth", True, f"{size / 1e6:.1f} MB <= 500 MB"))
    else:
        results.append(CheckResult("File size: submission/model.pth", False, "File missing"))

    return results

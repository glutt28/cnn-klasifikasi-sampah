import subprocess
import sys
import tempfile
from pathlib import Path
from .constants import CheckResult, CLASSES, REPO_ROOT


def check_evaluate(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    eval_script = repo_root / "evaluate.py"
    sub_dir = repo_root / "submission"
    model_py = sub_dir / "model.py"
    model_pth = sub_dir / "model.pth"

    try:
        from PIL import Image
    except ImportError as e:
        return [CheckResult("Integration: evaluate.py", False, f"Missing dependency: {e}")]

    if not model_py.is_file() or model_py.stat().st_size == 0 or not model_pth.is_file() or model_pth.stat().st_size == 0:
        return [CheckResult("Integration: evaluate.py", False, "Prerequisite failed (model.py or model.pth empty)")]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for cls in CLASSES:
            cls_folder = tmp_path / cls
            cls_folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (224, 224)).save(cls_folder / "sample.jpg")

        cmd = [
            sys.executable,
            str(eval_script),
            "--submission", str(sub_dir),
            "--data", str(tmp_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return [CheckResult("Integration: evaluate.py", True, "Executed without errors")]
        else:
            detail = (res.stderr or res.stdout).strip().splitlines()
            err_msg = detail[-1] if detail else f"Exit code {res.returncode}"
            return [CheckResult("Integration: evaluate.py", False, err_msg)]

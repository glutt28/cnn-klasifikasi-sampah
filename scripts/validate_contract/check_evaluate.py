import subprocess
import sys
import tempfile
from pathlib import Path
from .constants import CLASSES, REPO_ROOT


def check_evaluate(repo_root: Path = REPO_ROOT) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"Required dependency missing: {e}")

    eval_script = repo_root / "evaluate.py"
    sub_dir = repo_root / "submission"

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
        if res.returncode != 0:
            raise RuntimeError(f"evaluate.py failed with code {res.returncode}:\n{res.stderr or res.stdout}")

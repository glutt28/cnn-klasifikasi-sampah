from dataclasses import dataclass
from pathlib import Path

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
MAX_MODEL_SIZE_BYTES = 500 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

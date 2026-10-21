import importlib.util
import sys
from pathlib import Path

from .constants import CLASSES, REPO_ROOT, CheckResult

DOWNSTREAM = [
    "Function: get_model()",
    "Function: get_transform()",
    "Weights load (strict=True)",
    "Transform output format",
    "Model logits shape",
]


def _blocked(reason: str, names: list[str] = DOWNSTREAM) -> list[CheckResult]:
    return [CheckResult(n, False, f"Prerequisite failed ({reason})") for n in names]


def _load_module(model_py: Path):
    spec = importlib.util.spec_from_file_location("submission_model", model_py)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load module spec")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(model_py.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(model_py.parent))
    return mod


def _call(mod, name: str, label: str, validate):
    """Call mod.<name>(); return (value_or_None, CheckResult)."""
    if not hasattr(mod, name):
        return None, CheckResult(label, False, "Function not found")
    try:
        value = getattr(mod, name)()
    except Exception as e:
        return None, CheckResult(label, False, f"Call failed: {e}")
    ok, detail = validate(value)
    return (value if ok else None), CheckResult(label, ok, detail)


def check_model(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    sub_dir = (repo_root / "submission").resolve()
    model_py, model_pth = sub_dir / "model.py", sub_dir / "model.pth"

    try:
        import torch
        import torch.nn as nn
        from PIL import Image
    except ImportError as e:
        return [
            CheckResult("Dependencies (torch, PIL)", False, f"Missing: {e}"),
            CheckResult("Module import: model.py", False, "Prerequisite failed (dependencies missing)"),
            *_blocked("dependencies missing"),
        ]

    if not model_py.is_file() or model_py.stat().st_size == 0:
        return [
            CheckResult("Module import: model.py", False, "model.py missing or empty"),
            *_blocked("model.py missing or empty"),
        ]

    try:
        mod = _load_module(model_py)
    except Exception as e:
        return [CheckResult("Module import: model.py", False, f"Execution failed: {e}"), *_blocked("import failed")]
    results = [CheckResult("Module import: model.py", True, "Imported successfully")]

    model, r = _call(
        mod, "get_model", "Function: get_model()",
        lambda m: (True, "Returns torch.nn.Module") if isinstance(m, nn.Module)
        else (False, f"Returned {type(m).__name__}, expected torch.nn.Module"),
    )
    results.append(r)

    tf, r = _call(
        mod, "get_transform", "Function: get_transform()",
        lambda t: (True, "Returns callable") if callable(t)
        else (False, f"Returned {type(t).__name__}, expected callable"),
    )
    results.append(r)

    results.append(_check_weights(model, model_pth, torch))
    t1, t2, r = _check_transform(tf, Image, torch)
    results.append(r)
    results.append(_check_logits(model, t1, t2, torch))
    return results


def _check_weights(model, model_pth: Path, torch) -> CheckResult:
    name = "Weights load (strict=True)"
    if model is None:
        return CheckResult(name, False, "Prerequisite failed (model initialization)")
    if not model_pth.is_file() or model_pth.stat().st_size == 0:
        return CheckResult(name, False, "model.pth missing or empty")
    try:
        model.load_state_dict(torch.load(model_pth, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        return CheckResult(name, True, "Loaded state_dict cleanly")
    except Exception as e:
        return CheckResult(name, False, str(e))


def _check_transform(tf, Image, torch):
    name = "Transform output format"
    if tf is None:
        return None, None, CheckResult(name, False, "Prerequisite failed (get_transform)")
    try:
        t1, t2 = tf(Image.new("RGB", (100, 200))), tf(Image.new("RGB", (300, 150)))
    except Exception as e:
        return None, None, CheckResult(name, False, f"Execution failed: {e}")
    if not (isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor)):
        return None, None, CheckResult(name, False, "Output is not torch.Tensor")
    if t1.ndim != 3 or t1.shape[0] != 3:
        return None, None, CheckResult(name, False, f"Shape is {tuple(t1.shape)}, expected [3, H, W]")
    if t1.shape != t2.shape:
        return None, None, CheckResult(name, False, f"Inconsistent sizes: {tuple(t1.shape)} vs {tuple(t2.shape)}")
    return t1, t2, CheckResult(name, True, f"Tensor [3, H, W] uniform shape {tuple(t1.shape)}")


def _check_logits(model, t1, t2, torch) -> CheckResult:
    name = "Model logits shape"
    if model is None or t1 is None or t2 is None:
        return CheckResult(name, False, "Prerequisite failed (model or transform)")
    try:
        with torch.no_grad():
            logits = model(torch.stack([t1, t2]))
    except Exception as e:
        return CheckResult(name, False, f"Forward pass failed: {e}")
    expected = (2, len(CLASSES))
    if tuple(logits.shape) == expected:
        return CheckResult(name, True, f"Valid logits shape [batch, {len(CLASSES)}]")
    return CheckResult(name, False, f"Got {tuple(logits.shape)}, expected {expected}")

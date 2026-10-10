import importlib.util
import sys
from pathlib import Path
from .constants import CheckResult, CLASSES, REPO_ROOT


def check_model(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    results = []
    sub_dir = (repo_root / "submission").resolve()
    model_py = sub_dir / "model.py"
    model_pth = sub_dir / "model.pth"

    try:
        from PIL import Image
        import torch
        import torch.nn as nn
        deps_ok = True
    except ImportError as e:
        deps_ok = False
        results.append(CheckResult("Dependencies (torch, PIL)", False, f"Missing: {e}"))

    if not model_py.is_file() or model_py.stat().st_size == 0:
        results.append(CheckResult("Module import: model.py", False, "model.py missing or empty"))
        results.append(CheckResult("Function: get_model()", False, "Prerequisite failed (model.py missing or empty)"))
        results.append(CheckResult("Function: get_transform()", False, "Prerequisite failed (model.py missing or empty)"))
        results.append(CheckResult("Weights load (strict=True)", False, "Prerequisite failed (model.py missing or empty)"))
        results.append(CheckResult("Transform output format", False, "Prerequisite failed (model.py missing or empty)"))
        results.append(CheckResult("Model logits shape", False, "Prerequisite failed (model.py missing or empty)"))
        return results

    if not deps_ok:
        results.append(CheckResult("Module import: model.py", False, "Prerequisite failed (dependencies missing)"))
        results.append(CheckResult("Function: get_model()", False, "Prerequisite failed (dependencies missing)"))
        results.append(CheckResult("Function: get_transform()", False, "Prerequisite failed (dependencies missing)"))
        results.append(CheckResult("Weights load (strict=True)", False, "Prerequisite failed (dependencies missing)"))
        results.append(CheckResult("Transform output format", False, "Prerequisite failed (dependencies missing)"))
        results.append(CheckResult("Model logits shape", False, "Prerequisite failed (dependencies missing)"))
        return results

    spec = importlib.util.spec_from_file_location("submission_model", model_py)
    if spec is None or spec.loader is None:
        results.append(CheckResult("Module import: model.py", False, "Cannot load module spec"))
        return results

    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sub_dir))
    try:
        spec.loader.exec_module(mod)
        results.append(CheckResult("Module import: model.py", True, "Imported successfully"))
    except Exception as e:
        results.append(CheckResult("Module import: model.py", False, f"Execution failed: {e}"))
        return results

    # Check get_model()
    model = None
    if not hasattr(mod, "get_model"):
        results.append(CheckResult("Function: get_model()", False, "Function not found"))
    else:
        try:
            model = mod.get_model()
            if isinstance(model, nn.Module):
                results.append(CheckResult("Function: get_model()", True, "Returns torch.nn.Module"))
            else:
                results.append(CheckResult("Function: get_model()", False, f"Returned {type(model).__name__}, expected torch.nn.Module"))
                model = None
        except Exception as e:
            results.append(CheckResult("Function: get_model()", False, f"Call failed: {e}"))

    # Check get_transform()
    tf = None
    if not hasattr(mod, "get_transform"):
        results.append(CheckResult("Function: get_transform()", False, "Function not found"))
    else:
        try:
            tf = mod.get_transform()
            if callable(tf):
                results.append(CheckResult("Function: get_transform()", True, "Returns callable"))
            else:
                results.append(CheckResult("Function: get_transform()", False, f"Returned {type(tf).__name__}, expected callable"))
                tf = None
        except Exception as e:
            results.append(CheckResult("Function: get_transform()", False, f"Call failed: {e}"))

    # Check weights loading
    if model is None:
        results.append(CheckResult("Weights load (strict=True)", False, "Prerequisite failed (model initialization)"))
    elif not model_pth.is_file() or model_pth.stat().st_size == 0:
        results.append(CheckResult("Weights load (strict=True)", False, "model.pth missing or empty"))
    else:
        try:
            state = torch.load(model_pth, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            results.append(CheckResult("Weights load (strict=True)", True, "Loaded state_dict cleanly"))
        except Exception as e:
            results.append(CheckResult("Weights load (strict=True)", False, str(e)))

    # Check transform output
    t1, t2 = None, None
    if tf is None:
        results.append(CheckResult("Transform output format", False, "Prerequisite failed (get_transform)"))
    else:
        try:
            t1 = tf(Image.new("RGB", (100, 200)))
            t2 = tf(Image.new("RGB", (300, 150)))
            if not (isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor)):
                results.append(CheckResult("Transform output format", False, "Output is not torch.Tensor"))
                t1, t2 = None, None
            elif t1.ndim != 3 or t1.shape[0] != 3:
                results.append(CheckResult("Transform output format", False, f"Shape is {tuple(t1.shape)}, expected [3, H, W]"))
                t1, t2 = None, None
            elif t1.shape != t2.shape:
                results.append(CheckResult("Transform output format", False, f"Inconsistent sizes: {tuple(t1.shape)} vs {tuple(t2.shape)}"))
                t1, t2 = None, None
            else:
                results.append(CheckResult("Transform output format", True, f"Tensor [3, H, W] uniform shape {tuple(t1.shape)}"))
        except Exception as e:
            results.append(CheckResult("Transform output format", False, f"Execution failed: {e}"))
            t1, t2 = None, None

    # Check forward pass logits
    if model is None or t1 is None or t2 is None:
        results.append(CheckResult("Model logits shape", False, "Prerequisite failed (model or transform)"))
    else:
        try:
            with torch.no_grad():
                logits = model(torch.stack([t1, t2]))
            expected = (2, len(CLASSES))
            if logits.ndim == 2 and logits.shape == expected:
                results.append(CheckResult("Model logits shape", True, f"Valid logits shape [batch, {len(CLASSES)}]"))
            else:
                results.append(CheckResult("Model logits shape", False, f"Got {tuple(logits.shape)}, expected {expected}"))
        except Exception as e:
            results.append(CheckResult("Model logits shape", False, f"Forward pass failed: {e}"))

    return results

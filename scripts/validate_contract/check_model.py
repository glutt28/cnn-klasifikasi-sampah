import importlib.util
import sys
from pathlib import Path
from .constants import CLASSES, REPO_ROOT


def check_model(repo_root: Path = REPO_ROOT) -> None:
    try:
        from PIL import Image
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(f"Required dependency missing: {e}")

    sub_dir = (repo_root / "submission").resolve()
    model_py = sub_dir / "model.py"
    spec = importlib.util.spec_from_file_location("submission_model", model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {model_py}")

    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sub_dir))
    spec.loader.exec_module(mod)

    for fn in ("get_model", "get_transform"):
        if not hasattr(mod, fn):
            raise AttributeError(f"submission/model.py missing function: {fn}()")

    model = mod.get_model()
    if not isinstance(model, nn.Module):
        raise TypeError(f"get_model() must return torch.nn.Module, got {type(model).__name__}")

    pth_path = sub_dir / "model.pth"
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    tf = mod.get_transform()
    if not callable(tf):
        raise TypeError(f"get_transform() must return a callable, got {type(tf).__name__}")

    t1 = tf(Image.new("RGB", (100, 200)))
    t2 = tf(Image.new("RGB", (300, 150)))

    if not (isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor)):
        raise TypeError("Transform must return torch.Tensor")
    if t1.ndim != 3 or t1.shape[0] != 3:
        raise ValueError(f"Transform tensor must have shape [3, H, W], got {tuple(t1.shape)}")
    if t1.shape != t2.shape:
        raise ValueError(f"Transform must produce uniform resolution, got {tuple(t1.shape)} vs {tuple(t2.shape)}")

    with torch.no_grad():
        logits = model(torch.stack([t1, t2]))

    expected_shape = (2, len(CLASSES))
    if logits.ndim != 2 or logits.shape != expected_shape:
        raise ValueError(f"Model output shape must be {expected_shape}, got {tuple(logits.shape)}")

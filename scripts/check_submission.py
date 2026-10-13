#!/usr/bin/env python3
"""Check that model.pth matches the official Studio 2 submission contract."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from submission.model import CLASS_NAMES, get_model, get_transform  # noqa: E402


def main() -> None:
    checkpoint_path = PROJECT_ROOT / "submission" / "model.pth"
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\nRun `python train.py` after dataset setup."
        )

    model = get_model()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))

    assert tuple(logits.shape) == (1, len(CLASS_NAMES)), tuple(logits.shape)
    assert callable(get_transform())
    print(f"OK: checkpoint loads strictly; output shape={tuple(logits.shape)}; classes={list(CLASS_NAMES)}")


if __name__ == "__main__":
    main()

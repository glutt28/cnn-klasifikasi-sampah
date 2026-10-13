#!/usr/bin/env python3
"""Smoke-test PyTorch GPU compute, including a CNN backward pass."""
from __future__ import annotations

import torch
from torch import nn


def main() -> None:
    print("PyTorch:", torch.__version__)
    print("HIP:", torch.version.hip)
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: torch.cuda.is_available() is False.")

    print("GPU:", torch.cuda.get_device_name(0))
    properties = torch.cuda.get_device_properties(0)
    print("Architecture:", getattr(properties, "gcnArchName", "unknown"))
    print(f"VRAM: {properties.total_memory / (1024 ** 3):.2f} GiB")

    left = torch.randn(2048, 2048, device="cuda")
    right = torch.randn(2048, 2048, device="cuda")
    product = left @ right
    torch.cuda.synchronize()
    print("Matrix multiplication: PASS", tuple(product.shape))

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 5),
    ).cuda()
    images = torch.randn(4, 3, 224, 224, device="cuda")
    loss = model(images).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    print("CNN forward/backward: PASS")
    print("GPU smoke test: PASS")


if __name__ == "__main__":
    main()

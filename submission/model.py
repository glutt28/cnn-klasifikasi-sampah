"""UniRepLKNet-S submission adapter for Studio 2."""
from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:  # Imported as submission.model during training.
    from .unireplknet import unireplknet_s
except ImportError:  # Loaded from a file path by the official evaluator.
    from unireplknet import unireplknet_s


CLASS_NAMES = (
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam",
)

IMAGE_SIZE = 288
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_model():
    """Build the five-class architecture without loading any weights."""
    return unireplknet_s(num_classes=len(CLASS_NAMES), attempt_use_lk_impl=False)


def get_transform(image_size: int = IMAGE_SIZE):
    """Return the deterministic PIL RGB -> normalized tensor transform."""
    return transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

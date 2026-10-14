"""Submission Studio 2 - Klasifikasi Sampah. Backbone: EfficientNet (timm)."""
from torchvision import transforms as T

import timm

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]

BACKBONE = "efficientnet_b0"
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)  
STD = (0.229, 0.224, 0.225)


def build_model(pretrained=False, **kwargs):
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=len(CLASSES), **kwargs)


def get_model():
    return build_model(pretrained=False)


def get_transform():
    # Resize langsung ke persegi (tanpa crop) supaya seluruh objek tetap terlihat.
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

"""Albumentations augmentation pipeline for cephalometric images.

IMPORTANT: horizontal_flip is permanently disabled.
Lateral cephalograms have strict anatomical orientation — flipping invalidates ANS/PNS/landmarks.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(
    rotation_limit: int = 5,
    zoom_limit: float = 0.1,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    clahe: bool = True,
    horizontal_flip: bool = False,  # must remain False
) -> A.Compose:
    if horizontal_flip:
        raise ValueError(
            "horizontal_flip=True is forbidden for lateral cephalograms. "
            "Flipping swaps ANS/PNS and makes landmark coordinates anatomically invalid."
        )

    transforms = [
        A.Affine(
            translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
            scale=(1 - zoom_limit, 1 + zoom_limit),
            rotate=(-rotation_limit, rotation_limit),
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.7,
        ),
        A.ElasticTransform(alpha=30, sigma=4, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
        A.GridDistortion(distort_limit=0.2, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
    ]

    if clahe:
        transforms.append(A.CLAHE(clip_limit=4.0, p=0.5))

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_val_transform() -> A.Compose:
    return A.Compose(
        [],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

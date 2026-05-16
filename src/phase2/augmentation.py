"""Albumentations augmentation pipeline for cephalometric images.

IMPORTANT: horizontal_flip is permanently disabled.
Lateral cephalograms have strict anatomical orientation — flipping invalidates ANS/PNS/landmarks.

With only ~92 annotated images, augmentation is kept minimal to avoid destroying
landmark geometry. Only affine transforms (translate/scale/rotate) + CLAHE are used.
ElasticTransform, GridDistortion, Perspective are too aggressive for a tiny dataset.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(
    rotation_limit: int = 5,
    zoom_limit: float = 0.1,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    clahe: bool = True,
    horizontal_flip: bool = False,  # must remain False
) -> A.Compose:
    if horizontal_flip:
        raise ValueError(
            "horizontal_flip=True is forbidden for lateral cephalograms. "
            "Flipping swaps ANS/PNS and makes landmark coordinates anatomically invalid."
        )

    # Only affine (translate/scale/rotate) + mild color augmentation
    # Elastic/GridDistortion/Perspective removed — too aggressive for 92-image dataset
    transforms = [
        A.Affine(
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            scale=(1 - zoom_limit, 1 + zoom_limit),
            rotate=(-rotation_limit, rotation_limit),
            p=0.8,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
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

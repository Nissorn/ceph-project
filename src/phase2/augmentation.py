"""Albumentations augmentation pipeline for cephalometric images.

IMPORTANT: horizontal_flip is permanently disabled.
Lateral cephalograms have strict anatomical orientation — flipping invalidates ANS/PNS/landmarks.

With only ~92 annotated images, augmentation is critical for regularization.
We use affine (translate/scale/rotate) + elastic deformation + CLAHE.
Elastic transforms the spatial structure of landmarks, forcing the model to learn
invariant features rather than memorizing specific positions.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(
    rotation_limit: int = 15,
    zoom_limit: float = 0.2,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    clahe: bool = True,
    horizontal_flip: bool = False,  # must remain False
    elastic_deform: bool = True,
    grid_distort: bool = True,
) -> A.Compose:
    if horizontal_flip:
        raise ValueError(
            "horizontal_flip=True is forbidden for lateral cephalograms. "
            "Flipping swaps ANS/PNS and makes landmark coordinates anatomically invalid."
        )

    transforms = [
        # Geometric augmentation: rotation ±15°, scale ±20%, translate ±6%
        A.Affine(
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            scale=(1 - zoom_limit, 1 + zoom_limit),
            rotate=(-rotation_limit, rotation_limit),
            p=0.9,
        ),
        # Elastic deformation — warps landmark positions, forces spatial invariance
        A.ElasticTransform(
            alpha=1.0,
            sigma=50,
            p=0.1,
        ),
        # Grid distortion — moderate spatial warp
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.05,
            p=0.1,
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

"""ANS-PNS rigid registration for T1/T2 superimposition."""

import numpy as np


# Keypoint indices (matches KEYPOINT_NAMES constant)
ANS_IDX = 6
PNS_IDX = 7


def compute_rigid_transform(
    ans: np.ndarray, pns: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute rigid transform that maps coordinate system so:
        - ANS → origin
        - ANS-PNS vector → horizontal (positive x)

    Returns:
        translation: [2] — subtract from points before rotation
        angle_rad: float — rotation angle (counterclockwise)
        rotation_matrix: [2, 2]
    """
    translation = ans.copy()
    vec = pns - ans
    angle_rad = -np.arctan2(vec[1], vec[0])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[c, -s], [s, c]], dtype=np.float64)
    return translation, angle_rad, rotation_matrix


def apply_transform(
    points: np.ndarray,
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply translation then rotation to Nx2 point array."""
    shifted = points - translation
    return (rotation_matrix @ shifted.T).T


def superimpose_on_ans_pns(
    keypoints_t1: np.ndarray,
    keypoints_t2: np.ndarray,
    valid_mask_t1: np.ndarray,
    valid_mask_t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Register T1 and T2 coordinate frames using the ANS-PNS reference plane.

    Both T1 and T2 are transformed so that ANS (T1) is the origin and
    ANS-PNS (T1) is the horizontal axis. The same T1 transform is applied to T2
    so that displacements can be directly compared.

    Returns:
        kp_t1_reg: [N, 2] T1 keypoints in registered space
        kp_t2_reg: [N, 2] T2 keypoints in registered space
        success: bool — False if ANS/PNS not available in T1
    """
    if not (valid_mask_t1[ANS_IDX] and valid_mask_t1[PNS_IDX]):
        return keypoints_t1, keypoints_t2, False

    ans_t1 = keypoints_t1[ANS_IDX]
    pns_t1 = keypoints_t1[PNS_IDX]
    translation, _, rotation_matrix = compute_rigid_transform(ans_t1, pns_t1)

    kp_t1_reg = apply_transform(keypoints_t1, translation, rotation_matrix)
    kp_t2_reg = apply_transform(keypoints_t2, translation, rotation_matrix)

    return kp_t1_reg, kp_t2_reg, True

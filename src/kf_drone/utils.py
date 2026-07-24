"""Quaternion algebra and rotation utilities for the ESKF.

All quaternions use Hamilton convention [w, x, y, z] with w as scalar part.
Active rotation: v' = q ⊗ v ⊗ q⁻¹ (pure quaternion v).
Frame: local NED (North-East-Down).
"""

import numpy as np
from numpy.typing import NDArray


def quat_multiply(q: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product q ⊗ p."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inverse(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Conjugate of a unit quaternion (= inverse)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_exp(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponential map: rotation vector → unit quaternion.

    phi is a 3-vector where |phi| = rotation angle in radians,
    phi / |phi| = rotation axis.

    Uses Taylor expansion for very small angles to avoid 0/0.
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        angle_sq = angle * angle
        w = 1.0 - angle_sq / 8.0
        factor = 0.5 * (1.0 - angle_sq / 24.0)
        xyz = phi * factor
    else:
        half = 0.5 * angle
        w = np.cos(half)
        factor = np.sin(half) / angle
        xyz = phi * factor
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quat_log(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logarithm map: unit quaternion → rotation vector.

    Returns the *shortest* rotation (angle ≤ π).  Because q and −q represent
    the same rotation, the scalar part is forced to be non‑negative first.
    """
    w, x, y, z = q
    xyz = np.array([x, y, z])
    v_norm = np.linalg.norm(xyz)
    if v_norm < 1e-12:
        return np.zeros(3)
    # Force shortest rotation: −q = same orientation, shorter path when w < 0
    if w < 0.0:
        w = -w
        xyz = -xyz
        v_norm = np.linalg.norm(xyz)
    w_clamped = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arctan2(v_norm, w_clamped)
    return angle * xyz / v_norm


def quat_to_rotmat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit quaternion → 3×3 rotation matrix (body → world)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def rotmat_to_quat(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix → unit quaternion (Shepperd's method)."""
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def skew_symmetric(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Skew-symmetric matrix [v]× such that [v]× · u = v × u."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def quat_normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize to unit quaternion. Returns identity if near-zero."""
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_rotate(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate 3-vector v by unit quaternion q: q ⊗ v ⊗ q⁻¹.

    Uses the efficient sandwich-product formula, avoiding full quaternion multiplies.
    """
    w, x, y, z = q
    vx, vy, vz = v
    x2, y2, z2 = x + x, y + y, z + z
    wx2, wy2, wz2 = w * x2, w * y2, w * z2
    xx2, xy2, xz2 = x * x2, x * y2, x * z2
    yy2, yz2, zz2 = y * y2, y * z2, z * z2

    return np.array(
        [
            vx * (1.0 - yy2 - zz2) + vy * (xy2 - wz2) + vz * (xz2 + wy2),
            vx * (xy2 + wz2) + vy * (1.0 - xx2 - zz2) + vz * (yz2 - wx2),
            vx * (xz2 - wy2) + vy * (yz2 + wx2) + vz * (1.0 - xx2 - yy2),
        ]
    )

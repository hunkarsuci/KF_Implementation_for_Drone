"""Tests for kf_drone.utils — quaternion algebra and rotation utilities."""

import numpy as np

from kf_drone.utils import (
    quat_exp,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_rotmat,
    rotmat_to_quat,
    skew_symmetric,
)


class TestQuaternionAlgebra:
    def test_multiply_identity(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        q = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        np.testing.assert_allclose(quat_multiply(q_id, q), q, atol=1e-12)
        np.testing.assert_allclose(quat_multiply(q, q_id), q, atol=1e-12)

    def test_inverse_is_conjugate(self):
        q = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        result = quat_multiply(q, quat_inverse(q))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_sequential_rotations_match_combined(self):
        """Two consecutive quaternion rotations = one combined rotation."""
        rng = np.random.default_rng(55)
        q1 = quat_exp(rng.normal(size=3))
        q2 = quat_exp(rng.normal(size=3))
        q_combined = quat_multiply(q2, q1)  # q2 after q1
        v = rng.normal(size=3)
        sequential = quat_rotate(q2, quat_rotate(q1, v))
        combined = quat_rotate(q_combined, v)
        np.testing.assert_allclose(sequential, combined, atol=1e-12)

    def test_rotate_preserves_vector_norm(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = quat_exp(rng.normal(size=3))
            v = rng.normal(size=3)
            v_rot = quat_rotate(q, v)
            np.testing.assert_allclose(np.linalg.norm(v_rot), np.linalg.norm(v), atol=1e-12)


class TestExpLogRoundtrip:
    def test_roundtrip(self):
        """exp → log roundtrip for rotation vectors with angle < π."""
        rng = np.random.default_rng(7)
        for _ in range(30):
            # Generate random axis, random angle < π (so w ≥ 0, exact roundtrip)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.01, np.pi - 0.01)
            phi = axis * angle
            q = quat_exp(phi)
            phi_back = quat_log(q)
            np.testing.assert_allclose(phi_back, phi, atol=1e-10)

    def test_roundtrip_large_angle(self):
        """Rotation vectors with |φ| > π: log returns the shorter equivalent."""
        # 200° rotation about Z → same quaternion as −160° about Z
        phi = np.array([0.0, 0.0, np.deg2rad(200.0)])
        q = quat_exp(phi)
        phi_back = quat_log(q)
        # quat_log should return the shorter rotation: −160° about Z
        np.testing.assert_allclose(phi_back, np.array([0.0, 0.0, np.deg2rad(-160.0)]), atol=1e-10)

    def test_sign_ambiguity(self):
        """log(−q) should equal log(q): both represent the same rotation."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.1, np.pi - 0.1)
            q = quat_exp(axis * angle)
            np.testing.assert_allclose(quat_log(q), quat_log(-q), atol=1e-10)

    def test_zero_angle(self):
        q = quat_exp(np.zeros(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_small_angle_taylor_branch(self):
        phi = np.array([1e-8, -2e-8, 3e-8])
        q_taylor = quat_exp(phi)
        angle = np.linalg.norm(phi)
        q_exact = np.array([np.cos(angle / 2), *(phi * np.sin(angle / 2) / angle)])
        np.testing.assert_allclose(q_taylor, q_exact, atol=1e-14)


class TestRotationMatrix:
    def test_orthogonal_determinant_one(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = quat_exp(rng.normal(size=3))
            R = quat_to_rotmat(q)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_quat_to_rotmat_roundtrip(self):
        rng = np.random.default_rng(13)
        for _ in range(20):
            q_orig = quat_exp(rng.normal(size=3))
            R = quat_to_rotmat(q_orig)
            q_back = rotmat_to_quat(R)
            # Handle sign ambiguity: both q and -q represent the same rotation
            R_back = quat_to_rotmat(q_back)
            np.testing.assert_allclose(R_back, R, atol=1e-12)

    def test_rotation_axis_unchanged(self):
        """Rotating the rotation axis vector shouldn't change it."""
        q = quat_exp(np.array([0.0, 0.0, 0.5]))  # rotation about Z
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12)

    def test_90_degree_z_rotation(self):
        q = quat_exp(np.array([0, 0, np.pi / 2]))
        v_rot = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(quat_to_rotmat(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


class TestSkewSymmetric:
    def test_cross_product_equivalent(self):
        rng = np.random.default_rng(99)
        for _ in range(10):
            a = rng.normal(size=3)
            b = rng.normal(size=3)
            np.testing.assert_allclose(skew_symmetric(a) @ b, np.cross(a, b), atol=1e-12)

    def test_zero_vector(self):
        np.testing.assert_allclose(skew_symmetric(np.zeros(3)), np.zeros((3, 3)), atol=1e-15)


class TestQuatNormalize:
    def test_unit_preserving(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(quat_normalize(q), [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_near_zero_returns_identity(self):
        np.testing.assert_allclose(quat_normalize(np.zeros(4)), [1.0, 0.0, 0.0, 0.0], atol=1e-12)

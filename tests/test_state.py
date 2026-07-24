"""Tests for kf_drone.state — nominal and error state vectors."""

import numpy as np

from kf_drone.state import ErrorState, NominalState, error_reset_jacobian, inject_error
from kf_drone.utils import quat_to_rotmat


class TestNominalState:
    def test_defaults(self):
        ns = NominalState()
        np.testing.assert_allclose(ns.p, [0, 0, 0])
        np.testing.assert_allclose(ns.v, [0, 0, 0])
        np.testing.assert_allclose(ns.q, [1, 0, 0, 0])

    def test_custom_init(self):
        ns = NominalState(
            p=np.array([1.0, 2.0, 3.0]),
            v=np.array([4.0, 5.0, 6.0]),
            q=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(ns.p, [1, 2, 3])
        np.testing.assert_allclose(ns.v, [4, 5, 6])
        np.testing.assert_allclose(ns.q, [0, 0, 0, 1])

    def test_copy_is_deep(self):
        ns = NominalState(p=np.array([1.0, 0.0, 0.0]))
        ns2 = ns.copy()
        ns2.p[0] = 99.0
        assert ns.p[0] == 1.0  # original unchanged

    def test_as_vector(self):
        ns = NominalState(
            p=np.array([1, 2, 3]),
            v=np.array([4, 5, 6]),
            q=np.array([0.7, 0.1, 0.2, 0.3]),
        )
        v = ns.as_vector()
        assert v.shape == (10,)
        np.testing.assert_allclose(v[:3], ns.p)
        np.testing.assert_allclose(v[3:6], ns.v)
        np.testing.assert_allclose(v[6:10], ns.q)


class TestErrorState:
    def test_defaults_zero(self):
        es = ErrorState()
        v = es.as_vector()
        np.testing.assert_allclose(v, np.zeros(15), atol=1e-15)

    def test_from_vector_roundtrip(self):
        rng = np.random.default_rng(1)
        vec = rng.normal(size=15)
        es = ErrorState()
        es.from_vector(vec)
        np.testing.assert_allclose(es.as_vector(), vec, atol=1e-12)

    def test_reset(self):
        es = ErrorState()
        es.delta_p = np.ones(3)
        es.reset()
        np.testing.assert_allclose(es.as_vector(), np.zeros(15), atol=1e-15)

    def test_dimension_constant(self):
        assert ErrorState.DIM == 15


class TestInjectError:
    def test_inject_position(self):
        ns = NominalState()
        es = ErrorState()
        es.delta_p = np.array([1.0, 2.0, 3.0])
        inject_error(ns, es)
        np.testing.assert_allclose(ns.p, [1, 2, 3])
        np.testing.assert_allclose(es.as_vector(), np.zeros(15), atol=1e-15)

    def test_inject_velocity(self):
        ns = NominalState()
        es = ErrorState()
        es.delta_v = np.array([10.0, 0.0, -5.0])
        inject_error(ns, es)
        np.testing.assert_allclose(ns.v, [10, 0, -5])

    def test_inject_attitude_small(self):
        ns = NominalState()
        es = ErrorState()
        es.delta_theta = np.array([0.0, 0.0, 0.5])
        inject_error(ns, es)
        # Should be a rotation about Z by ~0.5 rad
        R = quat_to_rotmat(ns.q)
        np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-12)  # Z-axis unchanged
        np.testing.assert_allclose(np.linalg.norm(ns.q), 1.0, atol=1e-12)

    def test_inject_then_reset(self):
        ns = NominalState()
        es = ErrorState()
        es.delta_p = np.array([1.0, 0.0, 0.0])
        es.delta_theta = np.array([0.1, 0.2, 0.3])
        inject_error(ns, es)
        # Error must be zeroed after injection
        np.testing.assert_allclose(es.as_vector(), np.zeros(15), atol=1e-15)

    def test_injection_quaternion_is_normalized(self):
        """After repeated injections, quaternion must stay unit-norm."""
        rng = np.random.default_rng(42)
        ns = NominalState()
        for _ in range(100):
            es = ErrorState()
            es.delta_theta = rng.normal(scale=0.1, size=3)
            inject_error(ns, es)
            assert abs(np.linalg.norm(ns.q) - 1.0) < 1e-12


class TestErrorResetJacobian:
    def test_identity_for_zero_error(self):
        G = error_reset_jacobian(np.zeros(3))
        np.testing.assert_allclose(G, np.eye(15), atol=1e-15)

    def test_attitude_block_correct(self):
        delta_theta = np.array([0.1, -0.2, 0.05])
        G = error_reset_jacobian(delta_theta)
        # Attitude block should be I - 0.5 * skew(delta_theta)
        expected_block = np.eye(3) - 0.5 * np.array(
            [
                [0, -delta_theta[2], delta_theta[1]],
                [delta_theta[2], 0, -delta_theta[0]],
                [-delta_theta[1], delta_theta[0], 0],
            ]
        )
        np.testing.assert_allclose(G[6:9, 6:9], expected_block, atol=1e-14)

    def test_non_attitude_blocks_are_identity(self):
        delta_theta = np.array([1.0, 2.0, 3.0])
        G = error_reset_jacobian(delta_theta)
        # Position block (0:3, 0:3)
        np.testing.assert_allclose(G[0:3, 0:3], np.eye(3), atol=1e-15)
        # Velocity block (3:6, 3:6)
        np.testing.assert_allclose(G[3:6, 3:6], np.eye(3), atol=1e-15)
        # Bias blocks
        np.testing.assert_allclose(G[9:12, 9:12], np.eye(3), atol=1e-15)
        np.testing.assert_allclose(G[12:15, 12:15], np.eye(3), atol=1e-15)

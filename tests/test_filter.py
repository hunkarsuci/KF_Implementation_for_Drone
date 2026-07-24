"""Tests for kf_drone.filter — the Error-State Kalman Filter."""

import numpy as np
import pytest

from kf_drone.filter import ESKF
from kf_drone.state import ErrorState


class TestESKFInit:
    def test_default_initialisation(self):
        kf = ESKF()
        np.testing.assert_allclose(kf.position, [0, 0, 0])
        np.testing.assert_allclose(kf.velocity, [0, 0, 0])
        np.testing.assert_allclose(kf.attitude, [1, 0, 0, 0])
        np.testing.assert_allclose(kf.gyro_bias, [0, 0, 0])
        np.testing.assert_allclose(kf.accel_bias, [0, 0, 0])
        assert kf.covariance.shape == (15, 15)
        # Diagonal P
        P = kf.covariance
        np.testing.assert_allclose(P, np.diag(np.diag(P)), atol=1e-12)

    def test_custom_init(self):
        kf = ESKF(
            init_p=np.array([1.0, 2.0, 3.0]),
            init_v=np.array([4.0, 5.0, 6.0]),
            init_b_g=np.array([0.01, -0.02, 0.03]),
            init_b_a=np.array([0.1, -0.1, 0.05]),
        )
        np.testing.assert_allclose(kf.position, [1, 2, 3])
        np.testing.assert_allclose(kf.velocity, [4, 5, 6])
        np.testing.assert_allclose(kf.gyro_bias, [0.01, -0.02, 0.03])
        np.testing.assert_allclose(kf.accel_bias, [0.1, -0.1, 0.05])

    def test_init_P_diag(self):
        diag = np.full(15, 0.5)

        kf = ESKF(init_P_diag=diag)
        P = kf.covariance
        np.testing.assert_allclose(np.diag(P), 0.25 * np.ones(15), atol=1e-12)

    def test_state_vector_dimension(self):
        kf = ESKF()
        sv = kf.get_state_vector()
        assert sv.shape == (16,)  # 3+3+4+3+3
        np.testing.assert_allclose(sv[0:3], kf.position)
        np.testing.assert_allclose(sv[3:6], kf.velocity)
        np.testing.assert_allclose(sv[6:10], kf.attitude)
        np.testing.assert_allclose(sv[10:13], kf.gyro_bias)
        np.testing.assert_allclose(sv[13:16], kf.accel_bias)


class TestPredict:
    def test_stationary_no_bias(self):
        """Stationary drone: IMU measures gravity, position should stay."""
        kf = ESKF()
        # Stationary: w_m = 0, a_m = Rᵀ·(-g) + b_a
        # For level attitude: a_m = -g_world = [0, 0, -9.81], w_m = [0, 0, 0]
        w_m = np.zeros(3)
        a_m = np.array([0.0, 0.0, -9.81])  # counteracting gravity
        dt = 0.01

        kf.predict(w_m, a_m, dt)
        np.testing.assert_allclose(kf.position, [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(kf.velocity, [0, 0, 0], atol=1e-12)

    def test_predict_increases_covariance(self):
        """Process noise should increase the covariance."""
        kf = ESKF(sigma_g=0.01, sigma_a=0.05, sigma_bg=0.001, sigma_ba=0.01)
        P_before = kf.covariance.copy()
        w_m = np.array([0.01, 0.0, 0.0])
        a_m = np.array([0.0, 0.0, -9.81])
        kf.predict(w_m, a_m, dt=0.01)
        P_after = kf.covariance.copy()

        # Trace should increase due to process noise
        assert np.trace(P_after) > np.trace(P_before)

    def test_predict_moves_position_with_velocity(self):
        """With non-zero velocity, position should change."""
        kf = ESKF(init_v=np.array([1.0, 0.0, 0.0]))
        w_m = np.zeros(3)
        a_m = np.array([0.0, 0.0, -9.81])  # level, stationary (specific force = -g)
        dt = 0.1
        kf.predict(w_m, a_m, dt)
        # Position should move in x: v_x * dt + 0.5 * a_x * dt²
        # a_x = 0 (no world accel: R=I and a_m = [0,0,-9.81] + g = 0)
        np.testing.assert_allclose(kf.position[0], 0.1, atol=1e-12)
        np.testing.assert_allclose(kf.position[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(kf.position[2], 0.0, atol=1e-12)

    def test_predict_negative_dt_raises(self):
        kf = ESKF()
        with pytest.raises(ValueError, match="dt must be positive"):
            kf.predict(np.zeros(3), np.zeros(3), dt=-0.01)

    def test_predict_zero_dt_raises(self):
        kf = ESKF()
        with pytest.raises(ValueError, match="dt must be positive"):
            kf.predict(np.zeros(3), np.zeros(3), dt=0.0)

    def test_predict_rotates_with_angular_velocity(self):
        """Pure rotation about z should rotate the quaternion."""
        kf = ESKF()
        w_m = np.array([0.0, 0.0, 1.0])  # 1 rad/s about z
        a_m = np.array([0.0, 0.0, -9.81])  # counteract gravity
        dt = 0.01
        kf.predict(w_m, a_m, dt)
        # After dt, q should be rotated about z by w*dt = 0.01 rad
        from kf_drone.utils import quat_exp

        expected_q = quat_exp(np.array([0.0, 0.0, 0.01]))
        np.testing.assert_allclose(kf.attitude, expected_q, atol=1e-10)

    def test_covariance_remains_symmetric(self):
        """Covariance must stay symmetric after predict."""
        kf = ESKF(sigma_g=0.01, sigma_a=0.05, sigma_bg=0.001, sigma_ba=0.01)
        rng = np.random.default_rng(42)
        for _ in range(20):
            w_m = rng.normal(scale=0.1, size=3)
            a_m = np.array([0.0, 0.0, -9.81]) + rng.normal(scale=0.5, size=3)
            kf.predict(w_m, a_m, dt=0.01)
            P = kf.covariance
            np.testing.assert_allclose(P, P.T, atol=1e-12)


class TestGPSUpdate:
    def test_update_reduces_covariance(self):
        """Measurement update should reduce covariance."""
        kf = ESKF()
        # First, add some uncertainty via predict
        kf.predict(np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, -9.81]), dt=0.01)
        P_before = kf.covariance.copy()

        kf.update_gps(
            p_meas=np.array([0.0, 0.0, 0.0]),
            v_meas=np.array([0.0, 0.0, 0.0]),
        )
        P_after = kf.covariance.copy()
        assert np.trace(P_after) < np.trace(P_before)

    def test_update_corrects_position(self):
        """GPS should pull position estimate toward measurement."""
        kf = ESKF(
            init_p=np.array([5.0, 0.0, 0.0]),
            init_P_diag=np.full(15, 1.0),
        )
        # Measurement says we're at origin with low noise → should pull there
        kf.update_gps(
            p_meas=np.array([0.0, 0.0, 0.0]),
            v_meas=np.array([0.0, 0.0, 0.1]),
            sigma_pos=0.01,
            sigma_vel=0.01,
        )
        # Position should have moved toward measurement
        assert abs(kf.position[0]) < 5.0  # moved toward 0

    def test_update_covariance_remains_symmetric(self):
        kf = ESKF()
        kf.predict(np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, -9.81]), dt=0.01)
        kf.update_gps(
            p_meas=np.array([0.0, 0.0, 0.0]),
            v_meas=np.array([0.0, 0.0, 0.0]),
        )
        P = kf.covariance
        np.testing.assert_allclose(P, P.T, atol=1e-12)


class TestBaroUpdate:
    def test_update_reduces_covariance(self):
        kf = ESKF()
        kf.predict(np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, -9.81]), dt=0.01)
        P_before = kf.covariance.copy()

        kf.update_baro(alt_meas=0.0)  # altitude = –p_z
        P_after = kf.covariance.copy()
        assert np.trace(P_after) < np.trace(P_before)

    def test_update_corrects_altitude(self):
        kf = ESKF(
            init_p=np.array([0.0, 0.0, -10.0]),  # 10m altitude
            init_P_diag=np.full(15, 1.0),
        )
        # Measurement says altitude = 5m with low noise
        kf.update_baro(alt_meas=5.0, sigma_alt=0.01)
        # p_z should move toward -5
        assert abs(kf.position[2] + 5.0) < 10.0  # moved toward -5


class TestErrorStateDimension:
    def test_dimension(self):
        assert ErrorState.DIM == 15


class TestStdAccessors:
    def test_positive_stds(self):
        kf = ESKF()
        pos_std = kf.get_position_std()
        vel_std = kf.get_velocity_std()
        att_std = kf.get_attitude_std()

        assert pos_std.shape == (3,)
        assert vel_std.shape == (3,)
        assert att_std.shape == (3,)
        assert np.all(pos_std >= 0)
        assert np.all(vel_std >= 0)
        assert np.all(att_std >= 0)

    def test_stds_decrease_after_update(self):
        kf = ESKF()
        kf.predict(np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, -9.81]), dt=0.01)
        pos_before = kf.get_position_std().copy()

        kf.update_gps(p_meas=np.array([0.0, 0.0, 0.0]), v_meas=np.array([0.0, 0.0, 0.0]))
        pos_after = kf.get_position_std()

        assert np.all(pos_after <= pos_before)

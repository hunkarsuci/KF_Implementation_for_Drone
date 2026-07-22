"""Tests for kf_drone.sensors — IMU, GPS, and barometer models."""

import numpy as np

from kf_drone.sensors import (
    BaroParams,
    GPSParams,
    IMUParams,
    compute_true_imu,
    generate_baro_measurement,
    generate_gps_measurement,
    generate_imu_measurement,
)
from kf_drone.utils import quat_exp


class TestIMUParams:
    def test_defaults(self):
        p = IMUParams()
        assert p.sigma_g == 0.01
        assert p.sigma_a == 0.05
        assert p.sigma_bg == 0.0002
        assert p.sigma_ba == 0.001
        np.testing.assert_allclose(p.b_g_init, [0, 0, 0])
        np.testing.assert_allclose(p.b_a_init, [0, 0, 0])

    def test_custom_biases(self):
        b_g = np.array([0.1, -0.2, 0.3])
        b_a = np.array([0.05, 0.0, -0.1])
        p = IMUParams(b_g_init=b_g, b_a_init=b_a)
        np.testing.assert_allclose(p.b_g_init, b_g)
        np.testing.assert_allclose(p.b_a_init, b_a)
        # Original arrays must not be mutated
        b_g[0] = 99.0
        assert p.b_g_init[0] == 0.1


class TestGPSParams:
    def test_defaults(self):
        p = GPSParams()
        assert p.sigma_pos == 1.0
        assert p.sigma_vel == 0.1


class TestBaroParams:
    def test_defaults(self):
        p = BaroParams()
        assert p.sigma_alt == 0.5


class TestIMUMeasurement:
    def test_no_noise_no_bias(self):
        """With zero noise and zero initial biases, measurement = true."""
        rng = np.random.default_rng(123)
        params = IMUParams(sigma_g=0.0, sigma_a=0.0, sigma_bg=0.0, sigma_ba=0.0)
        true_w = np.array([0.1, 0.2, 0.3])
        true_a = np.array([1.0, 2.0, 3.0])
        b_g = np.zeros(3)
        b_a = np.zeros(3)

        w_m, a_m, b_g_new, b_a_new = generate_imu_measurement(
            true_w, true_a, b_g, b_a, params, dt=0.01, rng=rng
        )
        np.testing.assert_allclose(w_m, true_w, atol=1e-12)
        np.testing.assert_allclose(a_m, true_a, atol=1e-12)
        np.testing.assert_allclose(b_g_new, b_g, atol=1e-12)
        np.testing.assert_allclose(b_a_new, b_a, atol=1e-12)

    def test_bias_random_walk_accumulates(self):
        """Bias random walk changes the bias each step."""
        rng = np.random.default_rng(456)
        params = IMUParams(
            sigma_g=0.0, sigma_a=0.0, sigma_bg=0.1, sigma_ba=0.2
        )
        true_w = np.zeros(3)
        true_a = np.zeros(3)
        b_g = np.zeros(3)
        b_a = np.zeros(3)

        for _ in range(50):
            _, _, b_g, b_a = generate_imu_measurement(
                true_w, true_a, b_g, b_a, params, dt=0.01, rng=rng
            )

        # After 50 steps, biases should have drifted from zero
        assert np.linalg.norm(b_g) > 0.001
        assert np.linalg.norm(b_a) > 0.001

    def test_output_shapes(self):
        rng = np.random.default_rng(789)
        params = IMUParams()
        b_g = np.zeros(3)
        b_a = np.zeros(3)
        w_m, a_m, b_g_new, b_a_new = generate_imu_measurement(
            np.zeros(3), np.zeros(3), b_g, b_a, params, dt=0.01, rng=rng
        )
        assert w_m.shape == (3,)
        assert a_m.shape == (3,)
        assert b_g_new.shape == (3,)
        assert b_a_new.shape == (3,)

    def test_small_dt_large_noise(self):
        """Small dt amplifies discrete measurement noise (σ/√dt)."""
        rng = np.random.default_rng(111)
        params = IMUParams(sigma_g=0.1, sigma_a=0.5, sigma_bg=0.0, sigma_ba=0.0)
        true_w = np.zeros(3)
        true_a = np.zeros(3)
        b_g = np.zeros(3)
        b_a = np.zeros(3)

        # Very small dt → large noise amplification
        w_m, a_m, _, _ = generate_imu_measurement(
            true_w, true_a, b_g, b_a, params, dt=0.001, rng=rng
        )
        # The noise should be significant
        assert np.any(np.abs(w_m) > 0.5)
        assert np.any(np.abs(a_m) > 1.0)


class TestGPSMeasurement:
    def test_no_noise(self):
        rng = np.random.default_rng(222)
        params = GPSParams(sigma_pos=0.0, sigma_vel=0.0)
        true_p = np.array([10.0, 20.0, -30.0])
        true_v = np.array([1.0, -2.0, 0.5])
        p_m, v_m = generate_gps_measurement(true_p, true_v, params, rng)
        np.testing.assert_allclose(p_m, true_p, atol=1e-12)
        np.testing.assert_allclose(v_m, true_v, atol=1e-12)

    def test_noise_is_additive(self):
        rng = np.random.default_rng(333)
        params = GPSParams(sigma_pos=5.0, sigma_vel=2.0)
        true_p = np.zeros(3)
        true_v = np.zeros(3)

        # With large noise, measurement should differ from truth
        diffs_p = []
        diffs_v = []
        for _ in range(50):
            p_m, v_m = generate_gps_measurement(true_p, true_v, params, rng)
            diffs_p.append(np.linalg.norm(p_m))
            diffs_v.append(np.linalg.norm(v_m))

        avg_p_norm = np.mean(diffs_p)
        avg_v_norm = np.mean(diffs_v)
        # Expected: sqrt(3) * sigma on average (vector norm of 3 uncorrelated Gaussians)
        assert avg_p_norm > 0.5
        assert avg_v_norm > 0.5


class TestBaroMeasurement:
    def test_no_noise(self):
        rng = np.random.default_rng(444)
        params = BaroParams(sigma_alt=0.0)
        # NED: p_z = -10 means 10m altitude
        alt = generate_baro_measurement(np.array([1.0, 2.0, -10.0]), params, rng)
        np.testing.assert_allclose(alt, 10.0, atol=1e-12)

    def test_altitude_sign(self):
        """Altitude = -p_z always."""
        rng = np.random.default_rng(555)
        params = BaroParams(sigma_alt=0.0)
        alt = generate_baro_measurement(np.array([0.0, 0.0, 5.0]), params, rng)
        # p_z = 5 (down), so altitude = -5
        np.testing.assert_allclose(alt, -5.0, atol=1e-12)


class TestComputeTrueIMU:
    def test_hovering(self):
        """Hover: zero acceleration, gravity alone → specific force = Rᵀ·g."""
        p = np.zeros(3)
        v = np.zeros(3)
        q = quat_exp(np.array([0.1, 0.2, 0.3]))  # arbitrary attitude
        a_body = np.zeros(3)
        w_body = np.array([0.0, 0.0, 0.1])

        sf, w = compute_true_imu(p, v, q, a_body, w_body)
        np.testing.assert_allclose(w, w_body)
        # Specific force in body frame should be -g_body
        from kf_drone.utils import quat_to_rotmat
        R = quat_to_rotmat(q)
        g_w = np.array([0.0, 0.0, 9.81])
        expected_sf = -R.T @ g_w
        np.testing.assert_allclose(sf, expected_sf, atol=1e-12)

    def test_accelerating_up(self):
        """Accelerating upward reduces the specific force magnitude."""
        p = np.zeros(3)
        v = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])  # level attitude
        a_body = np.array([0.0, 0.0, -5.0])  # accelerating up in NED (z negative = up)
        w_body = np.zeros(3)

        sf, _ = compute_true_imu(p, v, q, a_body, w_body)
        # In level flight, body = world. g_w = [0, 0, 9.81], a_body = [0, 0, -5]
        # sf = a_body - Rᵀ·g_w = [0, 0, -5] - [0, 0, 9.81] = [0, 0, -14.81]
        np.testing.assert_allclose(sf, [0, 0, -14.81], atol=1e-10)

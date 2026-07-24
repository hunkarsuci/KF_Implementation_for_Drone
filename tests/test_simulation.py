"""Tests for kf_drone.simulation — trajectory and sensor generation."""

import numpy as np

from kf_drone.simulation import (
    generate_sensor_data,
    generate_trajectory,
    run_simulation,
)


class TestGenerateTrajectory:
    def test_output_shapes(self):
        t = np.arange(0, 10, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)
        N = len(t)
        assert traj["t"].shape == (N,)
        assert traj["p"].shape == (N, 3)
        assert traj["v"].shape == (N, 3)
        assert traj["a"].shape == (N, 3)
        assert traj["q"].shape == (N, 4)
        assert traj["w_body"].shape == (N, 3)
        assert traj["a_body"].shape == (N, 3)
        assert traj["sf_body"].shape == (N, 3)

    def test_quaternions_unit_norm(self):
        t = np.arange(0, 5, 0.05, dtype=np.float64)
        traj = generate_trajectory(t)
        norms = np.linalg.norm(traj["q"], axis=1)
        np.testing.assert_allclose(norms, np.ones(len(t)), atol=1e-12)

    def test_velocity_is_position_derivative(self):
        """Velocity should be the derivative of position (check via finite diff)."""
        t = np.arange(0, 10, 0.001, dtype=np.float64)
        traj = generate_trajectory(t)

        # Finite-difference velocity
        p = traj["p"]
        v_fd = np.gradient(p, t, axis=0)

        # Should match analytic velocity closely (small dt)
        np.testing.assert_allclose(traj["v"], v_fd, atol=0.02)

    def test_acceleration_is_velocity_derivative(self):
        """Acceleration should be the derivative of velocity."""
        t = np.arange(0, 10, 0.001, dtype=np.float64)
        traj = generate_trajectory(t)

        a_fd = np.gradient(traj["v"], t, axis=0)
        np.testing.assert_allclose(traj["a"], a_fd, atol=0.03)

    def test_angular_velocity_consistent_with_attitude(self):
        """Angular velocity norm matches attitude change rate."""
        t = np.arange(0, 5, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)

        # Verify w_body is non-zero (drone is maneuvering)
        w_norms = np.linalg.norm(traj["w_body"], axis=1)
        max_w = np.max(w_norms)
        mean_w = np.mean(w_norms)

        # Drone should have meaningful rotation during the figure-8
        assert max_w > 0.1, f"Expected significant angular velocity, max={max_w:.4f}"
        assert mean_w > 0.01, f"Expected non-trivial mean angular velocity, mean={mean_w:.4f}"

    def test_quaternion_integration_consistency(self):
        """Check q_{i+1} ≈ q_i ⊗ exp(w_i * dt) for a few sample points."""
        from kf_drone.utils import quat_exp, quat_multiply, quat_normalize

        t = np.arange(0, 5, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)
        dt_val = float(t[1] - t[0])

        # Test a few interior points where central diff is most accurate
        errors = []
        for i in [50, 100, 150, 200, 250, 300, 350, 400]:
            q_pred = quat_multiply(
                traj["q"][i],
                quat_exp(traj["w_body"][i] * dt_val),
            )
            q_pred = quat_normalize(q_pred)
            q_actual = traj["q"][i + 1]
            err = min(
                np.linalg.norm(q_pred - q_actual),
                np.linalg.norm(q_pred + q_actual),
            )
            errors.append(err)

        median_err = np.median(errors)
        # Integration error should be small for fine dt (O(dt²) ≈ 1e-4)
        assert median_err < 0.02, f"Median quaternion integration error: {median_err:.6f}"

    def test_specific_force_reasonable(self):
        """Specific force should be on the order of g (~10 m/s²) for normal flight."""
        t = np.arange(0, 5, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)
        sf_norms = np.linalg.norm(traj["sf_body"], axis=1)
        # Should be close to g (~9.81) for most of the trajectory
        assert np.all(sf_norms > 5.0)
        assert np.all(sf_norms < 25.0)


class TestGenerateSensorData:
    def test_output_shapes(self):
        t = np.arange(0, 1, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)
        sensors = generate_sensor_data(traj, gps_rate=10, baro_rate=50)

        N = len(t)
        assert sensors["w_meas"].shape == (N, 3)
        assert sensors["a_meas"].shape == (N, 3)
        assert sensors["b_g_true"].shape == (N, 3)
        assert sensors["b_a_true"].shape == (N, 3)
        # GPS at 10 Hz for 1s ≈ 10 samples
        assert 5 <= len(sensors["gps_t"]) <= 15
        # Baro at 50 Hz for 1s ≈ 50 samples
        assert 30 <= len(sensors["baro_t"]) <= 70

    def test_imu_measurements_contain_noise(self):
        t = np.arange(0, 1, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)
        rng = np.random.default_rng(42)

        sensors1 = generate_sensor_data(traj, rng=rng)
        sensors2 = generate_sensor_data(traj, rng=np.random.default_rng(99))

        # Different seeds → different measurements
        diff = np.max(np.abs(sensors1["w_meas"] - sensors2["w_meas"]))
        assert diff > 1e-6

    def test_gps_subsampling(self):
        t = np.arange(0, 2, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)
        sensors = generate_sensor_data(traj, gps_rate=5)
        # 5 Hz for 2s ≈ 10 samples
        assert 5 <= len(sensors["gps_t"]) <= 15

    def test_baro_subsampling(self):
        t = np.arange(0, 2, 0.01, dtype=np.float64)
        traj = generate_trajectory(t)
        sensors = generate_sensor_data(traj, baro_rate=25)
        # 25 Hz for 2s ≈ 50 samples
        assert 40 <= len(sensors["baro_t"]) <= 60


class TestRunSimulation:
    def test_returns_merged_dict(self):
        result = run_simulation(duration=1.0, dt=0.01, seed=42)
        # Should contain both trajectory and sensor keys
        for key in [
            "t",
            "p",
            "v",
            "a",
            "q",
            "w_body",
            "a_body",
            "sf_body",
            "w_meas",
            "a_meas",
            "b_g_true",
            "b_a_true",
            "gps_t",
            "gps_p",
            "gps_v",
            "baro_t",
            "baro_alt",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_reproducible_with_seed(self):
        r1 = run_simulation(duration=1.0, dt=0.01, seed=42)
        r2 = run_simulation(duration=1.0, dt=0.01, seed=42)
        np.testing.assert_allclose(r1["p"], r2["p"])
        np.testing.assert_allclose(r1["w_meas"], r2["w_meas"])
        np.testing.assert_allclose(r1["a_meas"], r2["a_meas"])

    def test_different_seeds_different_sensors(self):
        r1 = run_simulation(duration=1.0, dt=0.01, seed=42)
        r2 = run_simulation(duration=1.0, dt=0.01, seed=99)
        diff = np.max(np.abs(r1["w_meas"] - r2["w_meas"]))
        assert diff > 1e-6

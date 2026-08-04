"""Tests for examples/evaluate_filter — metrics computation and reproducibility."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make evaluate_filter importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from evaluate_filter import DEFAULT_CONFIG, attitude_error_deg, compute_metrics


class TestAttitudeErrorDeg:
    def test_identity_returns_zero(self):
        q_id = np.array([[1.0, 0.0, 0.0, 0.0]])
        err = attitude_error_deg(q_id, q_id)
        np.testing.assert_allclose(err, np.zeros((1, 3)), atol=1e-12)

    def test_90deg_z_rotation(self):
        """90° rotation about Z → 90° error."""
        from kf_drone.utils import quat_exp

        q_true = np.array([quat_exp(np.array([0.0, 0.0, np.pi / 2]))])
        q_est = np.array([[1.0, 0.0, 0.0, 0.0]])
        err = attitude_error_deg(q_true, q_est)
        np.testing.assert_allclose(err, np.array([[0.0, 0.0, 90.0]]), atol=1e-10)

    def test_sign_ambiguity_handled(self):
        """q and -q are same rotation → zero error."""
        q = np.array([[0.5, 0.5, 0.5, 0.5]])
        q_neg = np.array([[-0.5, -0.5, -0.5, -0.5]])
        err = attitude_error_deg(q, q_neg)
        np.testing.assert_allclose(err, np.zeros((1, 3)), atol=1e-10)

    def test_shape(self):
        rng = np.random.default_rng(99)
        from kf_drone.utils import quat_exp, quat_normalize

        q_true = np.array([quat_normalize(quat_exp(rng.normal(size=3))) for _ in range(50)])
        q_est = np.array([quat_normalize(quat_exp(rng.normal(size=3))) for _ in range(50)])
        err = attitude_error_deg(q_true, q_est)
        assert err.shape == (50, 3)


class TestComputeMetrics:
    def test_zero_error_all_channels(self):
        """Perfect estimation → all RMSE = 0."""
        N = 500
        tail = 0.2
        sim = {
            "t": np.arange(N, dtype=np.float64) * 0.01,
            "p": np.zeros((N, 3)),
            "v": np.zeros((N, 3)),
            "q": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1)),
            "b_g_true": np.zeros((N, 3)),
            "b_a_true": np.zeros((N, 3)),
        }
        hist = {
            "p_est": np.zeros((N, 3)),
            "v_est": np.zeros((N, 3)),
            "q_est": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1)),
            "b_g_est": np.zeros((N, 3)),
            "b_a_est": np.zeros((N, 3)),
        }
        metrics = compute_metrics(sim, hist, tail)
        s = metrics["summary"]
        assert s["position_rmse_m"] == 0.0
        assert s["velocity_rmse_ms"] == 0.0
        assert s["attitude_total_rmse_deg"] == 0.0

    def test_known_position_error(self):
        """Constant 1 m error on all axes → 3D RMSE = sqrt(3)."""
        N = 500
        tail = 0.2
        sim = {
            "t": np.arange(N, dtype=np.float64) * 0.01,
            "p": np.zeros((N, 3)),
            "v": np.zeros((N, 3)),
            "q": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1)),
            "b_g_true": np.zeros((N, 3)),
            "b_a_true": np.zeros((N, 3)),
        }
        hist = {
            "p_est": np.ones((N, 3)),
            "v_est": np.zeros((N, 3)),
            "q_est": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1)),
            "b_g_est": np.zeros((N, 3)),
            "b_a_est": np.zeros((N, 3)),
        }
        metrics = compute_metrics(sim, hist, tail)
        s = metrics["summary"]
        np.testing.assert_allclose(s["position_rmse_m"], np.sqrt(3.0), atol=1e-12)
        # Individual axes
        assert metrics["position"]["north"]["rmse"] == 1.0
        assert metrics["position"]["east"]["rmse"] == 1.0
        assert metrics["position"]["down"]["rmse"] == 1.0

    def test_tail_fraction_respected(self):
        """Only the tail portion is used for steady-state metrics."""
        N = 500
        # First 400 steps have 1 m error, last 100 have 0 error
        sim = {
            "t": np.arange(N, dtype=np.float64) * 0.01,
            "p": np.zeros((N, 3)),
            "v": np.zeros((N, 3)),
            "q": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1)),
            "b_g_true": np.zeros((N, 3)),
            "b_a_true": np.zeros((N, 3)),
        }
        hist = {
            "p_est": np.ones((N, 3)),
            "v_est": np.zeros((N, 3)),
            "q_est": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1)),
            "b_g_est": np.zeros((N, 3)),
            "b_a_est": np.zeros((N, 3)),
        }
        hist["p_est"][400:] = 0.0  # perfect in tail

        # tail_fraction=0.2 → last 100 steps → RMSE should be 0
        metrics = compute_metrics(sim, hist, 0.2)
        assert metrics["summary"]["position_rmse_m"] == 0.0

        # tail_fraction=0.5 → last 250 steps → some error still
        metrics = compute_metrics(sim, hist, 0.5)
        assert metrics["summary"]["position_rmse_m"] > 0.0


class TestEvaluateFilterEndToEnd:
    """Minimal end-to-end test of evaluate_filter with a short simulation."""

    def test_short_run_returns_valid_metrics(self):
        from evaluate_filter import compute_metrics, run_filter

        from kf_drone.simulation import run_simulation

        config = {
            "simulation": {
                "duration": 2.0,
                "dt": 0.01,
                "amplitude_xy": 10.0,
                "amplitude_z": 2.0,
                "omega_xy": 0.5,
                "omega_z": 1.0,
                "gps_rate": 10,
                "baro_rate": 50,
            },
            "sensors": {
                "sigma_g": 0.01,
                "sigma_a": 0.05,
                "sigma_bg": 0.0002,
                "sigma_ba": 0.001,
                "gps_sigma_pos": 1.0,
                "gps_sigma_vel": 0.1,
                "baro_sigma_alt": 0.5,
            },
            "filter": {
                "gravity": 9.81,
                "init_P_diag": [0.5] * 15,
                "init_pos_noise_std": 0.5,
                "init_vel_noise_std": 0.2,
            },
            "seed": 123,
            "tail_fraction": 0.5,
        }

        sim = run_simulation(
            duration=config["simulation"]["duration"],
            dt=config["simulation"]["dt"],
            seed=config["seed"],
        )
        hist = run_filter(sim, config)
        metrics = compute_metrics(sim, hist, config["tail_fraction"])

        # All summary metrics should be finite numbers
        for key, val in metrics["summary"].items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
            assert val >= 0, f"{key} is negative: {val}"

    def test_reproducible_with_same_seed(self):
        """Two runs with the same seed must produce identical metrics."""
        from evaluate_filter import compute_metrics, run_filter

        from kf_drone.simulation import run_simulation

        config = {
            "simulation": {
                "duration": 2.0,
                "dt": 0.01,
                "amplitude_xy": 10.0,
                "amplitude_z": 2.0,
                "omega_xy": 0.5,
                "omega_z": 1.0,
                "gps_rate": 10,
                "baro_rate": 50,
            },
            "sensors": {
                "sigma_g": 0.01,
                "sigma_a": 0.05,
                "sigma_bg": 0.0002,
                "sigma_ba": 0.001,
                "gps_sigma_pos": 1.0,
                "gps_sigma_vel": 0.1,
                "baro_sigma_alt": 0.5,
            },
            "filter": {
                "gravity": 9.81,
                "init_P_diag": [0.5] * 15,
                "init_pos_noise_std": 0.5,
                "init_vel_noise_std": 0.2,
            },
            "seed": 42,
            "tail_fraction": 0.5,
        }

        sim1 = run_simulation(duration=2.0, dt=0.01, seed=42)
        hist1 = run_filter(sim1, config)
        metrics1 = compute_metrics(sim1, hist1, 0.5)

        sim2 = run_simulation(duration=2.0, dt=0.01, seed=42)
        hist2 = run_filter(sim2, config)
        metrics2 = compute_metrics(sim2, hist2, 0.5)

        for key in metrics1["summary"]:
            assert metrics1["summary"][key] == pytest.approx(
                metrics2["summary"][key]
            ), f"Mismatch in {key}"


class TestEvaluateFilterOutput:
    """Test that evaluate_filter.py handles --output correctly."""

    def _get_artifact_dir(self):
        return tempfile.mkdtemp(prefix="eskf_test_")

    def test_no_output_creates_no_file(self):
        """Running without --output must not create any JSON file."""
        # Run via subprocess to test the full CLI path
        import subprocess

        script = str(Path(__file__).resolve().parent.parent / "examples" / "evaluate_filter.py")
        result = subprocess.run(
            [sys.executable, script, "--duration", "1", "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        # The script should NOT emit "Results written to:"
        assert "Results written to:" not in result.stdout

    def test_output_creates_valid_json(self):
        """Running with --output creates a valid JSON file."""
        import subprocess

        script = str(Path(__file__).resolve().parent.parent / "examples" / "evaluate_filter.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "result.json"
            result = subprocess.run(
                [sys.executable, script, "--duration", "1", "--seed", "42",
                 "--output", str(out_file)],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            assert result.returncode == 0
            assert "Results written to:" in result.stdout
            assert out_file.exists()

            data = json.loads(out_file.read_text(encoding="utf-8"))
            assert "metrics" in data
            assert "config" in data
            assert data["config"]["seed"] == 42

    def test_output_creates_parent_dir(self):
        """--output to a non-existent parent directory must succeed."""
        import subprocess

        script = str(Path(__file__).resolve().parent.parent / "examples" / "evaluate_filter.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "subdir" / "nested" / "result.json"
            result = subprocess.run(
                [sys.executable, script, "--duration", "1", "--seed", "42",
                 "--output", str(nested)],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            assert result.returncode == 0
            assert nested.exists()

    def test_json_metrics_match_returned_metrics(self):
        """The JSON metrics must match what compute_metrics + run_filter produce."""
        import subprocess

        from evaluate_filter import compute_metrics, run_filter

        from kf_drone.simulation import run_simulation

        script = str(Path(__file__).resolve().parent.parent / "examples" / "evaluate_filter.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "result.json"
            subprocess.run(
                [sys.executable, script, "--duration", "2", "--seed", "42",
                 "--output", str(out_file)],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            json_data = json.loads(out_file.read_text(encoding="utf-8"))

        # Recompute directly
        config = {
            "simulation": {
                "duration": 2.0, "dt": 0.01, "amplitude_xy": 10.0, "amplitude_z": 2.0,
                "omega_xy": 0.5, "omega_z": 1.0, "gps_rate": 10, "baro_rate": 50,
            },
            "sensors": {
                "sigma_g": 0.01, "sigma_a": 0.05, "sigma_bg": 0.0002, "sigma_ba": 0.001,
                "gps_sigma_pos": 1.0, "gps_sigma_vel": 0.1, "baro_sigma_alt": 0.5,
            },
            "filter": {
                "gravity": 9.81,
                "init_P_diag": [0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.05, 0.05, 0.1,
                                0.005, 0.005, 0.005, 0.02, 0.02, 0.02],
                "init_pos_noise_std": 0.5, "init_vel_noise_std": 0.2,
            },
            "seed": 42, "tail_fraction": 0.2,
        }
        sim = run_simulation(duration=2.0, dt=0.01, seed=42)
        hist = run_filter(sim, config)
        metrics = compute_metrics(sim, hist, 0.2)

        for key in metrics["summary"]:
            assert metrics["summary"][key] == pytest.approx(
                json_data["metrics"]["summary"][key],
                rel=1e-10,
            ), f"Mismatch in {key}"


class TestDefaultConfig:
    def test_config_is_valid(self):
        assert DEFAULT_CONFIG["seed"] == 42
        assert DEFAULT_CONFIG["simulation"]["duration"] == 30.0
        assert DEFAULT_CONFIG["simulation"]["dt"] == 0.01
        assert len(DEFAULT_CONFIG["filter"]["init_P_diag"]) == 15

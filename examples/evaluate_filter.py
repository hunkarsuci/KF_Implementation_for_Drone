#!/usr/bin/env python3
"""Deterministic ESKF evaluation: run simulation + filter, compute metrics, save results.

Produces a machine-readable JSON result file and prints a summary table.

Usage:
    python examples/evaluate_filter.py                        # print summary only
    python examples/evaluate_filter.py --duration 30 --seed 42
    python examples/evaluate_filter.py --output /tmp/results.json

The output JSON contains:
    config:     simulation and filter parameters
    metrics:    per-axis RMSE, mean, std for each error channel
    summary:    scalar RMSE values matching the README table
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kf_drone.filter import ESKF
from kf_drone.simulation import run_simulation
from kf_drone.utils import quat_inverse, quat_log, quat_multiply

# ---------------------------------------------------------------------------
# Default configuration (all values explicitly stated for reproducibility)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "simulation": {
        "duration": 30.0,
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
        "init_P_diag": [
            0.5,
            0.5,
            0.5,
            0.2,
            0.2,
            0.2,
            0.05,
            0.05,
            0.1,
            0.005,
            0.005,
            0.005,
            0.02,
            0.02,
            0.02,
        ],
        "init_pos_noise_std": 0.5,
        "init_vel_noise_std": 0.2,
    },
    "seed": 42,
    "tail_fraction": 0.2,
}


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def attitude_error_deg(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Attitude error as rotation vector (degrees).

    q_err = q_est⁻¹ ⊗ q_true, then δθ = quat_log(q_err).
    Shortest-path rotation is enforced by flipping q_err when w < 0.
    """
    N = q_true.shape[0]
    errors = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        q_err = quat_multiply(quat_inverse(q_est[i]), q_true[i])
        if q_err[0] < 0.0:
            q_err = -q_err
        errors[i] = np.degrees(quat_log(q_err))
    return errors


def compute_metrics(
    sim: dict,
    hist: dict,
    tail_fraction: float = 0.2,
) -> dict:
    """Compute per-axis and summary error metrics.

    Args:
        sim: Simulation output from run_simulation.
        hist: Filter output history dict with keys p_est, v_est, q_est,
              b_g_est, b_a_est, pos_std, vel_std, att_std.
        tail_fraction: Fraction of trajectory tail used for steady-state metrics.

    Returns:
        Dictionary with per-axis RMSE/mean/std and summary values.
    """
    t = sim["t"]
    N = len(t)
    tail_start = int((1.0 - tail_fraction) * N)
    tail = slice(tail_start, None)

    pos_err = hist["p_est"] - sim["p"]
    vel_err = hist["v_est"] - sim["v"]
    att_err = attitude_error_deg(sim["q"], hist["q_est"])
    att_err_total = np.linalg.norm(att_err, axis=1)
    bg_err = hist["b_g_est"] - sim["b_g_true"]
    ba_err = hist["b_a_est"] - sim["b_a_true"]

    def per_axis(err: np.ndarray, labels: list[str], unit: str) -> dict:
        result = {}
        for j, lbl in enumerate(labels):
            result[lbl] = {
                "rmse": float(np.sqrt(np.mean(err[tail, j] ** 2))),
                "mean": float(np.mean(err[tail, j])),
                "std": float(np.std(err[tail, j])),
                "unit": unit,
            }
        return result

    metrics = {
        "position": per_axis(pos_err, ["north", "east", "down"], "m"),
        "velocity": per_axis(vel_err, ["vn", "ve", "vd"], "m/s"),
        "attitude": per_axis(att_err, ["dth_x", "dth_y", "dth_z"], "deg"),
        "attitude_total": {
            "rmse": float(np.sqrt(np.mean(att_err_total[tail] ** 2))),
            "mean": float(np.mean(att_err_total[tail])),
            "std": float(np.std(att_err_total[tail])),
            "unit": "deg",
        },
        "gyro_bias": per_axis(bg_err, ["bg_x", "bg_y", "bg_z"], "rad/s"),
        "accel_bias": per_axis(ba_err, ["ba_x", "ba_y", "ba_z"], "m/s^2"),
        "summary": {
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(pos_err[tail] ** 2, axis=1)))),
            "velocity_rmse_ms": float(np.sqrt(np.mean(np.sum(vel_err[tail] ** 2, axis=1)))),
            "attitude_total_rmse_deg": float(np.sqrt(np.mean(att_err_total[tail] ** 2))),
            "gyro_bias_rmse_rads": float(np.sqrt(np.mean(np.sum(bg_err[tail] ** 2, axis=1)))),
            "accel_bias_rmse_ms2": float(np.sqrt(np.mean(np.sum(ba_err[tail] ** 2, axis=1)))),
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# Filter runner
# ---------------------------------------------------------------------------


def run_filter(sim: dict, config: dict) -> dict:
    """Step the ESKF through simulated sensor data.

    Returns a dict of estimation history arrays.
    """
    t = sim["t"]
    N = len(t)
    dt = float(t[1] - t[0])

    # Initialise filter with perturbed initial state
    rng_init = np.random.default_rng(config["seed"] + 1)
    fc = config["filter"]

    kf = ESKF(
        init_p=sim["p"][0] + rng_init.normal(scale=fc["init_pos_noise_std"], size=3),
        init_v=sim["v"][0] + rng_init.normal(scale=fc["init_vel_noise_std"], size=3),
        init_q=sim["q"][0],
        init_P_diag=np.array(fc["init_P_diag"], dtype=np.float64),
        gravity=fc["gravity"],
        sigma_g=config["sensors"]["sigma_g"],
        sigma_a=config["sensors"]["sigma_a"],
        sigma_bg=config["sensors"]["sigma_bg"],
        sigma_ba=config["sensors"]["sigma_ba"],
    )

    hist = {
        "p_est": np.empty((N, 3), dtype=np.float64),
        "v_est": np.empty((N, 3), dtype=np.float64),
        "q_est": np.empty((N, 4), dtype=np.float64),
        "b_g_est": np.empty((N, 3), dtype=np.float64),
        "b_a_est": np.empty((N, 3), dtype=np.float64),
        "pos_std": np.empty((N, 3), dtype=np.float64),
        "vel_std": np.empty((N, 3), dtype=np.float64),
        "att_std": np.empty((N, 3), dtype=np.float64),
    }

    gps_idx = 0
    baro_idx = 0

    for i in range(N):
        kf.predict(sim["w_meas"][i], sim["a_meas"][i], dt=dt)

        while gps_idx < len(sim["gps_t"]) and sim["gps_t"][gps_idx] <= t[i] + 1e-9:
            kf.update_gps(
                p_meas=sim["gps_p"][gps_idx],
                v_meas=sim["gps_v"][gps_idx],
                sigma_pos=config["sensors"]["gps_sigma_pos"],
                sigma_vel=config["sensors"]["gps_sigma_vel"],
            )
            gps_idx += 1

        while baro_idx < len(sim["baro_t"]) and sim["baro_t"][baro_idx] <= t[i] + 1e-9:
            kf.update_baro(
                alt_meas=sim["baro_alt"][baro_idx],
                sigma_alt=config["sensors"]["baro_sigma_alt"],
            )
            baro_idx += 1

        hist["p_est"][i] = kf.position
        hist["v_est"][i] = kf.velocity
        hist["q_est"][i] = kf.attitude
        hist["b_g_est"][i] = kf.gyro_bias
        hist["b_a_est"][i] = kf.accel_bias
        hist["pos_std"][i] = kf.get_position_std()
        hist["vel_std"][i] = kf.get_velocity_std()
        hist["att_std"][i] = kf.get_attitude_std()

    return hist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic ESKF evaluation — produces reproducible metrics."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_CONFIG["simulation"]["duration"],
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to this file.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=DEFAULT_CONFIG["tail_fraction"],
        help="Fraction of trajectory tail for steady-state metrics.",
    )
    args = parser.parse_args()

    # Build config from defaults + CLI overrides
    config = {
        "simulation": {**DEFAULT_CONFIG["simulation"], "duration": args.duration},
        "sensors": DEFAULT_CONFIG["sensors"],
        "filter": DEFAULT_CONFIG["filter"],
        "seed": args.seed,
        "tail_fraction": args.tail_fraction,
    }

    # Run simulation
    sim = run_simulation(
        duration=config["simulation"]["duration"],
        dt=config["simulation"]["dt"],
        amplitude_xy=config["simulation"]["amplitude_xy"],
        amplitude_z=config["simulation"]["amplitude_z"],
        omega_xy=config["simulation"]["omega_xy"],
        omega_z=config["simulation"]["omega_z"],
        gps_rate=config["simulation"]["gps_rate"],
        baro_rate=config["simulation"]["baro_rate"],
        seed=config["seed"],
    )

    # Run filter
    hist = run_filter(sim, config)

    # Compute metrics
    metrics = compute_metrics(sim, hist, config["tail_fraction"])

    # Assemble output
    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "config": config,
        "metrics": metrics,
    }

    # Print summary
    s = metrics["summary"]
    print("=" * 60)
    print("  ESKF Evaluation Results")
    print("=" * 60)
    print(f"  Duration:      {config['simulation']['duration']:.0f} s")
    print(f"  Seed:          {config['seed']}")
    print(f"  Tail fraction: {config['tail_fraction']:.0%}")
    print("-" * 60)
    print(f"  Position RMSE:      {s['position_rmse_m']:.4f} m")
    print(f"  Velocity RMSE:      {s['velocity_rmse_ms']:.4f} m/s")
    print(f"  Attitude RMSE:      {s['attitude_total_rmse_deg']:.2f} deg")
    print(f"  Gyro bias RMSE:     {s['gyro_bias_rmse_rads']:.6f} rad/s")
    print(f"  Accel bias RMSE:    {s['accel_bias_rmse_ms2']:.6f} m/s^2")
    print("=" * 60)

    # Output JSON only when --output is explicit
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nResults written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

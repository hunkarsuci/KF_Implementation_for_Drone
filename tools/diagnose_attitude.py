#!/usr/bin/env python3
"""Diagnostic: attitude-error decomposition across multiple seeds.

Characterises the attitude RMSE by decomposing it into:
  - Total quaternion geodesic angle (shortest-path rotation magnitude)
  - Rotation-vector components (dth_x, dth_y, dth_z in world NED frame)
  - Wrapped ZYX Euler-angle errors (roll, pitch, yaw)
  - Full-run RMSE and tail-20% RMSE
  - At least five deterministic random seeds

Euler-angle RMSE values are NOT an additive decomposition of the geodesic
metric.  They describe the same attitude error in a different
parametrisation and are affected by gimbal-lock singularities.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repository root: tools/ -> parent
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from evaluate_filter import DEFAULT_CONFIG, run_filter  # noqa: E402

from kf_drone.simulation import run_simulation  # noqa: E402
from kf_drone.utils import quat_inverse, quat_log, quat_multiply, quat_to_rotmat  # noqa: E402

# ---------------------------------------------------------------------------
# Attitude error decomposition
# ---------------------------------------------------------------------------


def rotation_vector_error_deg(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Attitude error as rotation vector (degrees).

    delta_theta = quat_log(q_est^{-1} * q_true). Shortest path enforced (w >= 0).

    This is the SAME quantity the ESKF error-state tracks. Each component
    represents a small rotation about the corresponding world-frame axis
    (North, East, Down in NED).  These are NOT Euler angles.
    """
    N = q_true.shape[0]
    err = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        q_err = quat_multiply(quat_inverse(q_est[i]), q_true[i])
        if q_err[0] < 0.0:
            q_err = -q_err
        err[i] = np.degrees(quat_log(q_err))
    return err


def geodesic_angle_deg(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Total quaternion geodesic angle (shortest-path, degrees).

    theta = 2 * arccos(|<q_true, q_est>|)
    where <,> is the 4D dot product. This is the magnitude of the
    rotation-vector error at each timestep.
    """
    N = q_true.shape[0]
    angles = np.empty(N, dtype=np.float64)
    for i in range(N):
        dot = np.abs(np.dot(q_true[i], q_est[i]))
        dot = np.clip(dot, -1.0, 1.0)
        angles[i] = 2.0 * np.degrees(np.arccos(dot))
    return angles


def rotation_matrix_to_euler_321(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to roll/pitch/yaw (intrinsic ZYX / 3-2-1).

    R is body->world: columns = body axes in world frame.
    Intrinsic ZYX: R = Rz(yaw)*Ry(pitch)*Rx(roll).

    Returns:
        [roll, pitch, yaw] in radians.
    """
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-10:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def euler_errors_deg(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Compute roll, pitch, yaw errors in degrees with wrapping to [-180, 180].

    The attitude error quaternion q_err = q_est^{-1} * q_true is converted to
    an intrinsic ZYX Euler sequence. These values are NOT an additive
    decomposition of the geodesic angle.
    """
    N = q_true.shape[0]
    errors = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        q_err = quat_multiply(quat_inverse(q_est[i]), q_true[i])
        if q_err[0] < 0.0:
            q_err = -q_err
        R_err = quat_to_rotmat(q_err)
        rpy = rotation_matrix_to_euler_321(R_err)
        errors[i] = np.degrees(np.arctan2(np.sin(rpy), np.cos(rpy)))
    return errors  # columns: roll, pitch, yaw


def rmse(x: np.ndarray, axis=None) -> float:
    return float(np.sqrt(np.mean(x**2, axis=axis)))


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 101112]
DURATION = 30.0
TAIL_FRACTION = 0.2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attitude-error diagnostic across multiple random seeds."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON diagnostics to this file (optional).",
    )
    args = parser.parse_args()

    all_results = {}

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"  Seed = {seed}")
        print(f"{'=' * 60}")

        config = {
            "simulation": {**DEFAULT_CONFIG["simulation"], "duration": DURATION},
            "sensors": DEFAULT_CONFIG["sensors"],
            "filter": DEFAULT_CONFIG["filter"],
            "seed": seed,
            "tail_fraction": TAIL_FRACTION,
        }

        sim = run_simulation(duration=DURATION, dt=0.01, seed=seed)
        hist = run_filter(sim, config)

        N = len(sim["t"])
        tail_start = int((1.0 - TAIL_FRACTION) * N)

        geod = geodesic_angle_deg(sim["q"], hist["q_est"])
        rotvec = rotation_vector_error_deg(sim["q"], hist["q_est"])
        euler = euler_errors_deg(sim["q"], hist["q_est"])

        full = {
            "geodesic_rmse": rmse(geod),
            "geodesic_mean": float(np.mean(geod)),
            "rotvec_rmse": [rmse(rotvec[:, j]) for j in range(3)],
            "rotvec_mean": [float(np.mean(rotvec[:, j])) for j in range(3)],
            "euler_rmse": [rmse(euler[:, j]) for j in range(3)],
            "euler_mean": [float(np.mean(euler[:, j])) for j in range(3)],
        }
        tail = {
            "geodesic_rmse": rmse(geod[tail_start:]),
            "geodesic_mean": float(np.mean(geod[tail_start:])),
            "rotvec_rmse": [rmse(rotvec[tail_start:, j]) for j in range(3)],
            "rotvec_mean": [float(np.mean(rotvec[tail_start:, j])) for j in range(3)],
            "euler_rmse": [rmse(euler[tail_start:, j]) for j in range(3)],
            "euler_mean": [float(np.mean(euler[tail_start:, j])) for j in range(3)],
        }
        all_results[str(seed)] = {"full": full, "tail": tail}

        labels = ["Roll (x)", "Pitch (y)", "Yaw (z)"]
        print(f"\n  Tail-{TAIL_FRACTION * 100:.0f}% RMSE:")
        print(f"    {'Metric':<18s}: {'Full':>8s}  {'Tail':>8s}")
        print(f"    {'-' * 36}")
        print(
            f"    {'Geodesic angle':<18s}:"
            f" {full['geodesic_rmse']:8.2f}  {tail['geodesic_rmse']:8.2f} deg"
        )
        print(
            f"    {'rot-vec dth_x (N)':<18s}:"
            f" {full['rotvec_rmse'][0]:8.2f}  {tail['rotvec_rmse'][0]:8.2f} deg"
        )
        print(
            f"    {'rot-vec dth_y (E)':<18s}:"
            f" {full['rotvec_rmse'][1]:8.2f}  {tail['rotvec_rmse'][1]:8.2f} deg"
        )
        print(
            f"    {'rot-vec dth_z (D)':<18s}:"
            f" {full['rotvec_rmse'][2]:8.2f}  {tail['rotvec_rmse'][2]:8.2f} deg"
        )
        print()
        for j, lbl in enumerate(labels):
            print(
                f"    Euler {lbl:<12s}:"
                f" {full['euler_rmse'][j]:8.2f}  {tail['euler_rmse'][j]:8.2f} deg"
            )

    # --- Cross-seed summary ---
    print(f"\n{'=' * 60}")
    print(f"  CROSS-SEED SUMMARY (tail {TAIL_FRACTION * 100:.0f}%)")
    print(f"{'=' * 60}")
    print(f"  Seeds: {SEEDS}")
    print()
    print("  Euler-angle RMSE values are NOT an additive decomposition of the")
    print("  geodesic metric. They express the same attitude error in a different")
    print("  parametrisation (ZYX intrinsic) subject to gimbal lock.")
    print()

    euler_labels = ["Roll (x)", "Pitch (y)", "Yaw (z)"]
    for j, lbl in enumerate(euler_labels):
        vals = [all_results[str(s)]["tail"]["euler_rmse"][j] for s in SEEDS]
        print(
            f"  Euler {lbl:<12s}: "
            f"mean={np.mean(vals):.2f}  min={np.min(vals):.2f}"
            f"  max={np.max(vals):.2f}  std={np.std(vals):.2f} deg"
        )

    rv_labels = ["dth_x (N)", "dth_y (E)", "dth_z (D)"]
    for j, lbl in enumerate(rv_labels):
        vals = [all_results[str(s)]["tail"]["rotvec_rmse"][j] for s in SEEDS]
        print(
            f"  Rot-vec {lbl:<12s}: "
            f"mean={np.mean(vals):.2f}  min={np.min(vals):.2f}"
            f"  max={np.max(vals):.2f}  std={np.std(vals):.2f} deg"
        )

    geod_vals = [all_results[str(s)]["tail"]["geodesic_rmse"] for s in SEEDS]
    print(
        f"  Geodesic total  : "
        f"mean={np.mean(geod_vals):.2f}  min={np.min(geod_vals):.2f}"
        f"  max={np.max(geod_vals):.2f}  std={np.std(geod_vals):.2f} deg"
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nDiagnostics written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

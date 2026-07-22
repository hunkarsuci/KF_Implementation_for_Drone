#!/usr/bin/env python3
"""ESKF drone demo: run simulation + filter, then animate the results.

Produces a multi‑panel animation showing:
  • 3D true vs. estimated trajectory
  • Position error (per axis) over time
  • Velocity error (per axis) over time
  • Attitude error (per axis) over time

Usage:
    python examples/animate_demo.py               # run & show animation
    python examples/animate_demo.py --save demo.mp4  # save to file
    python examples/animate_demo.py --no-animate   # just run filter, print stats
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kf_drone.filter import ESKF
from kf_drone.sensors import BaroParams, GPSParams
from kf_drone.simulation import run_simulation

# ---------------------------------------------------------------------------
# Run filter on simulated data
# ---------------------------------------------------------------------------


def run_filter(sim: dict, kf: ESKF) -> dict:
    """Step the ESKF through the simulated sensor data.

    Returns a dict of estimation history arrays.
    """
    t = sim["t"]
    N = len(t)

    # Storage for estimates
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

    # GPS update indices
    gps_idx = 0
    gps_times = sim["gps_t"]

    # Baro update indices
    baro_idx = 0
    baro_times = sim["baro_t"]

    for i in range(N):
        ti = t[i]

        # Predict with IMU
        kf.predict(sim["w_meas"][i], sim["a_meas"][i], dt=float(t[1] - t[0]))

        # GPS update if available at this time
        while gps_idx < len(gps_times) and gps_times[gps_idx] <= ti + 1e-9:
            kf.update_gps(
                p_meas=sim["gps_p"][gps_idx],
                v_meas=sim["gps_v"][gps_idx],
                sigma_pos=GPSParams().sigma_pos,
                sigma_vel=GPSParams().sigma_vel,
            )
            gps_idx += 1

        # Barometer update if available at this time
        while baro_idx < len(baro_times) and baro_times[baro_idx] <= ti + 1e-9:
            kf.update_baro(
                alt_meas=sim["baro_alt"][baro_idx],
                sigma_alt=BaroParams().sigma_alt,
            )
            baro_idx += 1

        # Store estimates
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
# Compute attitude error
# ---------------------------------------------------------------------------


def attitude_error_deg(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Compute attitude error as a rotation vector (degrees).

    Uses ``quat_log(q_err)`` where ``q_err = q_est⁻¹ ⊗ q_true``.
    This is the singularity‑free rotation‑vector parametrisation — the same
    ``δθ`` that the ESKF error‑state tracks.  Each component represents
    rotation about the corresponding world (NED) axis in degrees.

    No Euler angles anywhere — no gimbal lock, no 180° jumps.
    """
    from kf_drone.utils import quat_inverse, quat_log, quat_multiply

    N = q_true.shape[0]
    errors = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        q_err = quat_multiply(quat_inverse(q_est[i]), q_true[i])
        # Ensure shortest rotation: q and −q represent the SAME rotation,
        # but quat_log(−q) can blow up to nearly 360°.  Forcing w ≥ 0
        # guarantees the rotation vector stays within [−180°, 180°].
        if q_err[0] < 0.0:
            q_err = -q_err
        errors[i] = np.degrees(quat_log(q_err))
    return errors


def attitude_error_total_deg(att_err_rotvec: np.ndarray) -> np.ndarray:
    """Total angular error magnitude (degrees) from rotation-vector error."""
    return np.linalg.norm(att_err_rotvec, axis=1)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_animation(
    sim: dict,
    hist: dict,
    save_path: str | None = None,
    playback_speed: float = 1.0,
) -> None:
    """Create and display (or save) the matplotlib animation.

    Args:
        sim: Simulation output from run_simulation.
        hist: Filter output from run_filter.
        save_path: If given, save to this file instead of displaying.
        playback_speed: Multiplier for playback speed (1 = real time).
    """
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    t = sim["t"]
    p_true = sim["p"]
    v_true = sim["v"]
    p_est = hist["p_est"]
    v_est = hist["v_est"]
    q_true = sim["q"]
    q_est = hist["q_est"]

    att_err = attitude_error_deg(q_true, q_est)
    pos_err = p_est - p_true
    vel_err = v_est - v_true

    # Subsampling for animation (show every step_subsample-th frame)
    step_subsample = max(1, int(playback_speed * 1))
    # Effective frame interval in seconds
    frame_dt = float(t[1] - t[0]) * step_subsample
    # Recompute fps
    fps = 1.0 / frame_dt if frame_dt > 0 else 30.0
    # Cap fps
    if fps > 60:
        step_subsample = max(1, int(1.0 / 60.0 / float(t[1] - t[0])))
        frame_dt = float(t[1] - t[0]) * step_subsample
        fps = 1.0 / frame_dt

    frame_indices = list(range(0, len(t), step_subsample))
    n_frames = len(frame_indices)

    # ---- Set up figure ----
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "ESKF Drone State Estimation — Figure‑8 Trajectory",
        fontsize=14,
        fontweight="bold",
    )

    # 3D trajectory
    ax_3d: Axes3D = fig.add_subplot(2, 2, 1, projection="3d")  # type: ignore[assignment]
    ax_3d.set_title("Trajectory (NED frame)")
    ax_3d.set_xlabel("North (m)")
    ax_3d.set_ylabel("East (m)")
    ax_3d.set_zlabel("Down (m)")

    # Position error
    ax_perr = fig.add_subplot(2, 2, 2)
    ax_perr.set_title("Position Error")
    ax_perr.set_xlabel("Time (s)")
    ax_perr.set_ylabel("Error (m)")
    ax_perr.grid(True, alpha=0.3)

    # Velocity error
    ax_verr = fig.add_subplot(2, 2, 3)
    ax_verr.set_title("Velocity Error")
    ax_verr.set_xlabel("Time (s)")
    ax_verr.set_ylabel("Error (m/s)")
    ax_verr.grid(True, alpha=0.3)

    # Attitude error (rotation vector)
    ax_aerr = fig.add_subplot(2, 2, 4)
    ax_aerr.set_title("Attitude Error (rotation vector δθ)")
    ax_aerr.set_xlabel("Time (s)")
    ax_aerr.set_ylabel("Error (deg)")
    ax_aerr.grid(True, alpha=0.3)

    # ---- Axis limits ----
    # 3D
    all_p = np.vstack([p_true, p_est])
    margin_3d = 1.0
    ax_3d.set_xlim(all_p[:, 0].min() - margin_3d, all_p[:, 0].max() + margin_3d)
    ax_3d.set_ylim(all_p[:, 1].min() - margin_3d, all_p[:, 1].max() + margin_3d)
    # Invert Z axis (down is positive in NED)
    ax_3d.set_zlim(
        all_p[:, 2].max() + margin_3d,
        all_p[:, 2].min() - margin_3d,
    )

    # Error plots
    t_lim = (t[0], t[-1])
    ax_perr.set_xlim(t_lim)
    ax_verr.set_xlim(t_lim)
    ax_aerr.set_xlim(t_lim)

    # Compute reasonable y-limits for errors
    max_perr = max(np.max(np.abs(pos_err)), 0.5)
    ax_perr.set_ylim(-max_perr * 1.2, max_perr * 1.2)
    max_verr = max(np.max(np.abs(vel_err)), 0.5)
    ax_verr.set_ylim(-max_verr * 1.2, max_verr * 1.2)
    max_aerr = max(np.max(np.abs(att_err)), 1.0)
    ax_aerr.set_ylim(-max_aerr * 1.2, max_aerr * 1.2)

    # ---- Static elements ----
    # Full true trajectory (faded)
    ax_3d.plot(
        p_true[:, 0], p_true[:, 1], p_true[:, 2],
        color="blue", alpha=0.3, linewidth=0.8, label="True",
    )

    # Full error traces (faded)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]  # R, G, B for x, y, z
    labels_xyz = ["X (North)", "Y (East)", "Z (Down)"]
    for j, (c, lbl) in enumerate(zip(colors, labels_xyz)):
        ax_perr.plot(t, pos_err[:, j], color=c, alpha=0.25, linewidth=0.6)
        ax_verr.plot(t, vel_err[:, j], color=c, alpha=0.25, linewidth=0.6)
        ax_aerr.plot(t, att_err[:, j], color=c, alpha=0.25, linewidth=0.6)

    # 3-sigma bounds
    for j, c in enumerate(colors):
        ax_perr.fill_between(
            t,
            -3 * hist["pos_std"][:, j],
            3 * hist["pos_std"][:, j],
            color=c, alpha=0.08,
        )
        ax_verr.fill_between(
            t,
            -3 * hist["vel_std"][:, j],
            3 * hist["vel_std"][:, j],
            color=c, alpha=0.08,
        )
        ax_aerr.fill_between(
            t,
            np.degrees(-3 * hist["att_std"][:, j]),
            np.degrees(3 * hist["att_std"][:, j]),
            color=c, alpha=0.08,
        )

    # Legend for error plots
    for ax_e, title in [(ax_perr, "Pos"), (ax_verr, "Vel"), (ax_aerr, "Att")]:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
        ax_e.legend(custom_lines, labels_xyz, loc="upper right", fontsize=7)

    # ---- Animated elements ----
    (line_est_3d,) = ax_3d.plot([], [], [], color="red", linewidth=1.5, label="Estimated")
    (dot_true_3d,) = ax_3d.plot([], [], [], "bo", markersize=4)
    (dot_est_3d,) = ax_3d.plot([], [], [], "r.", markersize=4)
    ax_3d.legend(loc="upper left", fontsize=7)

    # Progress lines on error plots
    progress_lines_perr = [
        ax_perr.plot([], [], color=c, linewidth=1.5)[0] for c in colors
    ]
    progress_lines_verr = [
        ax_verr.plot([], [], color=c, linewidth=1.5)[0] for c in colors
    ]
    progress_lines_aerr = [
        ax_aerr.plot([], [], color=c, linewidth=1.5)[0] for c in colors
    ]

    # Time indicators
    time_line_perr = ax_perr.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    time_line_verr = ax_verr.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    time_line_aerr = ax_aerr.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # ---- Animation function ----
    def animate(frame_idx: int):
        """Update all animated elements for frame frame_idx."""
        i = frame_indices[frame_idx]
        current_t = t[i]

        # 3D: estimated trajectory up to current frame
        est_slice = slice(0, i + 1, step_subsample)
        line_est_3d.set_data(
            p_est[est_slice, 0],
            p_est[est_slice, 1],
        )
        line_est_3d.set_3d_properties(p_est[est_slice, 2])
        dot_true_3d.set_data([p_true[i, 0]], [p_true[i, 1]])
        dot_true_3d.set_3d_properties([p_true[i, 2]])
        dot_est_3d.set_data([p_est[i, 0]], [p_est[i, 1]])
        dot_est_3d.set_3d_properties([p_est[i, 2]])

        # Error progress
        t_slice = t[: i + 1]
        for j in range(3):
            progress_lines_perr[j].set_data(t_slice, pos_err[: i + 1, j])
            progress_lines_verr[j].set_data(t_slice, vel_err[: i + 1, j])
            progress_lines_aerr[j].set_data(t_slice, att_err[: i + 1, j])

        # Time cursor
        time_line_perr.set_xdata([current_t, current_t])
        time_line_verr.set_xdata([current_t, current_t])
        time_line_aerr.set_xdata([current_t, current_t])

        fig.suptitle(
            f"ESKF Drone State Estimation — t = {current_t:.1f} s  "
            f"|  3-sig dth: ({np.degrees(3*hist['att_std'][i, 0]):.1f}°, "
            f"{np.degrees(3*hist['att_std'][i, 1]):.1f}°, "
            f"{np.degrees(3*hist['att_std'][i, 2]):.1f}°)",
            fontsize=12,
            fontweight="bold",
        )

        return (
            line_est_3d, dot_true_3d, dot_est_3d,
            *progress_lines_perr, *progress_lines_verr, *progress_lines_aerr,
            time_line_perr, time_line_verr, time_line_aerr,
        )

    # ---- Create animation ----
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        interval=max(1, int(frame_dt * 1000)),  # ms per frame
        blit=False,
        repeat=True,
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path} …")
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        ani.save(save_path, writer=writer)
        print("Done.")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ESKF drone demo with animation")
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save animation to file (e.g. demo.mp4) instead of showing interactively.",
    )
    parser.add_argument(
        "--no-animate", action="store_true",
        help="Skip animation; just run filter and print final statistics.",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Simulation duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1).",
    )
    args = parser.parse_args()

    print(f"Running {args.duration:.0f}s simulation (seed={args.seed}) …")
    sim = run_simulation(
        duration=args.duration,
        dt=0.01,
        seed=args.seed,
    )

    # Realistic initialisation:  after calibration, biases start near zero.
    # Position/velocity have moderate uncertainty, attitude is initialised
    # from accelerometer leveling (gravity = "down"), yaw is most uncertain.
    rng_init = np.random.default_rng(args.seed + 1)
    kf = ESKF(
        init_p=sim["p"][0] + rng_init.normal(scale=0.5, size=3),
        init_v=sim["v"][0] + rng_init.normal(scale=0.2, size=3),
        init_q=sim["q"][0],
        init_P_diag=np.array([
            0.5, 0.5, 0.5,     # pos
            0.2, 0.2, 0.2,     # vel
            0.05, 0.05, 0.1,    # att (yaw more uncertain)
            0.005, 0.005, 0.005,  # gyro bias
            0.02, 0.02, 0.02,     # accel bias
        ]),
    )

    print("Running ESKF …")
    hist = run_filter(sim, kf)

    # ---- Final statistics ----
    pos_err = hist["p_est"] - sim["p"]
    vel_err = hist["v_est"] - sim["v"]
    att_err = attitude_error_deg(sim["q"], hist["q_est"])

    print("\n" + "=" * 55)
    print("  FINAL ESTIMATION ERRORS (last 20 % of trajectory)")
    print("=" * 55)
    tail = slice(int(0.8 * len(sim["t"])), None)

    def print_stats(name: str, err: np.ndarray, unit: str, labels: list[str]) -> None:
        print(f"\n  {name}:")
        for j, lbl in enumerate(labels):
            rmse = np.sqrt(np.mean(err[tail, j] ** 2))
            mean_e = np.mean(err[tail, j])
            std_e = np.std(err[tail, j])
            print(f"    {lbl:>12s}:  RMSE = {rmse:8.4f} {unit}  "
                  f"Mean = {mean_e:+.4f} {unit}  Std = {std_e:.4f} {unit}")

    print_stats("Position Error", pos_err, "m", ["North", "East", "Down"])
    print_stats("Velocity Error", vel_err, "m/s", ["Vn", "Ve", "Vd"])
    att_total = attitude_error_total_deg(att_err)
    print_stats("Attitude Error (rot vec)", att_err, "deg",
                ["dth_x (N)", "dth_y (E)", "dth_z (D)"])
    print(f"\n    {'Total angle':>12s}:  RMSE = {np.sqrt(np.mean(att_total[tail]**2)):8.4f} deg"
          f"  Mean = {np.mean(att_total[tail]):+.4f} deg"
          f"  Std = {np.std(att_total[tail]):.4f} deg")

    # Bias estimation quality
    bg_err = hist["b_g_est"] - sim["b_g_true"]
    ba_err = hist["b_a_est"] - sim["b_a_true"]
    print_stats("Gyro Bias Error", bg_err, "rad/s", ["bg_x", "bg_y", "bg_z"])
    print_stats("Accel Bias Error", ba_err, "m/s²", ["ba_x", "ba_y", "ba_z"])

    print(f"\n  Final 3-sigma position uncertainty: {3 * hist['pos_std'][-1]}")
    print(f"  Final 3-sigma velocity uncertainty: {3 * hist['vel_std'][-1]}")
    print(f"  Final 3-sigma attitude uncertainty (deg): {np.degrees(3 * hist['att_std'][-1])}")

    if not args.no_animate:
        print("\nRendering animation …")
        create_animation(sim, hist, save_path=args.save, playback_speed=args.speed)


if __name__ == "__main__":
    main()

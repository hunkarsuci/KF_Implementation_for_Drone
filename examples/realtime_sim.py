#!/usr/bin/env python3
"""Replay the synthetic drone sensors in wall-clock time.

This example keeps the estimator API unchanged while delivering IMU, GPS, and
barometer samples as timestamped events. It is useful for exercising scheduling
and live visualization before connecting a real sensor driver.

Usage:
    python examples/realtime_sim.py
    python examples/realtime_sim.py --duration 30 --speed 2
    python examples/realtime_sim.py --duration 20 --plot
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the package importable when this file is run from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kf_drone.filter import ESKF
from kf_drone.sensors import BaroParams, GPSParams
from kf_drone.simulation import run_simulation


def make_filter(sim: dict, seed: int) -> ESKF:
    """Create the same intentionally perturbed initial estimate used by the demos."""
    rng = np.random.default_rng(seed + 1)
    return ESKF(
        init_p=sim["p"][0] + rng.normal(scale=0.5, size=3),
        init_v=sim["v"][0] + rng.normal(scale=0.2, size=3),
        init_q=sim["q"][0],
        init_P_diag=np.array(
            [
                0.5,
                0.5,
                0.5,  # position
                0.2,
                0.2,
                0.2,  # velocity
                0.05,
                0.05,
                0.1,  # attitude
                0.005,
                0.005,
                0.005,  # gyro bias
                0.02,
                0.02,
                0.02,  # accel bias
            ],
            dtype=np.float64,
        ),
    )


def make_plot(sim: dict):
    """Create an optional lightweight live plot."""
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax_traj, ax_error) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("ESKF real-time sensor replay")
    ax_traj.set_title("Trajectory (NED)")
    ax_traj.set_xlabel("North (m)")
    ax_traj.set_ylabel("East (m)")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_error.set_title("Position error norm")
    ax_error.set_xlabel("Simulation time (s)")
    ax_error.set_ylabel("Error (m)")
    ax_error.grid(True, alpha=0.3)

    p = sim["p"]
    margin = 1.0
    ax_traj.set_xlim(p[:, 0].min() - margin, p[:, 0].max() + margin)
    ax_traj.set_ylim(p[:, 1].min() - margin, p[:, 1].max() + margin)
    (true_line,) = ax_traj.plot([], [], "b-", alpha=0.5, label="true")
    (est_line,) = ax_traj.plot([], [], "r-", label="estimated")
    (true_dot,) = ax_traj.plot([], [], "bo", markersize=4)
    (est_dot,) = ax_traj.plot([], [], "ro", markersize=4)
    ax_traj.legend(loc="upper right", fontsize=8)
    (error_line,) = ax_error.plot([], [], "k-")
    return fig, true_line, est_line, true_dot, est_dot, error_line


def run_realtime(
    sim: dict,
    kf: ESKF,
    speed: float,
    plot: bool = False,
) -> dict:
    """Replay sensor events against wall-clock time.

    speed=1 is real time; speed=2 plays twice as fast. GPS and barometer
    events are consumed by timestamp, so their lower rates and asynchronous
    timing remain visible.
    """
    if speed <= 0.0:
        raise ValueError("speed must be greater than zero")

    live = make_plot(sim) if plot else None
    if live is not None:
        fig, true_line, est_line, true_dot, est_dot, error_line = live
    else:
        fig = true_line = est_line = true_dot = est_dot = error_line = None

    t = sim["t"]
    dt = float(t[1] - t[0])
    gps_idx = 0
    baro_idx = 0
    gps_updates = 0
    baro_updates = 0
    est_positions: list[np.ndarray] = []
    errors: list[float] = []
    next_status = 0.0
    wall_start = time.monotonic()

    print(
        f"Replaying {t[-1]:.1f}s of simulated sensors at {speed:g}x speed "
        f"(Ctrl-C to stop)"
    )
    try:
        for i, sim_t in enumerate(t):
            deadline = wall_start + float(sim_t) / speed
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

            kf.predict(sim["w_meas"][i], sim["a_meas"][i], dt=dt)

            while gps_idx < len(sim["gps_t"]) and sim["gps_t"][gps_idx] <= sim_t + 1e-9:
                kf.update_gps(
                    sim["gps_p"][gps_idx],
                    sim["gps_v"][gps_idx],
                    sigma_pos=GPSParams().sigma_pos,
                    sigma_vel=GPSParams().sigma_vel,
                )
                gps_idx += 1
                gps_updates += 1

            while baro_idx < len(sim["baro_t"]) and sim["baro_t"][baro_idx] <= sim_t + 1e-9:
                kf.update_baro(
                    sim["baro_alt"][baro_idx],
                    sigma_alt=BaroParams().sigma_alt,
                )
                baro_idx += 1
                baro_updates += 1

            p_est = kf.position
            est_positions.append(p_est)
            error = float(np.linalg.norm(p_est - sim["p"][i]))
            errors.append(error)

            if live is not None:
                p_est_arr = np.asarray(est_positions)
                true_line.set_data(sim["p"][: i + 1, 0], sim["p"][: i + 1, 1])
                est_line.set_data(p_est_arr[:, 0], p_est_arr[:, 1])
                true_dot.set_data([sim["p"][i, 0]], [sim["p"][i, 1]])
                est_dot.set_data([p_est[0]], [p_est[1]])
                error_line.set_data(t[: i + 1], errors)
                ax_error = error_line.axes
                ax_error.set_xlim(t[0], max(t[-1], 1.0))
                ax_error.set_ylim(0.0, max(1.0, max(errors) * 1.2))
                fig.canvas.draw_idle()
                import matplotlib.pyplot as plt

                plt.pause(0.001)

            if sim_t + 1e-9 >= next_status or i == len(t) - 1:
                print(
                    f"\r t={sim_t:6.2f}s | pos error={error:6.3f}m | "
                    f"GPS={gps_updates:4d} | baro={baro_updates:4d}",
                    end="",
                    flush=True,
                )
                next_status += 1.0

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    finally:
        if live is not None:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.show(block=False)

    print()
    return {
        "samples": len(est_positions),
        "gps_updates": gps_updates,
        "baro_updates": baro_updates,
        "final_position_error_m": errors[-1] if errors else float("nan"),
        "mean_position_error_m": float(np.mean(errors)) if errors else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay synthetic IMU/GPS/barometer data in wall-clock time."
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier: 1 is real time, 2 is twice as fast.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a live trajectory and position-error plot.",
    )
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    if args.speed <= 0.0:
        parser.error("--speed must be greater than zero")

    sim = run_simulation(duration=args.duration, dt=0.01, seed=args.seed)
    summary = run_realtime(sim, make_filter(sim, args.seed), args.speed, args.plot)
    print(
        "Completed "
        f"{summary['samples']} IMU steps, {summary['gps_updates']} GPS updates, "
        f"{summary['baro_updates']} barometer updates. "
        f"Mean position error: {summary['mean_position_error_m']:.3f} m."
    )


if __name__ == "__main__":
    main()

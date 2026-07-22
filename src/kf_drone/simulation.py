"""Drone trajectory generation and synthetic sensor data for ESKF testing.

Generates a parametric 3-DOF trajectory (position, velocity, acceleration,
attitude, angular velocity) and synthesises noisy IMU, GPS, and barometer
readings.
"""

import numpy as np
from numpy.typing import NDArray

from kf_drone.sensors import (
    BaroParams,
    GPSParams,
    IMUParams,
    generate_baro_measurement,
    generate_gps_measurement,
    generate_imu_measurement,
)
from kf_drone.utils import quat_normalize, rotmat_to_quat

# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def generate_trajectory(
    t: NDArray[np.float64],
    amplitude_xy: float = 10.0,
    amplitude_z: float = 2.0,
    omega_xy: float = 0.5,
    omega_z: float = 1.0,
) -> dict:
    """Generate a 3D drone trajectory (figure-8 in XY + sinusoidal Z).

    The horizontal motion is a Lissajous-like figure-8:
        x(t) = A * sin(ω_xy * t)
        y(t) = A * sin(2 * ω_xy * t)
        z(t) = A_z * sin(ω_z * t)

    Attitude is computed so that the body z-axis aligns with the required
    thrust direction (a_world - g), body x-axis aligns with the velocity
    component perpendicular to thrust.

    Args:
        t: Time samples (s), 1D array of length N.
        amplitude_xy: Horizontal amplitude (m).
        amplitude_z: Vertical amplitude (m).
        omega_xy: Horizontal angular frequency (rad/s).
        omega_z: Vertical angular frequency (rad/s).

    Returns:
        Dictionary with keys:
            t:       Time array (N,).
            p:       Position in world NED (N, 3).
            v:       Velocity in world NED (N, 3).
            a:       Acceleration in world NED (N, 3).
            q:       Attitude quaternion [w, x, y, z] body→world (N, 4).
            w_body:  Angular velocity in body frame (N, 3).
            a_body:  Acceleration in body frame (N, 3).
            sf_body: Specific force in body frame (N, 3).
    """
    N = len(t)
    g = np.array([0.0, 0.0, 9.81], dtype=np.float64)

    # --- Position ---
    x = amplitude_xy * np.sin(omega_xy * t)
    y = amplitude_xy * np.sin(2.0 * omega_xy * t)
    z = amplitude_z * np.sin(omega_z * t)
    p = np.column_stack([x, y, z])

    # --- Velocity ---
    vx = amplitude_xy * omega_xy * np.cos(omega_xy * t)
    vy = amplitude_xy * 2.0 * omega_xy * np.cos(2.0 * omega_xy * t)
    vz = amplitude_z * omega_z * np.cos(omega_z * t)
    v = np.column_stack([vx, vy, vz])

    # --- Acceleration ---
    ax = -amplitude_xy * omega_xy ** 2 * np.sin(omega_xy * t)
    ay = -amplitude_xy * 4.0 * omega_xy ** 2 * np.sin(2.0 * omega_xy * t)
    az = -amplitude_z * omega_z ** 2 * np.sin(omega_z * t)
    a = np.column_stack([ax, ay, az])

    # --- Attitude & angular velocity ---
    q = np.empty((N, 4), dtype=np.float64)
    w_body = np.empty((N, 3), dtype=np.float64)
    sf_body = np.empty((N, 3), dtype=np.float64)
    a_body = np.empty((N, 3), dtype=np.float64)

    for i in range(N):
        # Thrust direction in world frame = a - g
        thrust_world = a[i] - g
        thrust_norm = np.linalg.norm(thrust_world)
        if thrust_norm < 1e-9:
            thrust_world = g.copy()
            thrust_norm = np.linalg.norm(thrust_world)

        # Body z-axis (in world frame): thrust direction (unit vector)
        z_body_w = thrust_world / thrust_norm

        # Body x-axis: velocity component perpendicular to z_body, or default
        v_i = v[i]
        v_parallel = np.dot(v_i, z_body_w) * z_body_w
        v_perp = v_i - v_parallel
        v_perp_norm = np.linalg.norm(v_perp)
        if v_perp_norm > 1e-9:
            x_body_w = v_perp / v_perp_norm
        else:
            # Default: pick any vector perpendicular to z_body_w
            if abs(z_body_w[0]) < 0.9:
                x_body_w = np.cross(z_body_w, np.array([1.0, 0.0, 0.0]))
            else:
                x_body_w = np.cross(z_body_w, np.array([0.0, 1.0, 0.0]))
            x_body_w /= np.linalg.norm(x_body_w)

        # Body y-axis completes the right-handed frame
        y_body_w = np.cross(z_body_w, x_body_w)
        y_body_w /= np.linalg.norm(y_body_w)

        # Re-orthogonalise x
        x_body_w = np.cross(y_body_w, z_body_w)

        # Rotation matrix: columns are body axes in world frame
        R = np.column_stack([x_body_w, y_body_w, z_body_w])
        q[i] = quat_normalize(rotmat_to_quat(R))

        # Acceleration in body frame
        a_body[i] = R.T @ a[i]

        # Specific force in body frame
        sf_body[i] = R.T @ thrust_world

    # --- Angular velocity (finite differences on quaternion) ---
    for i in range(N):
        if i == 0:
            # Forward difference
            q0 = q[0]
            q1 = q[1]
            dt_val = float(t[1] - t[0])
            # q1 = q0 ⊗ exp(w * dt / 2)  ⇒  exp(w*dt/2) = q0⁻¹ ⊗ q1
            dq = _quat_mult(q0, q1, conjugate_first=True)
            w_body[i] = _quat_to_omega(dq, dt_val)
        elif i == N - 1:
            # Backward difference
            q_prev = q[N - 2]
            q_curr = q[N - 1]
            dt_val = float(t[N - 1] - t[N - 2])
            dq = _quat_mult(q_prev, q_curr, conjugate_first=True)
            w_body[i] = _quat_to_omega(dq, dt_val)
        else:
            # Central difference
            q_prev = q[i - 1]
            q_next = q[i + 1]
            dt_val = float(t[i + 1] - t[i - 1])
            dq = _quat_mult(q_prev, q_next, conjugate_first=True)
            w_body[i] = _quat_to_omega(dq, dt_val)

    return {
        "t": t,
        "p": p,
        "v": v,
        "a": a,
        "q": q,
        "w_body": w_body,
        "a_body": a_body,
        "sf_body": sf_body,
    }


# ---------------------------------------------------------------------------
# Helpers for quaternion-based angular velocity extraction
# ---------------------------------------------------------------------------


def _quat_mult(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
    conjugate_first: bool = False,
) -> NDArray[np.float64]:
    """Hamilton product q1 ⊗ q2 (or q1⁻¹ ⊗ q2 if conjugate_first)."""
    if conjugate_first:
        w1, x1, y1, z1 = q1[0], -q1[1], -q1[2], -q1[3]
    else:
        w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_omega(dq: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """Extract angular velocity from quaternion difference.

    dq = exp(w * dt)  ⇒  w = log(dq) / dt.
    """
    # Normalise to handle numerical drift
    dq = quat_normalize(dq)
    w, x, y, z = dq
    xyz = np.array([x, y, z])
    v_norm = np.linalg.norm(xyz)
    if v_norm < 1e-12:
        return np.zeros(3)
    w_clamped = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arctan2(v_norm, w_clamped)
    return angle * xyz / (v_norm * dt)


# ---------------------------------------------------------------------------
# Synthetic sensor generation
# ---------------------------------------------------------------------------


def generate_sensor_data(
    trajectory: dict,
    imu_params: IMUParams | None = None,
    gps_params: GPSParams | None = None,
    baro_params: BaroParams | None = None,
    rng: np.random.Generator | None = None,
    gps_rate: int = 10,
    baro_rate: int = 50,
) -> dict:
    """Generate noisy sensor measurements along a trajectory.

    IMU measurements are generated at every trajectory time step.
    GPS and barometer are generated at lower rates (subsampled).

    Args:
        trajectory: Output of :func:`generate_trajectory`.
        imu_params: IMU noise parameters.  Uses defaults if None.
        gps_params: GPS noise parameters.  Uses defaults if None.
        baro_params: Barometer noise parameters.  Uses defaults if None.
        rng: Random generator.  Uses default_rng if None.
        gps_rate: GPS output rate (Hz).  Subsampled from trajectory.
        baro_rate: Barometer output rate (Hz).  Subsampled from trajectory.

    Returns:
        Dictionary with keys:
            t:            Time array (N,).
            w_meas:       Gyro measurements (N, 3).
            a_meas:       Accel measurements (N, 3).
            b_g_true:     True gyro bias trajectory (N, 3).
            b_a_true:     True accel bias trajectory (N, 3).
            gps_t:        GPS measurement times.
            gps_p:        GPS position measurements (M_gps, 3).
            gps_v:        GPS velocity measurements (M_gps, 3).
            baro_t:       Barometer measurement times.
            baro_alt:     Barometer altitude measurements (M_baro,).
    """
    if rng is None:
        rng = np.random.default_rng()
    if imu_params is None:
        imu_params = IMUParams()
    if gps_params is None:
        gps_params = GPSParams()
    if baro_params is None:
        baro_params = BaroParams()

    t = trajectory["t"]
    N = len(t)
    dt = float(t[1] - t[0])

    imu_freq = 1.0 / dt

    w_meas = np.empty((N, 3), dtype=np.float64)
    a_meas = np.empty((N, 3), dtype=np.float64)
    b_g_true_arr = np.empty((N, 3), dtype=np.float64)
    b_a_true_arr = np.empty((N, 3), dtype=np.float64)

    # Initialise biases at the IMUParams defaults
    b_g = imu_params.b_g_init.copy()
    b_a = imu_params.b_a_init.copy()

    for i in range(N):
        w_m, a_m, b_g, b_a = generate_imu_measurement(
            trajectory["w_body"][i],
            trajectory["sf_body"][i],
            b_g,
            b_a,
            imu_params,
            dt,
            rng,
        )
        w_meas[i] = w_m
        a_meas[i] = a_m
        b_g_true_arr[i] = b_g
        b_a_true_arr[i] = b_a

    # --- GPS: subsample ---
    gps_step = max(1, int(imu_freq / gps_rate))
    gps_indices = list(range(0, N, gps_step))
    gps_t = t[gps_indices]
    gps_p = np.empty((len(gps_indices), 3), dtype=np.float64)
    gps_v = np.empty((len(gps_indices), 3), dtype=np.float64)
    for j, idx in enumerate(gps_indices):
        gps_p[j], gps_v[j] = generate_gps_measurement(
            trajectory["p"][idx],
            trajectory["v"][idx],
            gps_params,
            rng,
        )

    # --- Barometer: subsample ---
    baro_step = max(1, int(imu_freq / baro_rate))
    baro_indices = list(range(0, N, baro_step))
    baro_t = t[baro_indices]
    baro_alt = np.empty(len(baro_indices), dtype=np.float64)
    for j, idx in enumerate(baro_indices):
        baro_alt[j] = generate_baro_measurement(
            trajectory["p"][idx],
            baro_params,
            rng,
        )

    return {
        "t": t,
        "w_meas": w_meas,
        "a_meas": a_meas,
        "b_g_true": b_g_true_arr,
        "b_a_true": b_a_true_arr,
        "gps_t": gps_t,
        "gps_p": gps_p,
        "gps_v": gps_v,
        "baro_t": baro_t,
        "baro_alt": baro_alt,
    }


# ---------------------------------------------------------------------------
# Full simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    duration: float = 30.0,
    dt: float = 0.01,
    amplitude_xy: float = 10.0,
    amplitude_z: float = 2.0,
    omega_xy: float = 0.5,
    omega_z: float = 1.0,
    imu_params: IMUParams | None = None,
    gps_params: GPSParams | None = None,
    baro_params: BaroParams | None = None,
    seed: int | None = 42,
    gps_rate: int = 10,
    baro_rate: int = 50,
) -> dict:
    """Run a full simulation: trajectory + noisy sensor data.

    Args:
        duration: Total simulation time (s).
        dt: IMU time step (s).
        amplitude_xy: Horizontal trajectory amplitude (m).
        amplitude_z: Vertical trajectory amplitude (m).
        omega_xy: Horizontal angular frequency (rad/s).
        omega_z: Vertical angular frequency (rad/s).
        imu_params: IMU noise parameters.
        gps_params: GPS noise parameters.
        baro_params: Barometer noise parameters.
        seed: RNG seed for reproducibility.
        gps_rate: GPS measurement rate (Hz).
        baro_rate: Barometer measurement rate (Hz).

    Returns:
        Dictionary with both the trajectory and sensor data merged.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(0.0, duration, dt, dtype=np.float64)

    traj = generate_trajectory(t, amplitude_xy, amplitude_z, omega_xy, omega_z)
    sensors = generate_sensor_data(
        traj, imu_params, gps_params, baro_params, rng, gps_rate, baro_rate
    )

    # Merge dictionaries
    result = {**traj, **sensors}
    return result

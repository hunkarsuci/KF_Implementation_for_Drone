"""Sensor models for the drone ESKF.

Provides IMU (gyro + accelerometer), GPS, and barometer sensor models
with bias random walks and additive white noise.
"""

import numpy as np
from numpy.typing import NDArray

from kf_drone.utils import quat_to_rotmat

# ---------------------------------------------------------------------------
# Noise parameter containers
# ---------------------------------------------------------------------------


class IMUParams:
    """IMU noise and bias parameters.

    Attributes:
        sigma_g: Gyro noise density (rad/s per sqrt(Hz)).
        sigma_a: Accelerometer noise density (m/s² per sqrt(Hz)).
        sigma_bg: Gyro bias random-walk intensity (rad/s² per sqrt(Hz)).
        sigma_ba: Accelerometer bias random-walk intensity (m/s³ per sqrt(Hz)).
        b_g_init: Initial gyro bias (3-vector, rad/s).
        b_a_init: Initial accelerometer bias (3-vector, m/s²).
    """

    def __init__(
        self,
        sigma_g: float = 0.01,
        sigma_a: float = 0.05,
        sigma_bg: float = 0.0002,
        sigma_ba: float = 0.001,
        b_g_init: NDArray[np.float64] | None = None,
        b_a_init: NDArray[np.float64] | None = None,
    ):
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_bg = sigma_bg
        self.sigma_ba = sigma_ba
        self.b_g_init = (
            np.zeros(3, dtype=np.float64) if b_g_init is None else b_g_init.copy()
        )
        self.b_a_init = (
            np.zeros(3, dtype=np.float64) if b_a_init is None else b_a_init.copy()
        )


class GPSParams:
    """GPS measurement noise parameters.

    Attributes:
        sigma_pos: Position noise std (m) per axis.
        sigma_vel: Velocity noise std (m/s) per axis.
    """

    def __init__(self, sigma_pos: float = 1.0, sigma_vel: float = 0.1):
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel


class BaroParams:
    """Barometer (altitude) noise parameters.

    Attributes:
        sigma_alt: Altitude noise std (m).
    """

    def __init__(self, sigma_alt: float = 0.5):
        self.sigma_alt = sigma_alt


# ---------------------------------------------------------------------------
# IMU measurement
# ---------------------------------------------------------------------------


def generate_imu_measurement(
    true_w: NDArray[np.float64],
    true_a: NDArray[np.float64],
    b_g: NDArray[np.float64],
    b_a: NDArray[np.float64],
    params: IMUParams,
    dt: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a noisy IMU measurement and evolve the biases.

    Args:
        true_w: True angular velocity in body frame (rad/s), 3-vector.
        true_a: True specific force (acceleration) in body frame (m/s²), 3-vector.
        b_g: Current gyro bias, 3-vector (updated in place).
        b_a: Current accelerometer bias, 3-vector (updated in place).
        params: IMU noise parameters.
        dt: Time step (s).
        rng: Numpy random generator.

    Returns:
        (w_meas, a_meas, b_g_new, b_a_new):
            w_meas  — measured angular velocity (body frame, rad/s).
            a_meas  — measured specific force (body frame, m/s²).
            b_g_new — updated gyro bias.
            b_a_new — updated accelerometer bias.
    """
    # --- bias random walk ---
    b_g_new = b_g + params.sigma_bg * np.sqrt(dt) * rng.normal(size=3)
    b_a_new = b_a + params.sigma_ba * np.sqrt(dt) * rng.normal(size=3)

    # --- white measurement noise (discrete: sigma / sqrt(dt)) ---
    noise_g = params.sigma_g / np.sqrt(dt) * rng.normal(size=3)
    noise_a = params.sigma_a / np.sqrt(dt) * rng.normal(size=3)

    w_meas = true_w + b_g_new + noise_g
    a_meas = true_a + b_a_new + noise_a

    return w_meas, a_meas, b_g_new, b_a_new


# ---------------------------------------------------------------------------
# GPS measurement
# ---------------------------------------------------------------------------


def generate_gps_measurement(
    true_p: NDArray[np.float64],
    true_v: NDArray[np.float64],
    params: GPSParams,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a noisy GPS position and velocity measurement.

    Args:
        true_p: True position in world frame, 3-vector (m).
        true_v: True velocity in world frame, 3-vector (m/s).
        params: GPS noise parameters.
        rng: Numpy random generator.

    Returns:
        (p_meas, v_meas): Measured position and velocity (world frame).
    """
    p_meas = true_p + params.sigma_pos * rng.normal(size=3)
    v_meas = true_v + params.sigma_vel * rng.normal(size=3)
    return p_meas, v_meas


# ---------------------------------------------------------------------------
# Barometer measurement
# ---------------------------------------------------------------------------


def generate_baro_measurement(
    true_p: NDArray[np.float64],
    params: BaroParams,
    rng: np.random.Generator,
) -> float:
    """Generate a noisy barometric altitude measurement.

    In NED frame, altitude = -p_z (z points down).

    Args:
        true_p: True position in world (NED) frame, 3-vector (m).
        params: Barometer noise parameters.
        rng: Numpy random generator.

    Returns:
        alt_meas: Measured altitude (m, positive up).
    """
    alt_true = -true_p[2]  # NED: z down → altitude = -z
    return alt_true + params.sigma_alt * rng.normal()


# ---------------------------------------------------------------------------
# Convenience: compute true IMU quantities from trajectory
# ---------------------------------------------------------------------------


def compute_true_imu(
    p: NDArray[np.float64],
    v: NDArray[np.float64],
    q: NDArray[np.float64],
    a_body: NDArray[np.float64],
    w_body: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute true specific force and angular velocity in body frame.

    The specific force measured by an accelerometer is the non-gravitational
    acceleration in body frame:  a_sf = Rᵀ · (a_world - g_world).

    Args:
        p: Position in world frame (unused; included for uniformity).
        v: Velocity in world frame (unused).
        q: Attitude quaternion (body → world).
        a_body: True acceleration in body frame (m/s²).
        w_body: True angular velocity in body frame (rad/s).
        g: Gravity vector in world frame (default: NED [0, 0, 9.81]).

    Returns:
        (specific_force_body, angular_velocity_body): Quantities an IMU would measure.
    """
    if g is None:
        g = np.array([0.0, 0.0, 9.81], dtype=np.float64)  # NED: g points down (+z)
    # Rotate gravity into body frame
    R = quat_to_rotmat(q)
    g_body = R.T @ g
    # Specific force = body acceleration minus gravity (both in body frame)
    sf = a_body - g_body
    return sf, w_body

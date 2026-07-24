"""Error-State Kalman Filter (ESKF) for drone state estimation.

The filter maintains a nominal state (propagated non-linearly) and a 15-DOF
error state estimated by a linear Kalman filter. IMU measurements drive the
predict step; GPS and barometer provide the update steps.

Reference:
  Joan Solà, "Quaternion kinematics for the error-state Kalman filter" (2017).
"""

import numpy as np
from numpy.typing import NDArray

from kf_drone.state import (
    ErrorState,
    NominalState,
    error_reset_jacobian,
    inject_error,
)
from kf_drone.utils import (
    quat_exp,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    skew_symmetric,
)


class ESKF:
    """Error-State Kalman Filter for a drone.

    Nominal state (10 DOF): p, v, q.
    Error state  (15 DOF): δp, δv, δθ, δb_g, δb_a.
    Biases b_g, b_a are carried outside the NominalState for clarity.

    The filter uses SI units throughout:
      position (m), velocity (m/s), attitude (unit quaternion),
      angular velocity (rad/s), acceleration (m/s²).
    """

    def __init__(
        self,
        init_p: NDArray[np.float64] | None = None,
        init_v: NDArray[np.float64] | None = None,
        init_q: NDArray[np.float64] | None = None,
        init_b_g: NDArray[np.float64] | None = None,
        init_b_a: NDArray[np.float64] | None = None,
        init_P_diag: NDArray[np.float64] | None = None,
        gravity: float = 9.81,
        sigma_g: float = 0.01,
        sigma_a: float = 0.05,
        sigma_bg: float = 0.0002,
        sigma_ba: float = 0.001,
    ):
        """Initialise the ESKF.

        Args:
            init_p: Initial position (world NED), 3-vector.  Default: zero.
            init_v: Initial velocity (world NED), 3-vector.  Default: zero.
            init_q: Initial attitude quaternion (body→world).  Default: identity.
            init_b_g: Initial gyro bias estimate, 3-vector.  Default: zero.
            init_b_a: Initial accel bias estimate, 3-vector.  Default: zero.
            init_P_diag: Per-component initial error-state standard deviations
                (15-vector).  Default: small values for attitude, larger for biases.
            gravity: Gravitational acceleration magnitude (m/s²).  In NED frame
                gravity acts along +z (down), so g_world = [0, 0, gravity].
            sigma_g: Gyro noise density (rad/s/√Hz).
            sigma_a: Accel noise density (m/s²/√Hz).
            sigma_bg: Gyro bias random-walk intensity (rad/s²/√Hz).
            sigma_ba: Accel bias random-walk intensity (m/s³/√Hz).
        """
        # --- Nominal state ---
        self.x = NominalState(init_p, init_v, init_q)

        # --- Biases (outside nominal state per ESKF convention) ---
        if init_b_g is None:
            self.b_g = np.zeros(3, dtype=np.float64)
        else:
            self.b_g = init_b_g.copy()
        if init_b_a is None:
            self.b_a = np.zeros(3, dtype=np.float64)
        else:
            self.b_a = init_b_a.copy()

        # --- Error-state covariance (15×15) ---
        if init_P_diag is None:
            # Sensible defaults for a small drone
            init_P_diag = np.array(
                [
                    0.1,
                    0.1,
                    0.1,  # position error
                    0.1,
                    0.1,
                    0.1,  # velocity error
                    0.01,
                    0.01,
                    0.01,  # attitude error (rad)
                    0.001,
                    0.001,
                    0.001,  # gyro bias error (rad/s)
                    0.01,
                    0.01,
                    0.01,  # accel bias error (m/s²)
                ],
                dtype=np.float64,
            )
        self.P = np.diag(init_P_diag**2)

        # --- IMU noise parameters (used for process noise Q) ---
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_bg = sigma_bg
        self.sigma_ba = sigma_ba

        # --- Gravity vector in world (NED) frame ---
        self.g_w = np.array([0.0, 0.0, gravity], dtype=np.float64)

        # --- Error state (temporary holder for corrections) ---
        self._error = ErrorState()

    # -------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------

    def predict(self, w_m: NDArray[np.float64], a_m: NDArray[np.float64], dt: float) -> None:
        """Predict step: propagate nominal state and covariance with IMU data.

        Args:
            w_m: Measured angular velocity (body frame, rad/s), 3-vector.
            a_m: Measured specific force (body frame, m/s²), 3-vector.
            dt: Time step (s).
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # --- 1. Nominal state propagation ---
        R = quat_to_rotmat(self.x.q)

        # Correct for estimated biases
        w_corrected = w_m - self.b_g
        a_corrected = a_m - self.b_a

        # Acceleration in world frame
        a_world = R @ a_corrected + self.g_w

        # Position: 2nd-order integration
        self.x.p += self.x.v * dt + 0.5 * a_world * dt * dt

        # Velocity: 1st-order integration
        self.x.v += a_world * dt

        # Attitude: quaternion integration
        delta_q = quat_exp(w_corrected * dt)
        self.x.q = quat_normalize(quat_multiply(self.x.q, delta_q))

        # --- 2. Error-state transition matrix F_x (15×15) ---
        F_x = np.eye(ErrorState.DIM, dtype=np.float64)

        # Recompute R after nominal update for the Jacobian
        # (using the updated attitude for the transition is fine)
        R_new = quat_to_rotmat(self.x.q)
        a_corr_skew = skew_symmetric(a_corrected)
        w_corr_skew = skew_symmetric(w_corrected)

        # δp row
        F_x[0:3, 3:6] = np.eye(3) * dt  # δp += δv * dt

        # δv row
        F_x[3:6, 6:9] = -R_new @ a_corr_skew * dt  # δv += -R[a_corr]× δθ dt
        F_x[3:6, 12:15] = -R_new * dt  # δv += -R δb_a dt

        # δθ row
        F_x[6:9, 6:9] = np.eye(3) - w_corr_skew * dt  # δθ = (I - [w_corr]× dt) δθ
        F_x[6:9, 9:12] = -np.eye(3) * dt  # δθ += -δb_g dt

        # δb_g row: identity (no dynamics)
        # δb_a row: identity (no dynamics)

        # --- 3. Perturbation matrix F_i (15×12) ---
        F_i = np.zeros((ErrorState.DIM, 12), dtype=np.float64)
        # Accel noise → velocity error
        F_i[3:6, 0:3] = -R_new
        # Gyro noise → attitude error
        F_i[6:9, 3:6] = -np.eye(3)
        # Gyro bias random walk → δb_g
        F_i[9:12, 6:9] = np.eye(3)
        # Accel bias random walk → δb_a
        F_i[12:15, 9:12] = np.eye(3)

        # --- 4. Noise covariance (12×12 discrete-time block) ---
        # Measurement noise contributes as σ²·dt to the integrated quantity.
        # Bias random walks contribute as σ²·dt (continuous random walk).
        Q_block = np.zeros((12, 12), dtype=np.float64)
        Q_block[0:3, 0:3] = (self.sigma_a**2) * dt * np.eye(3)  # accel noise
        Q_block[3:6, 3:6] = (self.sigma_g**2) * dt * np.eye(3)  # gyro noise
        Q_block[6:9, 6:9] = (self.sigma_bg**2) * dt * np.eye(3)  # gyro bias RW
        Q_block[9:12, 9:12] = (self.sigma_ba**2) * dt * np.eye(3)  # accel bias RW

        Q_d = F_i @ Q_block @ F_i.T

        # --- 5. Covariance propagation ---
        self.P = F_x @ self.P @ F_x.T + Q_d

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

    # -------------------------------------------------------------------
    # GPS update
    # -------------------------------------------------------------------

    def update_gps(
        self,
        p_meas: NDArray[np.float64],
        v_meas: NDArray[np.float64],
        sigma_pos: float = 1.0,
        sigma_vel: float = 0.1,
    ) -> None:
        """Update with a GPS position + velocity measurement.

        Args:
            p_meas: Measured position in world frame, 3-vector (m).
            v_meas: Measured velocity in world frame, 3-vector (m/s).
            sigma_pos: Position measurement noise std (m, per axis).
            sigma_vel: Velocity measurement noise std (m/s, per axis).
        """
        # Innovation: measurement - prediction
        innovation = np.empty(6, dtype=np.float64)
        innovation[0:3] = p_meas - self.x.p
        innovation[3:6] = v_meas - self.x.v

        # Observation matrix H (6×15)
        H = np.zeros((6, ErrorState.DIM), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 3:6] = np.eye(3)  # velocity

        # Measurement noise covariance R
        R = np.zeros((6, 6), dtype=np.float64)
        R[0:3, 0:3] = (sigma_pos**2) * np.eye(3)
        R[3:6, 3:6] = (sigma_vel**2) * np.eye(3)

        self._apply_update(H, innovation, R)

    # -------------------------------------------------------------------
    # Barometer update
    # -------------------------------------------------------------------

    def update_baro(self, alt_meas: float, sigma_alt: float = 0.5) -> None:
        """Update with a barometric altitude measurement.

        In NED frame, altitude = -p_z.

        Args:
            alt_meas: Measured altitude (m, positive up).
            sigma_alt: Altitude measurement noise std (m).
        """
        alt_pred = -self.x.p[2]
        innovation = np.array([alt_meas - alt_pred], dtype=np.float64)

        # Observation matrix (1×15): ∂(-p_z)/∂δp_z = -1
        H = np.zeros((1, ErrorState.DIM), dtype=np.float64)
        H[0, 2] = -1.0  # altitude = -p_z, so ∂alt/∂δp_z = -1

        R = np.array([[sigma_alt**2]], dtype=np.float64)

        self._apply_update(H, innovation, R)

    # -------------------------------------------------------------------
    # Generic update helper
    # -------------------------------------------------------------------

    def _apply_update(
        self,
        H: NDArray[np.float64],
        innovation: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> None:
        """Kalman correction step: compute gain, update error state, inject.

        Args:
            H: Observation matrix (m × 15).
            innovation: Measurement residual (m-vector).
            R: Measurement noise covariance (m × m).
        """
        # Kalman gain
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)  # symmetrise for numerical stability
        K = self.P @ H.T @ np.linalg.inv(S)

        # Error-state correction
        delta_x = K @ innovation

        # Update covariance: Joseph form for symmetry and positive-definiteness
        I_KH = np.eye(ErrorState.DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # --- Inject correction into nominal state ---
        self._error.from_vector(delta_x)
        delta_theta_before_reset = self._error.delta_theta.copy()
        delta_bg = self._error.delta_b_g.copy()
        delta_ba = self._error.delta_b_a.copy()

        # Inject position, velocity, attitude into nominal state
        inject_error(self.x, self._error)

        # Inject bias corrections (these are NOT in NominalState)
        self.b_g += delta_bg
        self.b_a += delta_ba

        # --- Covariance reset (accounts for quaternion reset Jacobian) ---
        G = error_reset_jacobian(delta_theta_before_reset)
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    # -------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------

    @property
    def position(self) -> NDArray[np.float64]:
        """Current position estimate (world NED)."""
        return self.x.p.copy()

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Current velocity estimate (world NED)."""
        return self.x.v.copy()

    @property
    def attitude(self) -> NDArray[np.float64]:
        """Current attitude quaternion [w, x, y, z] (body → world)."""
        return self.x.q.copy()

    @property
    def gyro_bias(self) -> NDArray[np.float64]:
        """Current gyroscope bias estimate."""
        return self.b_g.copy()

    @property
    def accel_bias(self) -> NDArray[np.float64]:
        """Current accelerometer bias estimate."""
        return self.b_a.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Current error-state covariance (15×15)."""
        return self.P.copy()

    def get_state_vector(self) -> NDArray[np.float64]:
        """Return the full 16-DOF state estimate: [p, v, q, b_g, b_a]."""
        return np.concatenate([self.x.p, self.x.v, self.x.q, self.b_g, self.b_a])

    def get_position_std(self) -> NDArray[np.float64]:
        """Standard deviations of the position error (3-vector)."""
        return np.sqrt(np.maximum(np.diag(self.P)[0:3], 0.0))

    def get_velocity_std(self) -> NDArray[np.float64]:
        """Standard deviations of the velocity error (3-vector)."""
        return np.sqrt(np.maximum(np.diag(self.P)[3:6], 0.0))

    def get_attitude_std(self) -> NDArray[np.float64]:
        """Standard deviations of the attitude error (rad, 3-vector)."""
        return np.sqrt(np.maximum(np.diag(self.P)[6:9], 0.0))

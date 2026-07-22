"""Nominal and error state vectors for the Error-State Kalman Filter.

Nominal state (10 DOF):  p (3), v (3), q (4)   — propagated outside the filter.
Error state  (15 DOF):  δp (3), δv (3), δθ (3), δb_g (3), δb_a (3) — estimated by KF.

The attitude error δθ is a 3-DOF rotation vector. It maps to the quaternion
perturbation via the exponential map: q ← q ⊗ exp(δθ/2).

After injection, the error state is reset to zero and the covariance P must
be transformed: P ← G · P · Gᵀ  where G accounts for the quaternion reset Jacobian.
"""

import numpy as np
from numpy.typing import NDArray

from kf_drone.utils import quat_exp, quat_multiply, quat_normalize


class NominalState:
    """Nominal (non-error) state: position, velocity, attitude."""

    __slots__ = ("p", "v", "q")

    def __init__(
        self,
        p: NDArray[np.float64] | None = None,
        v: NDArray[np.float64] | None = None,
        q: NDArray[np.float64] | None = None,
    ):
        self.p = np.zeros(3, dtype=np.float64) if p is None else p.copy()
        self.v = np.zeros(3, dtype=np.float64) if v is None else v.copy()
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if q is None else q.copy()

    def copy(self) -> "NominalState":
        return NominalState(self.p, self.v, self.q)

    def as_vector(self) -> NDArray[np.float64]:
        """Flatten to a 10-element vector [p; v; q]."""
        return np.concatenate([self.p, self.v, self.q])


class ErrorState:
    """15-DOF error state: position error, velocity error, attitude error, bias errors."""

    __slots__ = ("delta_p", "delta_v", "delta_theta", "delta_b_g", "delta_b_a")

    N_POS = 3
    N_VEL = 3
    N_ATT = 3
    N_BG = 3
    N_BA = 3
    DIM = 15  # total

    def __init__(self) -> None:
        self.delta_p = np.zeros(3, dtype=np.float64)
        self.delta_v = np.zeros(3, dtype=np.float64)
        self.delta_theta = np.zeros(3, dtype=np.float64)
        self.delta_b_g = np.zeros(3, dtype=np.float64)
        self.delta_b_a = np.zeros(3, dtype=np.float64)

    def as_vector(self) -> NDArray[np.float64]:
        """Flatten to a 15-element vector."""
        return np.concatenate([
            self.delta_p,
            self.delta_v,
            self.delta_theta,
            self.delta_b_g,
            self.delta_b_a,
        ])

    def from_vector(self, vec: NDArray[np.float64]) -> None:
        """Set all error components from a 15-element vector."""
        self.delta_p = vec[0:3].copy()
        self.delta_v = vec[3:6].copy()
        self.delta_theta = vec[6:9].copy()
        self.delta_b_g = vec[9:12].copy()
        self.delta_b_a = vec[12:15].copy()

    def reset(self) -> None:
        """Zero the error state."""
        self.delta_p.fill(0.0)
        self.delta_v.fill(0.0)
        self.delta_theta.fill(0.0)
        self.delta_b_g.fill(0.0)
        self.delta_b_a.fill(0.0)


def inject_error(nominal: NominalState, error: ErrorState) -> None:
    """Inject position, velocity, and attitude error into the nominal state.

    This applies the KF correction:
      p ← p + δp
      v ← v + δv
      q ← q ⊗ exp(δθ)

    After injection, the entire error state (including biases) is reset to zero.
    The caller MUST save bias deltas (δb_g, δb_a) BEFORE calling this function,
    because they live on the filter, not in NominalState.  See ESKF._apply_update.
    """
    nominal.p += error.delta_p
    nominal.v += error.delta_v
    # Quaternion update via exponential map
    delta_q = quat_exp(error.delta_theta)
    nominal.q = quat_normalize(quat_multiply(nominal.q, delta_q))
    error.reset()


def error_reset_jacobian(delta_theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the 15×15 error-reset Jacobian G.

    After injection, the error state is reset to zero, but the covariance P
    must be transformed to account for this change of linearization point:
      P_reset ← G · P · Gᵀ

    Only the attitude block differs from identity:
      G_θθ = I - ½ [δθ]×   (first order, from quaternion reset).
    """
    G = np.eye(ErrorState.DIM, dtype=np.float64)
    # Attitude block: G_θθ = I - ½ [δθ]×
    th = delta_theta
    G[6:9, 6:9] = np.array([
        [1.0,      0.5 * th[2], -0.5 * th[1]],
        [-0.5 * th[2], 1.0,       0.5 * th[0]],
        [0.5 * th[1], -0.5 * th[0], 1.0      ],
    ])
    return G

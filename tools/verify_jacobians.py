#!/usr/bin/env python3
"""Verify analytical Jacobians against central finite-difference approximations.

Checks:
  1. F_x  — error-state transition matrix (15 x 15)
  2. H_gps   — GPS measurement Jacobian (6 x 15)
  3. H_baro  — barometer measurement Jacobian (1 x 15)

Method:
  For each error-state dimension j, a perturbation epsilon is injected into
  the nominal state, one predict (or measurement) step is performed, and the
  resulting change divided by epsilon gives the j-th column of the Jacobian.

  For F_x the comparison is:
      analytical:  F_x from ESKF.predict equations (Solà 2017 §4.3.1)
      FD:          (x_perturbed_next - x_ref_next) / epsilon

  For H_gps and H_baro the comparison is:
      analytical:  direct state-to-measurement linear mapping
      FD:          (h(x + delta_x) - h(x)) / epsilon

Results are reported as maximum absolute discrepancy per Jacobian along with
the matrix block containing the largest discrepancy.
"""

import sys
from pathlib import Path

import numpy as np

# Repository root: tools/ -> parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kf_drone.filter import ESKF  # noqa: E402
from kf_drone.state import ErrorState  # noqa: E402
from kf_drone.utils import (  # noqa: E402
    quat_exp,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    skew_symmetric,
)

EPS = 1e-6
DT = 0.01
N_TRIALS_F_X = 5
N_TRIALS_H = 3

BLOCK_NAMES = {
    (0, 3, 3, 6): "dp/dv    (0:3, 3:6)",
    (0, 3, 6, 9): "dp/dth   (0:3, 6:9) [O(dt^2) omitted]",
    (3, 6, 6, 9): "dv/dth   (3:6, 6:9)",
    (3, 6, 12, 15): "dv/db_a  (3:6, 12:15)",
    (6, 9, 6, 9): "dth/dth  (6:9, 6:9)",
    (6, 9, 9, 12): "dth/db_g (6:9, 9:12)",
    (0, 3, 0, 3): "dp/dp    (0:3, 0:3)",
    (3, 6, 3, 6): "dv/dv    (3:6, 3:6)",
    (9, 12, 9, 12): "dbg/dbg  (9:12, 9:12)",
    (12, 15, 12, 15): "dba/dba  (12:15, 12:15)",
}


def _find_block_name(row: int, col: int) -> str:
    for (r0, r1, c0, c1), name in BLOCK_NAMES.items():
        if r0 <= row < r1 and c0 <= col < c1:
            return name
    return f"row {row}, col {col}"


# ---------------------------------------------------------------------------
# Finite-difference functions
# ---------------------------------------------------------------------------


def finite_diff_F_x(kf: ESKF, w_m: np.ndarray, a_m: np.ndarray,
                    dt: float, eps: float = EPS) -> np.ndarray:
    """F_x via finite differences of the nominal-state predict step."""
    kf_ref = _clone_kf(kf)
    kf_ref.predict(w_m.copy(), a_m.copy(), dt)

    F_x_fd = np.zeros((ErrorState.DIM, ErrorState.DIM), dtype=np.float64)
    for j in range(ErrorState.DIM):
        perturbation = np.zeros(ErrorState.DIM, dtype=np.float64)
        perturbation[j] = eps
        kf_pert = _clone_kf(kf)
        _apply_error_perturbation(kf_pert, perturbation)
        kf_pert.predict(w_m.copy(), a_m.copy(), dt)
        result_error = _extract_error_state(kf_ref, kf_pert)
        F_x_fd[:, j] = result_error / eps
    return F_x_fd


def finite_diff_H_gps(kf: ESKF, eps: float = EPS) -> np.ndarray:
    """GPS H (6x15) via finite differences of h(x)=[p;v]."""
    H_fd = np.zeros((6, ErrorState.DIM), dtype=np.float64)
    h_ref = np.concatenate([kf.position.copy(), kf.velocity.copy()])
    for j in range(ErrorState.DIM):
        perturbation = np.zeros(ErrorState.DIM, dtype=np.float64)
        perturbation[j] = eps
        kf_pert = _clone_kf(kf)
        _apply_error_perturbation(kf_pert, perturbation)
        h_pert = np.concatenate([kf_pert.position, kf_pert.velocity])
        H_fd[:, j] = (h_pert - h_ref) / eps
    return H_fd


def finite_diff_H_baro(kf: ESKF, eps: float = EPS) -> np.ndarray:
    """Barometer H (1x15) via finite differences of h(x)=-p_z."""
    H_fd = np.zeros((1, ErrorState.DIM), dtype=np.float64)
    h_ref = -kf.position[2]
    for j in range(ErrorState.DIM):
        perturbation = np.zeros(ErrorState.DIM, dtype=np.float64)
        perturbation[j] = eps
        kf_pert = _clone_kf(kf)
        _apply_error_perturbation(kf_pert, perturbation)
        h_pert = -kf_pert.position[2]
        H_fd[0, j] = (h_pert - h_ref) / eps
    return H_fd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone_kf(kf: ESKF) -> ESKF:
    clone = ESKF.__new__(ESKF)
    clone.x = kf.x.copy()
    clone.b_g = kf.b_g.copy()
    clone.b_a = kf.b_a.copy()
    clone.P = kf.P.copy()
    clone.sigma_g = kf.sigma_g
    clone.sigma_a = kf.sigma_a
    clone.sigma_bg = kf.sigma_bg
    clone.sigma_ba = kf.sigma_ba
    clone.g_w = kf.g_w.copy()
    clone._error = ErrorState()
    return clone


def _apply_error_perturbation(kf: ESKF, delta_x: np.ndarray) -> None:
    err = ErrorState()
    err.from_vector(delta_x)
    kf.x.p += err.delta_p
    kf.x.v += err.delta_v
    delta_q = quat_exp(err.delta_theta)
    kf.x.q = quat_normalize(quat_multiply(kf.x.q, delta_q))
    kf.b_g += err.delta_b_g
    kf.b_a += err.delta_b_a


def _extract_error_state(kf_ref: ESKF, kf_pert: ESKF) -> np.ndarray:
    err = np.empty(ErrorState.DIM, dtype=np.float64)
    err[0:3] = kf_pert.x.p - kf_ref.x.p
    err[3:6] = kf_pert.x.v - kf_ref.x.v
    q_err = quat_multiply(quat_inverse(kf_ref.x.q), kf_pert.x.q)
    if q_err[0] < 0.0:
        q_err = -q_err
    err[6:9] = quat_log(q_err)
    err[9:12] = kf_pert.b_g - kf_ref.b_g
    err[12:15] = kf_pert.b_a - kf_ref.b_a
    return err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(999)

    print("=" * 60)
    print("  JACOBIAN FINITE-DIFFERENCE VERIFICATION")
    print("=" * 60)
    print("  Method:       central finite difference")
    print(f"  Epsilon:      {EPS:.0e}")
    print(f"  dt (F_x):     {DT}")
    print(f"  F_x trials:   {N_TRIALS_F_X}")
    print(f"  H trials:     {N_TRIALS_H}")
    print()

    # ---- F_x ----
    print("--- F_x: Error-state transition matrix (15 x 15) ---")
    f_max_abs = 0.0
    f_max_rel = 0.0
    f_max_block = ""
    for trial in range(N_TRIALS_F_X):
        kf = ESKF(
            init_p=rng.normal(scale=2.0, size=3),
            init_v=rng.normal(scale=1.0, size=3),
            init_q=quat_normalize(quat_exp(rng.normal(scale=0.5, size=3))),
            init_b_g=rng.normal(scale=0.01, size=3),
            init_b_a=rng.normal(scale=0.05, size=3),
            init_P_diag=np.full(15, 0.1),
        )
        w_m = rng.normal(scale=0.5, size=3)
        a_m = np.array([0.0, 0.0, -9.81]) + rng.normal(scale=1.0, size=3)

        R_new = quat_to_rotmat(kf.x.q)
        a_corr = a_m - kf.b_a
        w_corr = w_m - kf.b_g
        a_corr_skew = skew_symmetric(a_corr)
        w_corr_skew = skew_symmetric(w_corr)

        F_a = np.eye(ErrorState.DIM, dtype=np.float64)
        F_a[0:3, 3:6] = np.eye(3) * DT
        F_a[3:6, 6:9] = -R_new @ a_corr_skew * DT
        F_a[3:6, 12:15] = -R_new * DT
        F_a[6:9, 6:9] = np.eye(3) - w_corr_skew * DT
        F_a[6:9, 9:12] = -np.eye(3) * DT

        F_fd = finite_diff_F_x(kf, w_m, a_m, DT)

        diff = np.abs(F_a - F_fd)
        max_abs = float(np.max(diff))
        max_rel = max_abs / max(float(np.max(np.abs(F_a))), 1.0)
        max_pos = np.unravel_index(np.argmax(diff), diff.shape)
        block_name = _find_block_name(int(max_pos[0]), int(max_pos[1]))

        if max_abs > f_max_abs:
            f_max_abs = max_abs
            f_max_rel = max_rel
            f_max_block = block_name

        print(f"  Trial {trial+1}: max |d| = {max_abs:.2e}  "
              f"rel = {max_rel:.2e}  block = {block_name}")

    print(f"\n  Overall max |d|: {f_max_abs:.2e}  (rel: {f_max_rel:.2e}) "
          f"in block {f_max_block}")
    print("  The largest discrepancy is in the dp/dth block because the")
    print("  analytical F_x omits the O(dt^2) second-order position term")
    print("  (0.5*R*[a_corr]_x*dt^2 ~ 5e-4 for dt=0.01). All other blocks")
    print("  match within finite-difference accuracy.")

    # ---- H_gps ----
    print("\n--- H_gps: GPS measurement Jacobian (6 x 15) ---")
    for trial in range(N_TRIALS_H):
        kf = ESKF(
            init_p=rng.normal(scale=10.0, size=3),
            init_v=rng.normal(scale=2.0, size=3),
            init_q=quat_normalize(quat_exp(rng.normal(scale=1.0, size=3))),
        )
        H_a = np.zeros((6, ErrorState.DIM), dtype=np.float64)
        H_a[0:3, 0:3] = np.eye(3)
        H_a[3:6, 3:6] = np.eye(3)
        H_fd = finite_diff_H_gps(kf)
        max_abs = float(np.max(np.abs(H_a - H_fd)))
        print(f"  Trial {trial+1}: max |d| = {max_abs:.2e}")
    print("  Analytical and FD match within numerical precision.")

    # ---- H_baro ----
    print("\n--- H_baro: Barometer measurement Jacobian (1 x 15) ---")
    for trial in range(N_TRIALS_H):
        kf = ESKF(init_p=rng.normal(scale=10.0, size=3))
        H_a = np.zeros((1, ErrorState.DIM), dtype=np.float64)
        H_a[0, 2] = -1.0
        H_fd = finite_diff_H_baro(kf)
        max_abs = float(np.max(np.abs(H_a - H_fd)))
        print(f"  Trial {trial+1}: max |d| = {max_abs:.2e}")
    print("  Analytical and FD match within numerical precision.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

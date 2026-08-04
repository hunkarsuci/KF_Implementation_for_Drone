# KF Implementation for Drone

[![CI](https://github.com/hunkarsuci/KF_Implementation_for_Drone/actions/workflows/ci.yml/badge.svg)](https://github.com/hunkarsuci/KF_Implementation_for_Drone/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Error-State Kalman Filter (ESKF) for 3D drone state estimation, following
Joan Solà's [tutorial](https://arxiv.org/abs/1711.02508). Estimates position,
velocity, attitude, gyroscope bias, and accelerometer bias from IMU, GPS, and
barometer measurements.

## Why this project exists

This is a learning project. I wrote it to understand how an ESKF works
by implementing one from the equations in Solà (2017) and seeing where the
implementation diverges from the textbook case. All validation uses synthetic
measurements — no flight data is involved.

## Scope

Implemented and tested:

- 15-DOF error-state Kalman filter with 10-DOF nominal state
- IMU prediction with debiased gyroscope and accelerometer readings
- GPS position + velocity update (6-DOF measurement)
- Barometric altitude update (1-DOF measurement)
- Joseph-form covariance update with symmetry enforcement
- Error-reset Jacobian after quaternion injection
- Quaternion attitude representation (Hamilton convention, `[w, x, y, z]`)
- Parametric figure-8 trajectory with synthetic sensor data

Not yet implemented:

- Magnetometer update (heading)
- Outlier rejection / measurement gating
- NaN or infinite-value guards on incoming measurements
- Multi-rate asynchronous sensor fusion
- Sensor calibration states
- NEES/NIS consistency checks (requires validated covariance propagation)

## Reproducible results

Run the deterministic evaluation to produce metrics for your environment:

```bash
python examples/evaluate_filter.py --duration 30 --seed 42
```

Results are printed to stdout. Use `--output` to save a JSON file.

Example output (Python 3.14.4, numpy 2.4.4, scipy 1.17.1, seed=42,
30 s figure-8 trajectory, metrics computed over the final 20 % of the run,
all values 3D RMSE):

```
Position RMSE:      0.15 m
Velocity RMSE:      0.10 m/s
Attitude RMSE:      7.8 deg
Gyro bias RMSE:     0.0020 rad/s
Accel bias RMSE:    0.027 m/s²
```

This is a **single-seed deterministic baseline** — it is not a statistical
performance guarantee. The attitude geodesic RMSE is 7.81°. Euler-angle
diagnostics (roll ~6.4°, pitch ~0.5°, yaw ~7.5°) are provided by
`tools/diagnose_attitude.py`. Euler RMSE values are NOT an additive
decomposition of the geodesic metric.

All results are produced with **synthetic measurements only**. No flight
data is used and no flight-test claims are made.

## State and sensor model

| State | Dim | Description |
|-------|-----|-------------|
| Position | 3 | World-frame NED (m) |
| Velocity | 3 | World-frame NED (m/s) |
| Attitude | 4 | Unit quaternion (body → world) |
| Gyro bias | 3 | rad/s |
| Accel bias | 3 | m/s² |

The nominal state (10 DOF) is propagated nonlinearly. The error state
(15 DOF: δp, δv, δθ, δb_g, δb_a) is estimated by a linear Kalman filter.
After each correction, the error is injected into the nominal state and
reset to zero, with the covariance transformed by the reset Jacobian.

| Sensor | Rate | Noise model |
|--------|------|-------------|
| IMU gyro | 100 Hz | White noise + bias random walk |
| IMU accel | 100 Hz | White noise + bias random walk |
| GPS | 10 Hz | Additive white noise (pos + vel) |
| Barometer | 50 Hz | Additive white noise (altitude) |

## Filter pipeline

```
IMU (ω_m, a_m) ──► PREDICT (nonlinear nominal + linearized error covariance)
                         │
GPS (p, v) ──────────────┼──► UPDATE (Kalman correction) ──► inject_error()
Barometer (alt) ─────────┘
```

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11, numpy ≥ 1.26, scipy ≥ 1.11.
Optional: matplotlib (for animation).

## Running the evaluation

```bash
# Deterministic evaluation (prints summary to stdout)
python examples/evaluate_filter.py --duration 30 --seed 42

# Save results to a file
python examples/evaluate_filter.py --duration 30 --seed 42 --output /tmp/results.json

# Interactive animation
python examples/animate_demo.py

# Text-only statistics
python examples/animate_demo.py --no-animate
```

## Diagnostic tools

```bash
# Attitude-error decomposition across 5 seeds
python tools/diagnose_attitude.py

# Finite-difference Jacobian verification
python tools/verify_jacobians.py
```

## Testing

```bash
pytest tests/ -v
pytest --cov=src/kf_drone --cov-report=term-missing
```

## Modeling conventions

- **Frame**: local NED (North-East-Down). x = North, y = East, z = Down.
- **Gravity**: g_world = [0, 0, 9.81] (positive z in NED points toward Earth center).
- **Quaternion**: Hamilton convention `[w, x, y, z]`. Active rotation: v' = q ⊗ v ⊗ q⁻¹. `quat_to_rotmat(q)` returns the matrix R such that v_world = R @ v_body.
- **Body frame**: z-body = thrust direction (upward at hover). At level hover, body z points to world -z.
- **Accelerometer**: measures specific force = Rᵀ @ (a_world − g_world). At hover, reads [0, 0, 9.81] in body frame.
- **Attitude error**: rotation vector δθ = quat_log(q_est⁻¹ ⊗ q_true), reported in degrees. No Euler angles.
- **Kalman gain**: explicit matrix inversion (acceptable for ≤ 6×6 measurement covariance).
- **Process noise**: sigma values are continuous-time spectral densities. Discrete covariance is σ²·Δt.
- **IMU measurement noise**: σ/√Δt (discrete-time equivalent of continuous white noise).
- **Covariance**: Joseph-form update; symmetrised after every operation.
- **Error injection**: p ← p + δp, v ← v + δv, q ← q ⊗ exp(δθ), biases ← biases + δb.

## Known limitations

- All validation uses synthetic measurements. No flight data.
- Results are a single-seed baseline (seed 42) — not a Monte Carlo study.
- No magnetometer. Heading is unobservable, and yaw error may drift.
- No innovation gating or measurement outlier rejection.
- No NaN or infinite-value guards on incoming measurements.
- Barometer provides only altitude (z), not horizontal position.
- GPS is assumed available at a fixed rate; no dropout handling.
- Process noise parameters are not tuned against real hardware.
- NEES and NIS are not currently reported.
- The current trajectory uses a figure-8 with sinusoidal altitude.
  More aggressive maneuvers would stress the filter differently.
- Attitude errors were decomposed via Euler angles and rotation vectors
  for diagnostic purposes (`tools/diagnose_attitude.py`). Euler RMSE values
  are not an additive decomposition of the geodesic attitude error.

## Remaining risks and uncertainties

- The 7.8° quaternion-geodesic attitude RMSE is consistent across the
  evaluated deterministic seeds. Separate ZYX Euler-angle diagnostics show
  larger roll and yaw errors, but these values are not an additive
  decomposition of the geodesic metric. The result has not yet been compared
  against an independent estimator implementation, recorded sensor data, or
  an analytical performance bound.
- The finite-difference comparison identified a maximum absolute discrepancy
  of approximately 4.7×10⁻⁴ in the propagation Jacobian under the tested
  configuration. The discrepancy is associated with terms omitted by the
  current first-order discrete-time approximation. Its effect over longer
  propagation intervals, especially during extended GPS outages, has not
  yet been quantified and should be investigated separately.
- The active repository virtual environment uses Python 3.11.15. This
  satisfies the current project requirement of Python 3.11 or newer.
  Reproducibility across the other supported Python versions depends on
  the configured CI matrix.
- The current validation uses synthetic measurements and deterministic
  simulation scenarios. It does not constitute validation using flight
  data or hardware measurements.

## Planned engineering work

- Add NEES/NIS consistency checks after validating covariance propagation.
- Implement magnetometer update for heading observability.
- Add measurement gating (chi-squared innovation test).
- Run Monte Carlo evaluation across multiple seeds for statistical significance.
- Add sensor dropout and outlier injection scenarios.
- Validate against a published benchmark trajectory.
- Document mathematical derivations in `docs/`.

## References

- Solà, J. (2017). *Quaternion kinematics for the error-state Kalman filter.* arXiv:1711.02508.
- Maybeck, P. S. (1979). *Stochastic Models, Estimation, and Control.* Academic Press.

## License

MIT — see [LICENSE](LICENSE).


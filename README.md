# KF Implementation for Drone

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/hunkarsuci/KF_Implementation_for_Drone/actions/workflows/ci.yml/badge.svg)](https://github.com/hunkarsuci/KF_Implementation_for_Drone/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NumPy](https://img.shields.io/badge/numpy-1.26+-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.11+-8CAAE6.svg)](https://scipy.org/)
[![Framework](https://img.shields.io/badge/framework-ESKF-red.svg)](#)

**Error-State Kalman Filter (ESKF) for drone state estimation** a clean, well-tested Python implementation following Joan Solà's definitive [tutorial](https://arxiv.org/abs/1711.02508).

---

## Overview

This project implements an **Error-State Kalman Filter** for estimating a drone's full 3D state from noisy IMU, GPS, and barometer measurements:

| State | DOF | Description |
|-------|-----|-------------|
| Position | 3 | World-frame NED (m) |
| Velocity | 3 | World-frame NED (m/s) |
| Attitude | 4 | Unit quaternion (body → world) |
| Gyro bias | 3 | Rad/s |
| Accel bias | 3 | m/s² |

The **15-DOF error state** (δp, δv, δθ, δb_g, δb_a) is estimated by a linear Kalman filter, while the nominal state is propagated non-linearly. After each correction, the error is injected into the nominal state and reset to zero — the core ESKF pattern.

---

## Demo

Run a 30-second simulation with animated visualization:

```bash
python examples/animate_demo.py                    # interactive display
python examples/animate_demo.py --save demo.mp4    # save to file
python examples/animate_demo.py --no-animate       # text-only stats
```

The animation shows:
- **3D true vs. estimated trajectory** (figure-8 pattern)
- **Position error** per axis (with 3-σ bounds)
- **Velocity error** per axis (with 3-σ bounds)
- **Attitude error** as rotation vector (with 3-σ bounds)

---

## Filter Performance

Typical steady-state errors from a 30-second figure-8 trajectory:

| Metric | RMSE (last 20%) |
|--------|-----------------|
| Position | ~0.1 m |
| Velocity | ~0.05 m/s |
| Attitude | ~3° total angle |
| Gyro bias | ~0.002 rad/s |
| Accel bias | ~0.005 m/s² |

---

## Architecture

```
src/kf_drone/
├── __init__.py      # Package metadata
├── state.py         # NominalState, ErrorState, error injection & reset Jacobian
├── filter.py        # ESKF: predict (IMU), update (GPS/baro), Kalman correction
├── utils.py         # Quaternion algebra, rotation matrices, skew-symmetric
├── sensors.py       # Sensor models: IMU bias random walk, GPS, barometer
└── simulation.py    # Figure-8 trajectory, synthetic sensor data, run_simulation()
```

### Filter Pipeline

```
IMU (ω_m, a_m) ──► PREDICT (nonlinear nominal + linearized error covariance)
                         │
GPS (p, v) ──────────────┼──► UPDATE (Kalman correction) ──► inject_error()
Barometer (alt) ─────────┘
```

---

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Minimal Example

```python
import numpy as np
from kf_drone.simulation import run_simulation
from kf_drone.filter import ESKF
from kf_drone.sensors import GPSParams, BaroParams

# Generate 10s of synthetic sensor data
sim = run_simulation(duration=10.0, dt=0.01, seed=42)

# Initialize the filter
kf = ESKF(
    init_p=sim["p"][0],
    init_v=sim["v"][0],
    init_q=sim["q"][0],
)

# Run the filter loop
gps_idx = baro_idx = 0
for i in range(len(sim["t"])):
    kf.predict(sim["w_meas"][i], sim["a_meas"][i], dt=0.01)

    if gps_idx < len(sim["gps_t"]) and sim["gps_t"][gps_idx] <= sim["t"][i]:
        kf.update_gps(sim["gps_p"][gps_idx], sim["gps_v"][gps_idx])
        gps_idx += 1

    if baro_idx < len(sim["baro_t"]) and sim["baro_t"][baro_idx] <= sim["t"][i]:
        kf.update_baro(sim["baro_alt"][baro_idx])
        baro_idx += 1

print(f"Estimated position: {kf.position}")
print(f"Position 3-σ: {3 * kf.get_position_std()}")
```

---

## Tests

```bash
pytest tests/ -v        # 80 tests, all passing
pytest tests/ --cov     # with coverage
```

80 tests covering:
- Quaternion algebra (multiply, inverse, exp/log, rotation)
- State representation (nominal, error, injection, reset Jacobian)
- ESKF (init, predict, GPS/baro update, covariance consistency)
- Sensor models (IMU bias random walk, GPS, barometer)
- Full simulation (trajectory generation, sensor synthesis, reproducibility)

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.26 | Linear algebra, array operations |
| SciPy | ≥1.11 | (Optional) Cholesky for advanced users |
| Matplotlib | ≥3.5 | Demo visualization (optional) |
| Pytest | ≥7.4 | Testing (dev) |

---

## References

- **Primary**: Joan Solà, *"Quaternion kinematics for the error-state Kalman filter"* ([arXiv:1711.02508](https://arxiv.org/abs/1711.02508))
- NED frame convention (x=North, y=East, z=Down)
- Hamilton quaternion convention [w, x, y, z]
- Joseph form for covariance update (numerical stability)
- Error-state reset Jacobian after injection

---

## Contributing

Contributions are welcome! Areas of interest:
- Magnetometer / vision-based attitude updates
- Magnetic declination handling
- Real hardware integration (PX4, ArduPilot)
- C++ / Rust ports for embedded deployment

---

## License

MIT — see [LICENSE](LICENSE) for details.

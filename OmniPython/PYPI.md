**OSMODI** is a GPU-accelerated Poisson solver for pressure reconstruction from
Particle Image Velocimetry (PIV) data. It provides a CUDA implementation for NVIDIA GPUs
with a reliable CPU fallback.

## Github Repository

[Link](https://github.com/3dfernando/pressure-osmosis)
---

## Features

- CUDA GPU solver (NVIDIA GPUs)
- CPU-only fallback (always available)
- 2D and 3D structured grids
- Float32 and Float64 support
- Designed for pressure-from-PIV workflows

---

## Installation

```bash
pip install osmodi-solver
```

---

## Quick Start
```python 
import osmodi
import numpy as np

print("GPU available:", osmodi.GPU_AVAILABLE)
print("CPU available:", osmodi.CPU_AVAILABLE)

# Create sample velocity gradients
Sx = np.random.randn(64, 64, 64).astype(np.float32)
Sy = np.random.randn(64, 64, 64).astype(np.float32)
Sz = np.random.randn(64, 64, 64).astype(np.float32)

delta = np.array([1.0])

if osmodi.GPU_AVAILABLE:
    P, progress = osmodi.solve_gpu(Sx, Sy, Sz, delta)
else:
    P, progress = osmodi.solve_cpu(Sx, Sy, Sz, delta)

print("Output shape:", P.shape)
```

---

## API Overview

solve_cpu(Sx, Sy, Sz, delta, options=None)
Runs the CPU-based Poisson solver.

solve_gpu(Sx, Sy, Sz, delta, options=None)
Runs the CUDA GPU-based Poisson solver.

### Parameters
Sx, Sy, Sz: NumPy arrays (2D or 3D) of velocity gradients

delta: grid spacing NumPy array

options: optional dictionary for solver configuration

### Returns
P: pressure field (same shape as input)

progress: convergence history (NumPy array)

### Solver Options
```python
options = {
    "Verbose": True,               # Print solver progress
    "SolverToleranceRel": 1e-4,   # Relative tolerance
    "SolverToleranceAbs": 1e-6,   # Absolute tolerance
    "Kernel": "face-crossing",     # or 'cell-centered'
}
```

---

## Supported Hardware
CPU: x86-64, any modern processor

GPU (optional): NVIDIA Pascal or newer 

RAM: 4 GB minimum, 16+ GB recommended for large grids

---

## Citation
If you use this solver in your work, please cite:

Zigunov, F., & Charonko, J. J. (2024). One-shot omnidirectional pressure integration through matrix-free multiscale methods. Measurement Science and Technology.

---

## License
GNU General Public License v3.0 (GPL-3.0) 

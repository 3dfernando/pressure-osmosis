# OSMODI Python Solver

GPU-accelerated Poisson solver for pressure integration from PIV (Particle Image Velocimetry) data.

## What This Does

Solves the Poisson equation for pressure from velocity gradient data:
- **GPU solver** (CUDA) - Fast, for NVIDIA GPUs
- **CPU solver** (OpenMP) - Fallback, works everywhere
- Supports 2D and 3D grids
- Float32 and float64 precision

---

## Installation

### Quick Install (Recommended)
```bash
pip install osmodi-solver
```

**Requirements:**
- Python 3.10, 3.11, 3.12, or 3.13
- Windows 64-bit
- NumPy (installed automatically)

### Build from Source (Advanced)

**Prerequisites:**
- **Python 3.10+**
- **CMake 3.18+** - [Download](https://cmake.org/download/)
- **Visual Studio 2017+** with "Desktop development with C++" - [Download](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
- **NVIDIA CUDA Toolkit 10.0+** (for GPU support) - [Download](https://developer.nvidia.com/cuda-downloads)

**Steps:**
```bash
git clone [repository-url]
cd OmniPython
pip install .
```

**For GPU support:** Requires NVIDIA GPU (GTX 10-series+, Pascal 2016+) and CUDA Toolkit.

See `docs/installation.md` for detailed instructions and troubleshooting.

---

## Quick Start
```python
import osmodi
import numpy as np

# Check available solvers
print(f"GPU: {osmodi.GPU_AVAILABLE}, CPU: {osmodi.CPU_AVAILABLE}")

# Create velocity gradient data
Sx = np.random.randn(64, 64, 64).astype(np.float32)
Sy = np.random.randn(64, 64, 64).astype(np.float32)
Sz = np.random.randn(64, 64, 64).astype(np.float32)
delta = np.array([1.0, 1.0, 1.0])

# Solve for pressure
P, progress = osmodi.solve_gpu(Sx, Sy, Sz, delta)
```

---

## Usage

### Solver Options
```python
options = {
    'Verbose': True,
    'SolverToleranceRel': 1e-4,
    'SolverToleranceAbs': 1e-4,
    'Kernel': 'face-crossing'
}

P, prog = osmodi.solve_gpu(Sx, Sy, Sz, delta, options)
```

See `docs/API_reference.md` for complete API documentation.

---

## Acknowledgments

This project is supported by:
- **Alfred P. Sloan Foundation** (Grant #G2023-20946)
- **Open Source Programs Office (OSPO)**

---

## Contributors

**Python Bindings Development:**
- Sohail Mulla

**Original OSMODI Solver:**
- Dr. Fernando Zigunov

---

## License

GNU General Public License v3.0 (GPL-3.0) - See `LICENSE.txt`
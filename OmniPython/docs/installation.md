# Installation Guide

Complete installation instructions for the OSMODI Python solver.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Windows Installation](#windows-installation)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required (All Platforms)
- **Python 3.8 or newer**
- **CMake 3.18 or newer** (required for CUDA support)
- **C++ Compiler**
  - Windows: Visual Studio 2017 or newer (with C++ tools)
- **pip** (usually comes with Python)

### (For GPU Support)
- **NVIDIA CUDA Toolkit 10.0 or newer**
- **NVIDIA GPU**
 Supported: RTX 20/30/40/50 series

### Python Packages
These are installed automatically:
- `numpy >= 1.20.0`
- `pybind11`

---

## Windows Installation

### Step 1: Install Prerequisites

#### 1.1 Install Python
Download from [python.org](https://www.python.org/downloads/)
- Minimum version: Python 3.8
- Check "Add Python to PATH" during installation
- Verify: Open Command Prompt and run `python --version`

#### 1.2 Install CMake
Download from [cmake.org](https://cmake.org/download/)
- Minimum version: CMake 3.18
- Select "Add CMake to system PATH" during installation
- Verify: `cmake --version`

#### 1.3 Install Visual Studio Build Tools
Download [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
- During installation, select: **"Desktop development with C++"**
- This includes MSVC compiler
- Visual Studio 2017 or newer is supported

#### 1.4 (Optional) Install CUDA Toolkit
For GPU support, download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- Minimum version: CUDA 10.0
- Recommended: CUDA 11.0 or newer
- Choose your Windows version
- Install with default settings
- Verify: `nvcc --version`

**Supported GPUs:**
- NVIDIA GTX 10-series or newer (compute capability 6.0+)
- Examples: GTX 1060, RTX 2060, RTX 3070, RTX 4090

### Step 2: Install OSMODI

Open Command Prompt or PowerShell:
```bash
# Navigate to the osmodi-solver folder
cd path\to\osmodi-solver

# Install the package
pip install .
```

**What happens:**
- CMake detects your GPU (if CUDA installed)
- Compiles GPU solver (if CUDA found and GPU supported)
- Compiles CPU solver (always)
- Installs to your Python environment

**Installation takes:** 2-5 minutes (longer first time, compiles for multiple GPU architectures)

### Step 3: Verify Installation
```bash
# Check if both solvers are available
python -c "import osmodi; print(f'GPU: {osmodi.GPU_AVAILABLE}, CPU: {osmodi.CPU_AVAILABLE}')"
```

**Expected output:**
- With CUDA and supported GPU: `GPU: True, CPU: True`
- Without CUDA or unsupported GPU: `GPU: False, CPU: True`

---

## Verifying Installation

### Quick Test Script

Create a file `test_install.py`:
```python
import osmodi
import numpy as np

print("OSMODI Installation Test")

# Check available solvers
print(f"GPU Solver Available: {osmodi.GPU_AVAILABLE}")
print(f"CPU Solver Available: {osmodi.CPU_AVAILABLE}")

if not (osmodi.GPU_AVAILABLE or osmodi.CPU_AVAILABLE):
    print("\n ERROR: No solvers available!")
    exit(1)

# Test with small 3D array
print("\nRunning quick test (32x32x32 grid)...")
Sx = np.random.randn(32, 32, 32).astype(np.float32)
Sy = np.random.randn(32, 32, 32).astype(np.float32)
Sz = np.random.randn(32, 32, 32).astype(np.float32)

try:
    if osmodi.GPU_AVAILABLE:
        P, _ = osmodi.solve_gpu(Sx, Sy, Sz, np.array([1.0]), {'Verbose': False})
        print(f" GPU solver works! Output shape: {P.shape}")
    
    if osmodi.CPU_AVAILABLE:
        P, _ = osmodi.solve_cpu(Sx, Sy, Sz, np.array([1.0]), {'Verbose': False})
        print(f" CPU solver works! Output shape: {P.shape}")
    
    print("\nInstallation successful!")
except Exception as e:
    print(f"\nError during test: {e}")
    exit(1)
```

Run it:
```bash
python test_install.py
```

---

## Troubleshooting

### "CMake not found" or "CMake version too old"
**Problem:** `pip install` fails with CMake error

**Solution:**
- Install CMake 3.18+ from [cmake.org](https://cmake.org/download/), ensure "Add to PATH" is checked
- Verify: `cmake --version` should show 3.18 or higher

### "Could not find Visual Studio"
**Problem:** Windows can't find MSVC compiler

**Solution:**
1. Install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. During installation, select **"Desktop development with C++"**
3. Restart terminal after installation

### "pybind11Config.cmake not found"
**Problem:** CMake can't find pybind11

**Solution:**
This should be fixed automatically by `pyproject.toml`. If you see this:
```bash
pip install pybind11
pip install --no-cache-dir .
```

### "GPU: False" but I have CUDA installed
**Problem:** GPU solver didn't compile even with CUDA

**Possible causes:**

1. **GPU too old (compute capability < 6.0)**
   - Check your GPU: [CUDA GPUs List](https://developer.nvidia.com/cuda-gpus)
   - Minimum: GTX 1050 (Pascal architecture, 2016)

2. **CUDA not in PATH**
   - Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin` to PATH
   - Verify: `nvcc --version`

3. **CUDA version too old**
   - Minimum: CUDA 10.0
   - Check: `nvcc --version`
   - Update from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

4. **Reinstall without cache**
```bash
   pip uninstall osmodi-solver -y
   pip install --no-cache-dir .
```

### "ImportError: DLL load failed" (Windows)
**Problem:** Python can't load compiled modules

**Solution:**
- Install [Visual C++ Redistributable (latest)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- Restart your terminal

### "Both GPU and CPU are False"
**Problem:** Importing from wrong location

**Solution:**
Make sure you're NOT in the `osmodi-solver` source directory:
```bash
cd ..  # Move out of source directory
python -c "import osmodi; print(osmodi.__file__)"
```

Should show: `...site-packages/osmodi/__init__.py`
NOT: `...osmodi-solver/osmodi/__init__.py`

### Installation takes very long
**Problem:** Compilation is slow

**Why:** The GPU solver compiles for multiple GPU architectures

**This is normal:** First installation may take 5-10 minutes. Subsequent installs are faster (uses cache).

### Performance Issues
**Problem:** Solver is slow

**Check:**
1. Are you using GPU solver when available?
```python
   import osmodi
   print(f"GPU Available: {osmodi.GPU_AVAILABLE}")
```

2. Monitor GPU usage:
   - Windows: Task Manager → Performance → GPU

3. Make sure your GPU is not in power-saving mode

---

## Supported Hardware

### Minimum Requirements
- **CPU:** Any x86-64 processor
- **RAM:** 4GB (16GB+ recommended for large grids)
- **GPU (optional):** NVIDIA GPU with compute capability 6.0+

### Tested Configurations
- **Windows 10/11** with Visual Studio 2019/2022
- **CUDA 11.0 - 13.x**
- **Python 3.8 - 3.13**

---

## Getting Help

If you encounter issues not covered here:

1. **Check you're using compatible versions:**
```bash
   python --version   # 3.8+
   cmake --version    # 3.18+
   nvcc --version     # 10.0+ (if using GPU)
   pip show osmodi-solver
```

2. **For GPU issues, check your hardware:**
```bash
   nvidia-smi  # Should show your GPU and driver version
```

3. **Open an issue on GitHub** with:
   - Your operating system and version
   - Python version (`python --version`)
   - CMake version (`cmake --version`)
   - CUDA version (`nvcc --version`, if applicable)
   - GPU model (`nvidia-smi`, if applicable)
   - Full error message
   - Output of `pip install .` (with `--verbose` flag)

---

## Uninstalling
```bash
pip uninstall osmodi-solver
```

This removes the package but keeps your Python environment intact.
# OSMODI Package Maintenance Guide

Guide for maintaining and updating the osmodi-solver Python package.

---

## Overview

This document explains how to:
- Rebuild wheels for new Python versions
- Upload new versions to PyPI
- Maintain the codebase

---

## Yearly Maintenance (When New Python Versions Release)

**Frequency:** Once per year (October when new Python version releases)

**Wait time:** 3-6 months after new Python release for adoption


### Step 1: Update Python Version List

Edit `setup.py`:
```python
python_requires='>=3.10',
classifiers=[
    'Programming Language :: Python :: 3.14',  # Add new version
    ...
]
```

Edit `.github/workflows/test-cpu.yml`:
```yaml
python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]  # Add new
```

### Step 2: Rebuild Wheels

**Requirements:**
- Working build machine with CUDA 12.6, Visual Studio 2022, CMake
- Python 3.10, 3.11, 3.12, 3.13 (and new version) installed

**Commands:**
```bash
cd osmodi-solver

# Delete old wheels
del dist\*.whl

# Build for each Python version
py -3.10 -m build --wheel
py -3.11 -m build --wheel
py -3.12 -m build --wheel
py -3.13 -m build --wheel
py -3.14 -m build --wheel  # New version
```

### Step 3: Test Wheels

```bash
# Test each wheel
pip install dist\osmodi_solver-X.X.X-cp314-cp314-win_amd64.whl
python -c "import osmodi; print(osmodi.GPU_AVAILABLE, osmodi.CPU_AVAILABLE)"
```

### Step 4: Upload to PyPI

```bash
pip install twine
twine upload dist\*.whl
```

**When prompted:**
- Username: `__token__`
- Password: [PyPI API token]

---

## Releasing New Version

### When to Release

**Patch version (0.1.X):**
- Bug fixes
- Documentation updates
- No code changes

**Minor version (0.X.0):**
- New features
- New Python version support
- Significant improvements

**Major version (X.0.0):**
- Breaking API changes
- Major rewrites

### Release Process

**Step 1: Update version**

Edit `setup.py`:
```python
version='0.2.0',  # Increment version
```

Edit `osmodi/__init__.py`:
```python
__version__ = '0.2.0'  # Match setup.py
```

**Step 2: Run tests**

```bash
pytest tests/ -v
```

**Step 3: Rebuild wheels**

```bash
del dist\*.whl
py -3.10 -m build --wheel
py -3.11 -m build --wheel
py -3.12 -m build --wheel
py -3.13 -m build --wheel
```

**Step 4: Upload to PyPI**

```bash
twine upload dist\*.whl
```

## Working Build Environment

**As of January 2026, known working configuration:**

- **Operating System:** Windows 10/11 64-bit
- **Python:** 3.10, 3.11, 3.12, 3.13
- **CUDA Toolkit:** 12.6
- **Visual Studio:** 2022
- **CMake:** 3.28+

---

## PyPI Account Management

### Current Setup
- **Package name:** osmodi-solver
- **Owner:** suskytop PyPI account
- **Maintainers:** Sohail Mulla (collaborator access)


### Adding New Maintainers

1. Go to https://pypi.org/project/osmodi-solver/
2. Manage → Collaborators
3. Add user with "Maintainer" role

---

## Troubleshooting Build Issues

### Compilation Fails

**Check:**
1. Visual Studio installed with C++ tools
2. CMake in PATH: `cmake --version`
3. For GPU: CUDA in PATH: `nvcc --version`

**Common fixes:**
```bash
# Clear build cache
rmdir /s build
rmdir /s dist

# Reinstall
pip install --no-cache-dir -e .
```

### Wheel Build Fails

**Check Python launcher:**
```bash
py --list  # Shows installed Python versions
```

**Ensure you have all versions installed:**
- py -3.10, py -3.11, py -3.12, py -3.13

---

## Code Structure

### Key Files

**Package code:**
- `osmodi/__init__.py` - Package entry, imports solvers
- `osmodi/src/gpu/osmodi_gpu.cu` - GPU solver (CUDA)
- `osmodi/src/cpu/osmodi_cpu.cpp` - CPU solver (OpenMP)

**Build:**
- `setup.py` - Package configuration
- `pyproject.toml` - Modern Python packaging metadata
- `osmodi/src/gpu/CMakeLists.txt` - GPU build config
- `osmodi/src/cpu/CMakeLists.txt` - CPU build config

**Documentation:**
- `docs/API_reference.md` - Function documentation
- `docs/installation.md` - Setup guide
- `docs/examples/` - Example scripts





## Contact

**Current Maintainers:**
- Sohail Mulla - somulla@syr.edu
- Dr. Fernando Zigunov (Original OSMODI)

**For issues:** Open GitHub issue or contact maintainers.

---

## Quick Reference Commands

```bash
# Run tests
pytest tests/ -v

# Build wheels
py -3.10 -m build --wheel
py -3.11 -m build --wheel
py -3.12 -m build --wheel
py -3.13 -m build --wheel

# Upload to PyPI
twine upload dist/*.whl

# Tag release
git tag v0.X.X
git push origin v0.X.X
```
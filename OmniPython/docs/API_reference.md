# API Reference

Complete API documentation for the OSMODI Python solver.

---

## Package Import

```python
import osmodi
```

---

## Module Attributes

### `osmodi.__version__`
- **Type:** `str`
- **Description:** Package version string
- **Example:** `'0.1.1'`

### `osmodi.GPU_AVAILABLE`
- **Type:** `bool`
- **Description:** True if GPU solver is available (requires NVIDIA GPU with CUDA drivers)

### `osmodi.CPU_AVAILABLE`
- **Type:** `bool`
- **Description:** True if CPU solver is available (should always be True)

---

## Functions

### `osmodi.solve_gpu(Sx, Sy, Sz, delta, options={})`

GPU Poisson solver using CUDA.

#### Parameters

**`Sx, Sy, Sz`** : numpy.ndarray
- Velocity gradient source terms (∂u/∂x, ∂v/∂y, ∂w/∂z)
- Shape: `(Nx, Ny, Nz)` for 3D or `(Nx, Ny)` for 2D
- Type: `float32` or `float64`
- Note: For 2D problems, set `Sz = np.zeros_like(Sx)`

**`delta`** : numpy.ndarray
- Grid spacing in physical units
- Array for non-uniform: `delta = np.array([dx, dy, dz])`

**`options`** : dict, optional
- Solver configuration (see [Solver Options](#solver-options))
- Default: `{}`

#### Returns

**`P`** : numpy.ndarray
- Computed pressure field
- Same shape and dtype as input

**`progress`** : dict
- `'iterations'` (int): Iterations performed
- `'residual'` (float): Final residual
- `'converged'` (bool): Convergence status

#### Raises

- **`RuntimeError`**: GPU unavailable or CUDA error
- **`ValueError`**: Invalid inputs

---

### `osmodi.solve_cpu(Sx, Sy, Sz, delta, options={})`

CPU Poisson solver.

#### Parameters

Same as `solve_gpu()`.

#### Returns

Same as `solve_gpu()`.

#### Raises

- **`RuntimeError`**: CPU solver error
- **`ValueError`**: Invalid inputs

#### Notes

- Works on any Windows system
- Uses available CPU cores
- Deterministic results

---

## Solver Options

Configuration dictionary for solver behavior. All optional.

### `'Verbose'`
- **Type:** `bool`
- **Default:** `False`
- **Description:** Print iteration progress

### `'SolverToleranceRel'`
- **Type:** `float`
- **Default:** `1e-4`
- **Description:** Relative convergence tolerance

### `'SolverToleranceAbs'`
- **Type:** `float`
- **Default:** `1e-4`
- **Description:** Absolute convergence tolerance

### `'Kernel'`
- **Type:** `str`
- **Default:** `'face-crossing'`
- **Options:** `'face-crossing'` or `'cell-centered'`
- **Description:** Finite difference kernel type

### `'SolverDevice'`
- **Type:** `str`
- **Options:** `'GPU'` or `'CPU'`
- **Description:** Force specific solver (normally auto-detected)

**Example:**
```python
options = {
    'Verbose': True,
    'SolverToleranceRel': 1e-6,
    'Kernel': 'face-crossing'
}
```

---

## Input Requirements

- All arrays (Sx, Sy, Sz) must have identical shapes
- Supported dtypes: `float32` or `float64`
- Arrays are automatically converted to Fortran-style (column-major) layout
- Grid spacing (delta) must be positive

---

## See Also

- [Installation Guide](installation.md)
- [README](README.md)
- [Examples](examples/) - taylor_green_2d.py, taylor_green_3d.py, tripv_bl.py
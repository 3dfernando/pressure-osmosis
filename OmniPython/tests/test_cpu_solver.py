"""Test CPU solver specific behavior."""
import numpy as np
import pytest

np.random.seed(0)  

def test_cpu_solver_small_3d():
    """Test CPU solver on small 3D problem."""
    import osmodi
    assert osmodi.CPU_AVAILABLE, "CPU solver MUST be available in installation!"

    Sx = np.random.randn(8, 8, 8).astype(np.float32)
    Sy = np.random.randn(8, 8, 8).astype(np.float32)
    Sz = np.random.randn(8, 8, 8).astype(np.float32)
    delta = np.array([1.0])

    P, progress = osmodi.solve_cpu(Sx, Sy, Sz, delta, {'Verbose': False})

    assert P.shape == Sx.shape
    assert P.dtype == Sx.dtype
    assert isinstance(progress, np.ndarray)


def test_cpu_solver_2d():
    """Test CPU solver on small 2D problem."""
    import osmodi
    assert osmodi.CPU_AVAILABLE

    Sx = np.random.randn(16, 16).astype(np.float64)
    Sy = np.random.randn(16, 16).astype(np.float64)
    Sz = np.zeros_like(Sx)
    delta = np.array([1.0])

    P, progress = osmodi.solve_cpu(Sx, Sy, Sz, delta, {'Verbose': False})

    assert P.shape == Sx.shape
    assert P.dtype == np.float64

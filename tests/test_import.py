"""Test package imports."""
import pytest


def test_import_osmodi():
    """Test that osmodi package imports successfully."""
    import osmodi
    assert osmodi is not None
    assert hasattr(osmodi, '__version__')


def test_solvers_available():
    """Test that at least one solver is available."""
    import osmodi
    assert osmodi.GPU_AVAILABLE or osmodi.CPU_AVAILABLE
"""OSMODI GPU-accelerated Poisson solver"""
__version__ = '0.1.1'

try:
    from .osmodi_gpu_bind import solver as _solve_gpu_raw
    GPU_AVAILABLE = True
except ImportError:
    _solve_gpu_raw = None
    GPU_AVAILABLE = False

try:
    from .osmodi_cpu_bind import solver as _solve_cpu_raw
    CPU_AVAILABLE = True
except ImportError:
    _solve_cpu_raw = None
    CPU_AVAILABLE = False

def solve_gpu(Sx, Sy, Sz, delta, options=None):
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU solver not available")
    if options is None:
        options = {}
    if 'SolverDevice' not in options:
        options['SolverDevice'] = 'GPU'
    return _solve_gpu_raw(Sx, Sy, Sz, delta, options)

def solve_cpu(Sx, Sy, Sz, delta, options=None):
    if not CPU_AVAILABLE:
        raise RuntimeError("CPU solver not available")
    if options is None:
        options = {}
    if 'SolverDevice' not in options:
        options['SolverDevice'] = 'CPU'
    return _solve_cpu_raw(Sx, Sy, Sz, delta, options)

__all__ = ['solve_gpu', 'solve_cpu', 'GPU_AVAILABLE', 'CPU_AVAILABLE', '__version__']
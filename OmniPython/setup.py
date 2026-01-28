"""
OSMODI Solver - GPU-accelerated Poisson solver for Python
Setup script with CMake build integration
"""
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Check if CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Create build directory
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)
        
        import pybind11
        pybind11_dir = pybind11.get_cmake_dir()

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPython_EXECUTABLE={sys.executable}',
            f'-Dpybind11_DIR={pybind11_dir}',  
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if sys.platform.startswith('win'):
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        # Run CMake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

# Check CUDA availability
def check_cuda_available():
    """Check if CUDA toolkit is available"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

# Determine which extensions to build
extensions_to_build = []

# Always build CPU version (fallback)
extensions_to_build.append(
    CMakeExtension('osmodi.osmodi_cpu_bind', sourcedir='osmodi/src/cpu')
)

# Build GPU version if CUDA available
if check_cuda_available():
    print("CUDA detected - building GPU solver")
    extensions_to_build.append(
        CMakeExtension('osmodi.osmodi_gpu_bind', sourcedir='osmodi/src/gpu')
    )
else:
    print("CUDA not detected - building CPU solver only")

setup(
    name='osmodi-solver',
    version='0.1.1',
    author='Sohail Mulla',
    author_email='suskytoplab@gmail.com', 
    description='GPU-accelerated Poisson solver for pressure from PIV',
    long_description=Path(__file__).parent.joinpath("PYPI.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    url='https://github.com/3dfernando/pressure-osmosis', 
    packages=find_packages(),
    ext_modules=extensions_to_build,
    cmdclass={'build_ext': CMakeBuild},
    install_requires=[
        'numpy>=1.20.0',
    ],
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: C++',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    keywords='pressure-from-piv particle-image-velocimetry PIV poisson poisson-solver gpu fluid-dynamics solver',
    zip_safe=False,
)
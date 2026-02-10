# OSMOSIS - One-Shot Matrix-based Omnidirectional Simultaneous pressure-Integration Solver for pressure from PIV

*If you use this code in your PIV project, please cite refs. [2] and [3] in your paper.* 

## Sample Results

Time-resolved PIV on a boundary layer:

<video width="700" controls>
  <source src="ImageSamples/Boundary_Layer.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

3D reconstruction of pressure, velocity, temperature and density from PIV on a double-fin SBLI flow at M=2:

<video width="500" controls>
  <source src="ImageSamples/SBLI_4field.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Summary

Two versions of the MODI method are implemented in this code. The I-MODI is an iterative solver, whereas the OS-MODI is a one-shot solver. The two solvers implement the same "idea", i.e., taking the omnidirectional integration to the limit of infinite parallel rays from the perspective of a single cell to devise coefficients for a matrix equation. 

The original ODI method by Liu, Moreto, and Mitchell [1] integrates the pressure-gradient field measured with PIV (i.e., the sum of the convective) by casting a bundle of parallel rays on the entire domain for a large count of different ray angles and performing integrals starting from a guess at the boundary of the domain. The boundary guess is updated after all integrals are performed; and a new iteration can be attempted. Multiple (100's to 1000's) of iterations are required to fully converge the boundaries.

The I-MODI solver uses the matrix inversion to perform the same set of integrals performed in one of the iterations of the ODI. Therefore, the same number (100's to 1000's) of iterations is still required to converge the pressure field solution. The full description of the solver is provided in our paper [2].

The OS-MODI solver performs a single matrix inversion that provides the pressure solution in one iteration only. This is done by finding the fixed point of the matrix equation in the I-MODI method (i.e., taking the number of iterations to infinity). The mathematics are further outlined in our follow-up paper [3].

## Installation - Matlab code - GPU-accelerated
The Matlab implementation is a mex file. A compiled .mexw64 file is provided under Releases, but it may not work in your machine (it was compiled in a x64 machine, Windows 10, Matlab R2023b). If it doesn't work, it will throw an error of the type: *MEX error: The specified module could not be found.*

It is best if you compile it for your own combination of Matlab version and OS. Compiling the program is straightforward:
1. Make sure your Matlab has the *parallel computing toolbox* installed. This app will not work without it. To check, in the command line you can type *ver* and look in the list if the *parallel computing toolbox* is present.
2. Download the file *Matlab/OSMODI.cu* from this repository. Place it in a folder of your choice. Best if you place it in a folder in Matlab's search path (say, C:/Users/your_user/Documents/Matlab/)
3. In Matlab, make sure the current path is the folder where OSMODI.cu is located. 
4. Run in the command line *mexcuda OSMODI.cu*
5. This should create a file OSMODI.mexw64 or OSMODI.mexa64 in the same folder, if it is successful.
6. If it throws an error:
  - Try installing the CPU-only version of this code (see next section)
  - Please let us know (you can contact the corresponding author at *fzigunov@syr.edu*) and we will try to provide assistance. Make sure you let us know your Matlab version, Operating System, and the error message you got.


## Installation - Matlab code - CPU-only
If the above installation procedure doesn't work, it may be because your system does not have a supported NVidia GPU. In this case, you may want to try our code using the CPU-only version. The CPU version is slower, even when using parallel processing. However, for 2D fields the performance difference is not quite as large and it may be sufficient for your application.

To compile the CPU-only code:
1. Your Matlab still needs the *parallel computing toolbox* installed, as the CPU version is multi-threaded.
2. Download the file *Matlab/OSMODI_CPU.cpp* from this repository. Place it in a folder of your choice. Best if you place it in a folder in Matlab's search path (say, C:/Users/your_user/Documents/Matlab/)
3. In Matlab, make sure the current path is the folder where OSMODI_CPU.cpp is located. 
4. Run in the command line *mex OSMODI_CPU.cpp*
5. This should create a file OSMODI_CPU.mexw64 or OSMODI_CPU.mexa64 in the same folder, if it is successful.
6. If it still throws an error, please contact the corresponding author at *fzigunov@syr.edu* and we will try to provide assistance. Make sure you let us know your Matlab version, Operating System, and the error message you got.



## Usage - Matlab Code
The Matlab implementation is a mex file. After you compile it (see above), you can use the following code to run it in Matlab:

*options struct and function prototype*
> opts.SolverToleranceRel=1e-4;  
> opts.SolverToleranceAbs=1e-6;   
> opts.SolverDevice='GPU';  %use 'CPU' or 'GPU'  
> opts.Kernel='cell-centered';   %use 'cell-centered' for more accuracy or 'face-crossing' for slightly faster processing  
> opts.Verbose=0; %use 1 if you want convergence info to be printed    
> \[P_OSMODI, CGS\]=OSMODI(dpdx, dpdy, dpdz , [dx dy dz], opts);

The advantage of the Matlab implementation is the one-line access to the pressure output by the solver. The Matlab function was only implemented for the ***one-shot solver***. The function prototype and usage can be found in the files Test_Taylor2D.m, Test_Taylor3D.m inside Example_SyntheticData. Examples with real data can be found in Example_TRPIV_2D and Example_NTRPIV_2D_AvgP for processing of 2D PIV snapshots for time-resolved and averaged pressure, respectively.

The structure 'opts' holds the options for this solver (shown above). These are case-sensitive. If the default options are to be used (the ones shown above), then you can provide an empty matrix for opts.

The inputs for Sx, Sy, Sz and \[dx, dy, dz\] are double-precision in the latest version. If you want to compute with single-precision you'll need to recompile the OSMODI.cu code by changing *typedef double varfloat*;  to *typedef single varfloat*; 

The outputs of the OSMODI function are P (Pressure field in same grid as dpdx) and CGS (the conjugate gradient solver convergence). CGS(:,1) = iteration; CGS(:,2)=residual; CGS(:,3)=time \[in seconds\].





## Usage - CUDA-C++ Independent Codes

The compiled CUDA-C++ implementations are executable files.
We ceased the support for the C++ implementations because the Matlab binding is far more convenient to use. We are working on a Python binding (future release).


The instructions below are kept for archival purposes. 

### Workflow 

1.  Generate the source term using the appropriate equations in your pre-processing program (say, Matlab). The equations vary depending on whether instantaneous or average pressure are sought, and whether compressibility is included. See example folder for usage example.

2.  Save source term as a VTK file using the vtkwrite function in Matlab. 

3.  Run the OmniCUDA.exe program with the appropriate Arguments.conf file in the same folder as the .exe file

4.  If succeeded, a Pressure_0.vtk (or some other name specified in SP_BoxOutputFile) is generated. 

5.  You can read this file using the function vtkread in Matlab (again, see example for details).

### Arguments.conf configuration file

The file uses an input argument text file called Arguments.conf. This file has the following contents:

*Iterative Method*
> SP_CGsolverToleranceRel 1e-4  
> SP_CGsolverToleranceAbs 1e-6  
> SP_SolverDevice GPU  
> SP_PressureSolverToleranceRel 1e-4  
> SP_BoxInputFile Z:\\Path to Whatever\\Filename\_\<frame\>.vtk  
> SP_BoxOutputFile Pressure\_\<frame\>.vtk  
> SP_NumberOfIterations 1000  
> SP_CheckpointIterations 50  
> SP_OverRelaxation 1

*One-shot Method*
> SP_CGsolverToleranceRel 1e-4  
> SP_CGsolverToleranceAbs 1e-6  
> SP_SolverDevice GPU  
> SP_BoxInputFile Z:\\Path to Whatever\\Filename\_<frame>.vtk  
> SP_BoxOutputFile Pressure\_\<frame\>.vtk  

The placeholder \<frame\> is not necessary. If present, all instances of \<frame\> are replaced by a number and the code is executed in a loop for all files in that folder matching the string pattern. Say, you have a file named Source_0.vtk, Source_1.vtk, etc. The tag can help with batch processing by using Source\_\<frame\>.vtk

The outputs will follow the same string pattern.

The tags for each argument (Say, SP_CGsolverToleranceRel) are not case sensitive but must be correctly spelled. 

Here’s the meaning for each of the tags:

**SP_CGsolverToleranceRel:** Relative tolerance (w.r.t. zero first guess) of the Conjugate Gradient matrix solver

**SP_CGsolverToleranceAbs:** Absolute tolerance for the Conjugate Gradient matrix solver

**SP_PressureSolverToleranceRel:** (Only for the iterative solver) Tolerance for the outer iterations of the pressure solver. i.e., “Momentum residual”, eq. 41 in paper: 𝜀_𝑅^(𝑛+1)=|[𝑊^(𝑛+1) ]{𝑃^(𝑛+1) }−[𝑊^𝑛 ]{𝑃^(𝑛+1) }−{𝑆}|/|{𝑆}|

**SP_BoxInputFile:** Path to input source term. Must be a *.vtk file (Paraview format). Must be a Rectilinear Grid. Recommended to use the special function vtkwrite.m to build this file: 

> vtkwrite(Filename, 'RECTILINEAR_GRID', x, y, z, 'vectors', 'SOURCE', SxN, SyN, SzN, 'BINARY');

The \*.vtk file must contain a vector field named SOURCE with the source term on a Ndgrid format (i.e., dimension order is x,y,z). The 1D vectors x, y and z are of the length of the corresponding dimensions and are evenly spaced (say, x=0:dx:xmax).

**SP_BoxOutputFile:** Filename for output pressure fields. We currently don’t support a path here, it will always output to the same folder where the .exe file is. The file must be a *.vtk and will be in rectilinear grid. Use the function vtkread.m to read the results.

**SP_NumberOfIterations:** (Only for the iterative solver) Maximum number of iterations for iterative solver (i.e., outputs even if SP_PressureSolverToleranceRel hasn’t been reached).

**SP_CheckpointIterations:** (Only for the iterative solver) Every “CheckpointIterations”, saves an output file (always overwrites). This is to check the progress in case the fields are large and take long to process

**SP_OverRelaxation:** (Only for the iterative solver) Implements over-relaxation to accelerate convergence. Doesn’t really work (i.e., it makes it diverge). Leave as 1.

**SP_SolverDevice:** Can be GPU or CPU. If GPU, attempts to use the GPU with the largest VRAM first. If there’s no GPU or it fails to find one, then it reverts to CPU (Multithreaded with OpenMP). The CPU implementation is ~20-100 times slower than the GPU.



## Systems supported

The standalone codes can be compiled to any system. We only tested in Windows machines. Visual Studio project files are provided for each project for convenience.

The Matlab binding was only compiled for Windows machines.

If you need to recompile the .mex file for your platform, you need to follow these steps:
- Make sure you have a Nvidia GPU, otherwise there will be no point;
- Download Nvidia GPU Computing Toolkit. You need to make sure the toolkit version is the exact one that your Matlab version was built for. This link has the tookit version for each Matlab version (hopefully it still works when you see this.)  
- Install Microsoft Visual Studio to get the correct compiler. You need the right version of Visual Studio for your Matlab, if the visual studio is too high it will not work. It is very temperamental so look up which one is the right one for you.
- Once all is set you should be able to run the line mexcuda OSMODI.cu and compile the \*.mexw64 file. This file now will be your function, which you can call by calling [P,CGS]=OSMODI(\~,\~,\~,\~,\~). Whatever the file name of the \*.mexw64 file is, is the name of the function you will call from within matlab.

The code is optimized for GPUs (NVidia only) using the CUDA-C++ library. If a compatible NVidia GPU is not detected or you choose in the input arguments *SP_SolverDevice = "CPU"*, then the CPU solver will be used instead. We used openMP for parallelizing CPU operations, but it still was considerably slower than the GPU solver in our tests (somewhere between 5 and 50 times slower in many cases).


## References cited

[1] Xiaofeng Liu, Jose R. Moreto and Seth Siddle-Mitchell. "Instantaneous Pressure Reconstruction from Measured Pressure Gradient using Rotating Parallel Ray Method," AIAA 2016-1049. 54th AIAA Aerospace Sciences Meeting. January 2016.

[2] Fernando Zigunov and John J Charonko 2024 Meas. Sci. Technol. 35 065302

[3] Zigunov, F., & Charonko, J. (2024). One-shot omnidirectional pressure integration through matrix inversion. arXiv [Physics.Flu-Dyn]. Retrieved from http://arxiv.org/abs/2402.09988
 
## Release
This project has been approved for open source release as O4661.


## Copyright notice

The bulk of this project is copyright Triad National Security, LLC. and distributed under GPLv3 or later.  

See each file for further information on the appropriate license, and COPYING for the full text of the GPLv3.  

© 2024. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

 This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

vtkread.m is derived from vtkread.m, Copyright (c) 2016 Joe Yeh and distributed under the MIT License. Modifications are copyright Triad National Security, LLC. 2024 and distributed under GPLv3 or later.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

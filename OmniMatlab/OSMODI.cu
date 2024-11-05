/*
 * © 2024. Triad National Security, LLC. All rights reserved.
 * This program was produced under U.S. Government contract
 * 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
 * operated by Triad National Security, LLC for the U.S. Department of
 * Energy/National Nuclear Security Administration. All rights in the
 * program are reserved by Triad National Security, LLC, and the U.S.
 * Department of Energy/National Nuclear Security Administration. The
 * Government is granted for itself and others acting on its behalf a
 * nonexclusive, paid-up, irrevocable worldwide license in this material
 * to reproduce, prepare. derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit
 * others to do so.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <https://www.gnu.org/licenses/>.
 */
 
//============================================================================
//============================================================================
//====================OMNIDIRECTIONAL CUDA MATRIX SOLVER======================
//============================================================================
//============================================================================
//Developed by Fernando Zigunov and John Charonko (2023) - Extreme Fluids Group - Los Alamos National Laboratory
//R01 - This is the ONE-SHOT version of the omnidirectional matrix solver. (i.e., one solution of the CG solver solves for the correct pressure field.)
//R02 - Added preconditioner to CG solver
//R03 (Nov 2024) - Added new kernel mode: "face-crossing" (same as previous versions, 5- or 7-point Laplacian); "cell-centered" (9- or 27-point kernel from parallel rays); 
//                 Added double handling functionality (double is default now; you can change varfloat to single and recompile if you need memory)
//                 Added CG restart if it does not converge to desired tolerance
//                 Fixed memory leak (RHS variable was not freed before returning at the main function)


//====This is a Matlab implementation using mexcuda.====
//The variables are transferred directly from Matlab memory to here.
//Compile with "mexcuda OSMODI.cu"
//If varfloat == double, make sure to follow this tutorial (https://stackoverflow.com/questions/52741049/how-can-i-specify-a-minimum-compute-capability-to-the-mexcuda-compiler-to-compil) to have Matlab use the NVCC sm_60 or greater tag, otherwise the atomicAdd function will not work (for previous versions of GPUs, the atomicAdd only worked with floats, not doubles)
//
//Call this function as:
//[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz); %Basic form, source term Sx, Sy, Sz are either 2D matrices or 3D matrices. 
//                     We use **ND grid** format here. Uses default options, and delta=1.
//                     If the fields are 2D, please provide Sz anyways, it can be zeros(size(Sx)).
//[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, delta); %Also provides a grid spacing delta which is the same in all directions.
//[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, [dx dy dz]); %Provides a grid spacing that is different for x, y, z but is still constant for each direction.
//[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, delta, options); %Also provides options as a struct. See below.

//options.SolverToleranceRel (default is 1e-4) %Relative error allowed for the CG solver
//options.SolverToleranceAbs (default is 1e-4) %Relative error allowed for the CG solver
//options.SolverDevice (default is 'GPU') %Choose between 'CPU' and 'GPU'.
//options.Kernel (default is 'face-crossing') %Choose between 'face-crossing', 'cell-centered'. 'cell-centered' is slower but has a better behavior against measurement error.
//options.Verbose (default is '0') %To make the code print the output (Verbose=1 prints).


#pragma once

#include "mex.h"
#include <iostream>
#include <regex>
#include <fstream> 
#include <string>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <random>
#include <signal.h>

using namespace std;

#define PI 3.141592653589793
#define _X 0
#define _Y 1
#define _Z 2

#define ZEROS 0
#define ONES 1
#define RANDOMS 2

#define FACE_CROSSING 0
#define CELL_CENTERED 1

#define CPU 0
#define GPU 1

#define BLOCKDIM_VEC 512

clock_t tic; clock_t toc; double timeTask;
clock_t tic2; clock_t toc2; double timeTask2;

typedef double varfloat; //Change this to "float" for single precision or "double" for double precision. Found that "float" is ~5-7% faster (not really all that much)

// ====================Structure Definitions================ 
#pragma region

template <typename varfloat>
struct varfloat3 {
    varfloat x;
    varfloat y;
    varfloat z;
};

template <typename varfloat>
struct SolverParameters {
    bool Verbose = false; //Whether to send messages to Matlab console
    int SolverDevice = GPU; //CPU or GPU
    int Kernel = FACE_CROSSING; //FACE_CROSSING, CELL_CENTERED
    varfloat solverToleranceRel = 1e-4; //Error allowed for the solver
    varfloat solverToleranceAbs = 1e-4; //Error allowed for the solver
    int3 BoxGridPoints = { 100, 100, 1 }; //Number of grid points in the box
    varfloat3<varfloat> GridDelta; //Delta value for derivative approximation
    long long totalBoxElements; //Tracks the size of the box

    varfloat cx = 1; //Constant for the x-face
    varfloat cy = 1; //Constant for the y-face
    varfloat cz = 1; //Constant for the z-face
    varfloat cxy = 1; //Constant for the xy-edge
    varfloat cxz = 1; //Constant for the xz-edge
    varfloat cyz = 1; //Constant for the yz-edge
    varfloat cxyz = 1; //Constant for the xyz-corner
} ; // Structure to define the parameters of the solver; initializes to default values

template <typename varfloat>
struct BoxContents {
    varfloat* SourceFn_Field_X; //Stores the Source Function here
    varfloat* SourceFn_Field_Y; //Stores the Source Function here
    varfloat* SourceFn_Field_Z; //Stores the Source Function here
    int3 BoxGridSize = { 100, 100, 1 }; //Number of grid points in the box
    varfloat3<varfloat> GridDelta; //Delta value for derivative approximation
    long long totalBoxElements; //Tracks the size of the box
} ; // Structure to hold the contents of the 3D box 

struct Progress { //Struct to store the progress of the CG solver
    int Iteration;
    float Residual;
    double TimeSeconds;
};

vector<Progress> CGS_Progress;

#pragma endregion



// ====================Helper Functions================ 
#pragma region

bool iequals(const string& a, const string& b)
{
	//Adapted from answers in
    //https://stackoverflow.com/questions/11635/case-insensitive-string-comparison-in-c
    return std::equal(a.begin(), a.end(),
        b.begin(), b.end(),
        [](char a, char b) {
            return tolower(a) == tolower(b);
        });
}

template <typename varfloat>
__host__ __device__ inline int isnan2(varfloat x)
{
    //Apparently uses less registers than the original isnan, See:
    //https://stackoverflow.com/questions/33922103/is-isnan2-as-fast-as-testing-equality
    return x != x;
}

void ClockTic() {
    //Starts the clock
    tic = clock();
}

double ClockToc() {
    //Returns the current clock time  in seconds
    toc = clock() - tic;
    timeTask = ((double)toc) / CLOCKS_PER_SEC; // in seconds
    return timeTask;
}

template <typename varfloat>
void FillBox(varfloat* boxToFill, int BoxContents, SolverParameters<varfloat> SP) {
    //Fills the 3D box with one of the objects as dictated by BoxContents
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (int zz = 0; zz < SP.BoxGridPoints.z; zz++) {
        for (int yy = 0; yy < SP.BoxGridPoints.y; yy++) {
            for (int xx = 0; xx < SP.BoxGridPoints.x; xx++) {
                long long idxBox = (long long)xx + (long long)SP.BoxGridPoints.x * ((long long)yy + (long long)SP.BoxGridPoints.y * ((long long)zz));

                if (BoxContents == ZEROS) {
                    boxToFill[idxBox] = 0.0f;
                }
                else if (BoxContents == ONES) {
                    boxToFill[idxBox] = 1.0f;
                }
                else if (BoxContents == RANDOMS) {
                    boxToFill[idxBox] = distribution(gen);
                }
                else {
                    boxToFill[idxBox] = 0.0f;
                }
            }
        }
    }
}

bool InitializeGPU(SolverParameters<varfloat> SP) {
    //This function initializes the GPU code by first enumerating the devices available and choosing the highest one for these computations.
    //Returns true if initialization was successful.

    if (SP.Verbose){ mexPrintf("Detecting GPUs...\n");}
    int NumberOfGPUs;
    cudaGetDeviceCount(&NumberOfGPUs);

    if (NumberOfGPUs == 0) {
        return false; //No GPU available
        mexPrintf("\n ***Warning! No GPU found in the system!***\n");
    }

    size_t HighestGlobalMem = 0;
    string LargestDevName;
    int LargestDevID = -1;
    for (int i = 0; i < NumberOfGPUs; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        size_t ThisDeviceMem = prop.totalGlobalMem;
        if (HighestGlobalMem < ThisDeviceMem) {
            HighestGlobalMem = ThisDeviceMem;
            LargestDevID = i;
            LargestDevName = prop.name;
        }

        if (SP.Verbose){ mexPrintf("Device Number: %d\n", i);}
        if (SP.Verbose){ mexPrintf("  Device name: %s\n", prop.name);}
        if (SP.Verbose){ mexPrintf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1.0e3);}
        if (SP.Verbose){ mexPrintf("  Device Global Memory (GB): %f\n\n", ThisDeviceMem / 1.0e9);}
    }

    if (LargestDevID >= 0) {
        if (SP.Verbose){ mexPrintf("\nUsing Device: %s\n", LargestDevName.c_str());}
        cudaSetDevice(LargestDevID);
        return true;
    }
    else{
        printf("\n***Warning! Found GPU device but for some reason code failed!***\n");
        return false;
    }

}

double Az(double dx, double dy, double dz){
    //Function Az from ProjectionCoefficients Writeup (Zigunov&Charonko, MST, 2024, Supp. Material, Eq 96)
    double Az_val = 0.0;
    double N = 1e3; //Number of integral steps

    double phi, theta;   
    double phi_start = 0.0;
    double phi_end = atan(dy/dx); //alpha_1

    double dphi = (1.0/N) * (phi_end-phi_start);

    //First integral
    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_1 = atan(dx/(dz * cos(phi)));
        double dtheta = (1.0/N) * (beta_1);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_1);

            double Sz = (dx - dz * tan(theta) * cos(phi)) * (dy - dz * tan(theta) * sin(phi));

            Az_val += Sz * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }
    
    //Second integral
    phi_start = atan(dy/dx); //alpha_1
    phi_end = PI/2;

    dphi = (1.0/N) * (phi_end-phi_start);
    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_2 = atan(dy/(dz * sin(phi)));
        double dtheta = (1.0/N) * (beta_2);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_2);

            double Sz = (dx - dz * tan(theta) * cos(phi)) * (dy - dz * tan(theta) * sin(phi));

            Az_val += Sz * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }

    return 4*Az_val; //don't forget the 4!
}

double Axz(double dx, double dy, double dz){
    //Function Axz from ProjectionCoefficients Writeup (Zigunov&Charonko, MST, 2024, Supp. Material, Eq 97)
    double Axz_val = 0.0;
    double N = 1e3; //Number of integral steps

    double phi, theta;   

    //First integral
    double phi_start = 0.0;
    double phi_end = atan(dy/dx); //alpha_1

    double dphi = (1.0/N) * (phi_end-phi_start);

    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_1 = atan(dx/(dz * cos(phi)));
        double dtheta = (1.0/N) * (beta_1);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_1);

            double Sxz = (dz * tan(theta) * cos(phi)) * (dy - dz * tan(theta) * sin(phi));

            Axz_val += Sxz * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }
    
    //Second integral
    phi_start = atan(dy/dx); //alpha_1
    phi_end = PI/2;

    dphi = (1.0/N) * (phi_end-phi_start);

    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_2 = atan(dy/(dz * sin(phi)));
        double dtheta = (1.0/N) * (beta_2);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_2);

            double Sxz = (dz * tan(theta) * cos(phi)) * (dy - dz * tan(theta) * sin(phi));

            Axz_val += Sxz * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }
    
    //Third integral
    phi_start = 0.0;
    phi_end = atan(dy/dz); //alpha_1*

    dphi = (1.0/N) * (phi_end-phi_start);
    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_1s = atan(dz/(dx * cos(phi)));
        double dtheta = (1.0/N) * (beta_1s);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_1s);

            double Sxzs = (dx * tan(theta) * cos(phi)) * (dy - dx * tan(theta) * sin(phi));

            Axz_val += Sxzs * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }

    //Fourth integral
    phi_start = atan(dy/dz); //alpha_1*
    phi_end = PI/2;

    dphi = (1.0/N) * (phi_end-phi_start);

    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_2s = atan(dy/(dx * sin(phi)));
        double dtheta = (1.0/N) * (beta_2s);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_2s);

            double Sxzs = (dx * tan(theta) * cos(phi)) * (dy - dx * tan(theta) * sin(phi));

            Axz_val += Sxzs * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }

    return 2*Axz_val; //don't forget the 2!
}

double Axyz(double dx, double dy, double dz){
    //Function Az from ProjectionCoefficients Writeup (Zigunov&Charonko, MST, 2024, Supp. Material, Eq 101)
    double Axyz_val = 0.0;
    double N = 1e3; //Number of integral steps

    double phi, theta;   
    double phi_start = 0.0;
    double phi_end = atan(dy/dx); //alpha_1

    double dphi = (1.0/N) * (phi_end-phi_start);

    //First integral
    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_1 = atan(dx/(dz * cos(phi)));
        double dtheta = (1.0/N) * (beta_1);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_1);

            double Sxyz = dz * dz * tan(theta) * tan(theta) * sin(phi) * cos(phi);

            Axyz_val += Sxyz * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }
    
    //Second integral
    phi_start = atan(dy/dx); //alpha_1
    phi_end = PI/2;

    dphi = (1.0/N) * (phi_end-phi_start);
    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_2 = atan(dy/(dz * sin(phi)));
        double dtheta = (1.0/N) * (beta_2);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_2);

            double Sxyz = dz * dz * tan(theta) * tan(theta) * sin(phi) * cos(phi);

            Axyz_val += Sxyz * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }
    
    //Third integral
    phi_start = 0.0;
    phi_end = atan(dy/dz); //alpha_1*

    dphi = (1.0/N) * (phi_end-phi_start);
    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_1s = atan(dz/(dx * cos(phi)));
        double dtheta = (1.0/N) * (beta_1s);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_1s);

            double Sxyzs = dx * dx * tan(theta) * tan(theta) * sin(phi) * cos(phi);

            Axyz_val += Sxyzs * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }

    //Fourth integral
    phi_start = atan(dy/dz); //alpha_1*
    phi_end = PI/2;

    dphi = (1.0/N) * (phi_end-phi_start);

    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_2s = atan(dy/(dx * sin(phi)));
        double dtheta = (1.0/N) * (beta_2s);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_2s);

            double Sxyzs = dx * dx * tan(theta) * tan(theta) * sin(phi) * cos(phi);

            Axyz_val += Sxyzs * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }
    
    //Fifth integral
    phi_start = 0.0;
    phi_end = atan(dz/dx); //alpha_1+

    dphi = (1.0/N) * (phi_end-phi_start);
    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_1p = atan(dx/(dy * cos(phi)));
        double dtheta = (1.0/N) * (beta_1p);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_1p);

            double Sxyzp = dy * dy * tan(theta) * tan(theta) * sin(phi) * cos(phi);

            Axyz_val += Sxyzp * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }

    //Sixth integral
    phi_start = atan(dz/dx); //alpha_1p
    phi_end = PI/2;

    dphi = (1.0/N) * (phi_end-phi_start);

    for (double j = 0; j < N; j++){
        phi = ((j+0.5)/N) * (phi_end-phi_start) + phi_start;

        double beta_2p = atan(dz/(dy * sin(phi)));
        double dtheta = (1.0/N) * (beta_2p);
        for (double i = 0; i < N; i++){
            theta = ((i+0.5)/N) * (beta_2p);

            double Sxyzp = dy * dy * tan(theta) * tan(theta) * sin(phi) * cos(phi);

            Axyz_val += Sxyzp * cos(theta) * sin(theta) * dtheta * dphi;
        }
    }

    return Axyz_val; 
}

template <typename varfloat>
void PrecomputeConstants(SolverParameters<varfloat>* SP){
    //Only for the 3D cell-centered case, we need to precompute the constants by solving a 2D integral:
    double dx = (double) SP->GridDelta.x;
    double dy = (double) SP->GridDelta.y;
    double dz = (double) SP->GridDelta.z;

    dy = dy / dx; dz = dz / dx; dx = 1.0; // Normalizes to improve accuracy of integral

    //1. Computes face constants
    double A_z = Az(dx,dy,dz);
    double A_x = Az(dz,dy,dx);
    double A_y = Az(dx,dz,dy);

    //2. Computes edge constants
    double A_xz = Axz(dx,dy,dz);
    double A_xy = Axz(dx,dz,dy);
    double A_yz = Axz(dy,dx,dz);

    //3. Computes corner constant
    double A_xyz = Axyz(dx,dy,dz);

    //Total Constants
    double A_tot = 2 * (A_x + A_y + A_z) + 4 * (A_xy + A_xz + A_yz) + 8 * A_xyz;

    //Normalized constants for calculation
    SP->cx = A_x / A_tot;
    SP->cy = A_y / A_tot;
    SP->cz = A_z / A_tot;
    SP->cxy = A_xy / A_tot;
    SP->cxz = A_xz / A_tot;
    SP->cyz = A_yz / A_tot;
    SP->cxyz = A_xyz / A_tot;
}


#pragma endregion

// ====================GPU CUDA Function Kernels================
#pragma region

template <typename varfloat>
__global__ void GPU_FillNan(varfloat* PressureField, SolverParameters<varfloat>* SP) {
    //Fills the box with nan values in case the CG diverges
    long long xx = blockIdx.x * blockDim.x + threadIdx.x;
    long long yy = blockIdx.y * blockDim.y + threadIdx.y;
    long long zz = blockIdx.z * blockDim.z + threadIdx.z;

    if ((xx >= SP->BoxGridPoints.x) || (yy >= SP->BoxGridPoints.y) || (zz >= SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }

    long long GridX = (long long)SP->BoxGridPoints.x;
    long long GridY = (long long)SP->BoxGridPoints.y;
    long long idxCenter = xx + GridX * (yy + GridY * zz);
    PressureField[idxCenter] = NAN;
}

template <typename varfloat>
__global__ void printScalar_GPU(varfloat* a) {
    //For debugging purposes
    printf("%f\n", *a);
}

template <typename varfloat>
__global__ void printVector1_GPU(varfloat* v) {
    //For debugging purposes
    printf("%e; ", v[0]);
}

template <typename varfloat>
__global__ void printVector2_GPU(varfloat* v) {
    //For debugging purposes
    printf("%e\n", v[49]);
}

template <typename varfloat>
__global__ void addVectors_GPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    long long idxCenter = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;

    if (idxCenter >= (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }
    out[idxCenter] = a[idxCenter] + b[idxCenter];
}

template <typename varfloat>
__global__ void subtractVectors_GPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    long long idxCenter = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;

    if (idxCenter >= (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }
    out[idxCenter] = a[idxCenter] - b[idxCenter];
}

template <typename varfloat>
__global__ void divide(varfloat* num, varfloat* den, varfloat* out) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x == 0) {
        *out = *num / *den;
    }
}

template <typename varfloat>
__global__ void vectorDot_GPU_Slow(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    //naive slow implementation
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    varfloat tmp = 0.0;
    if (index_x == 0) {
        for (long long i = 0; i < (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z); i++) {
            if (!isnan2(a[i]) && !isnan2(b[i])) {
                tmp += a[i] * b[i];
                //printf("i=%lld; a=%f, b=%f, tmp=%f, \n", i, a[i], b[i],tmp);
            }
        }
        *out = tmp;
    }
}

template <typename varfloat>
__global__ void scalarVectorMult_GPU(varfloat* scalar, varfloat* a, varfloat* out, SolverParameters<varfloat>* SP) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z)) {
        out[index_x] = a[index_x] * *scalar;
    }
}

template <typename varfloat>
__global__ void vectorDot_GPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    __shared__ varfloat shared_tmp[BLOCKDIM_VEC];

    long long index = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long maxIdx = (long long)SP->BoxGridPoints.x * (long long)SP->BoxGridPoints.y * (long long)SP->BoxGridPoints.z;

    if (index == 0) {
        *out = 0.0; //We need to reset the output variable when we start
    }

    if (index < maxIdx) {
        if (!isnan2(a[index]) || !isnan2(b[index])) {
            shared_tmp[threadIdx.x] = a[index] * b[index]; //Dont norm nans
        }
        else {
            shared_tmp[threadIdx.x] = 0.0;//nans become zeros
        }
    }
    else {
        shared_tmp[threadIdx.x] = 0.0;
    }

    __syncthreads();

    // reduction within block
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // atomic add the partial reduction in out
    if (threadIdx.x == 0) {
        atomicAdd(out, shared_tmp[0]);
    }
}

template <typename varfloat>
__global__ void vectorAverage_GPU(varfloat* a, varfloat* sum, varfloat* numel, SolverParameters<varfloat>* SP) {
    __shared__ varfloat shared_sum[BLOCKDIM_VEC];
    __shared__ varfloat shared_numel[BLOCKDIM_VEC];

    long long index = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long maxIdx = (long long)SP->BoxGridPoints.x * (long long)SP->BoxGridPoints.y * (long long)SP->BoxGridPoints.z;

    if (index == 0) {
        *sum = 0.0; //We need to reset the output variable when we start
        *numel = 0.0; //We need to reset the output variable when we start
    }

    if (index < maxIdx) {
        if (!isnan2(a[index])) {
            shared_sum[threadIdx.x] = a[index]; //Dont add nans
            shared_numel[threadIdx.x] = 1.0;
        }
        else {
            shared_sum[threadIdx.x] = 0.0;//nans become zeros
            shared_numel[threadIdx.x] = 0.0;
        }
    }
    else {
        shared_sum[threadIdx.x] = 0.0;
        shared_numel[threadIdx.x] = 0.0;
    }

    __syncthreads();

    // reduction within block
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            shared_numel[threadIdx.x] += shared_numel[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // atomic add the partial reduction in out
    if (threadIdx.x == 0) {
        atomicAdd(sum, shared_sum[0]);
        atomicAdd(numel, shared_numel[0]);
    }
}

template <typename varfloat>
__global__ void subtractAverage_GPU(varfloat* a, varfloat* sum, varfloat* numel, SolverParameters<varfloat>* SP){
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    varfloat mean_a = *sum / *numel;
    if (index_x < (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z)) {
        a[index_x] = a[index_x] - mean_a;
    }
}


template <typename varfloat>
__global__ void MatrixMul_Omnidirectional_GPU(varfloat* Result, varfloat* PressureField, varfloat* RHS, SolverParameters<varfloat>* SP) {
    //This is the bit of code that performs the matrix multiplication Result=A*x (where A is the weight matrix and x is the PressureField)
    //The RHS of the equation is also provided so we can find the points where we have NAN's
    long long xx = blockIdx.x * blockDim.x + threadIdx.x;
    long long yy = blockIdx.y * blockDim.y + threadIdx.y;
    long long zz = blockIdx.z * blockDim.z + threadIdx.z;

    if ((xx >= SP->BoxGridPoints.x) || (yy >= SP->BoxGridPoints.y) || (zz >= SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }

    if (SP->BoxGridPoints.z == 1) {
        //2D Case
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;

        long long idxCenter = xx + GridX * (yy + GridY * zz);

        //dx and dy for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;

        if ((RHS[idxCenter] != RHS[idxCenter])) {
            //The RHS here is a nan, so simply makes the result at this point a nan as well
            Result[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_xm = idxCenter - 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_ym = idxCenter - GridX;

            if (SP->Kernel == FACE_CROSSING){
                //Face crossing Kernel uses less points.
                varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
                varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
                varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);

                varfloat rhs_cx = SP->cx;
                varfloat rhs_cy = SP->cy;

                //Adds the pressure values to right-hand side for this cell 
                varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym); //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;

                varfloat R = PressureField[idxCenter];
                R -= bxp ? 0.0 : rhs_cx * PressureField[idx_xp] / w_in_1; //done this way to prevent access outside allocated memory 
                R -= bxm ? 0.0 : rhs_cx * PressureField[idx_xm] / w_in_1;
                R -= byp ? 0.0 : rhs_cy * PressureField[idx_yp] / w_in_1;
                R -= bym ? 0.0 : rhs_cy * PressureField[idx_ym] / w_in_1;
                Result[idxCenter] = R;
            }
            else if(SP->Kernel == CELL_CENTERED){
                //Cell centered Kernel 
                long long idx_xpyp = idxCenter + GridX + 1;
                long long idx_xmyp = idxCenter + GridX - 1;
                long long idx_xpym = idxCenter - GridX + 1;
                long long idx_xmym = idxCenter - GridX - 1;

                varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
                varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
                varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);
                
                varfloat bxpyp = (((xx + 1) >= GridX) || ((yy + 1) >= GridY)) || (RHS[idx_xpyp] != RHS[idx_xpyp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                varfloat bxmyp = (((yy + 1) >= GridY) || ((xx - 1) < 0)) || (RHS[idx_xmyp] != RHS[idx_xmyp]);
                varfloat bxpym = (((yy - 1) < 0) || ((xx + 1) >= GridX)) || (RHS[idx_xpym] != RHS[idx_xpym]);
                varfloat bxmym = (((yy - 1) < 0) || ((xx - 1) < 0)) || (RHS[idx_xmym] != RHS[idx_xmym]);

                varfloat rhs_cx = SP->cx;
                varfloat rhs_cy = SP->cy;
                varfloat rhs_cxy = SP->cxy;

                //Adds the pressure values to right-hand side for this cell 
                varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym) + rhs_cxy * (bxpyp + bxmyp + bxpym + bxmym); //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;

                varfloat R = PressureField[idxCenter];
                R -= bxp ? 0.0 : rhs_cx * PressureField[idx_xp] / w_in_1; //done this way to prevent access outside allocated memory 
                R -= bxm ? 0.0 : rhs_cx * PressureField[idx_xm] / w_in_1;
                R -= byp ? 0.0 : rhs_cy * PressureField[idx_yp] / w_in_1;
                R -= bym ? 0.0 : rhs_cy * PressureField[idx_ym] / w_in_1;

                R -= bxpyp ? 0.0 : rhs_cxy * PressureField[idx_xpyp] / w_in_1; //corners
                R -= bxmyp ? 0.0 : rhs_cxy * PressureField[idx_xmyp] / w_in_1;
                R -= bxpym ? 0.0 : rhs_cxy * PressureField[idx_xpym] / w_in_1;
                R -= bxmym ? 0.0 : rhs_cxy * PressureField[idx_xmym] / w_in_1;
                Result[idxCenter] = R;
            }
        }
    }
    else {
        //3D Case
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long GridZ = (long long)SP->BoxGridPoints.z;

        long long idxCenter = xx + GridX * (yy + GridY * zz);

        //dx and dy and dz for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;
        varfloat GridDZ = SP->GridDelta.z;

        if (RHS[idxCenter] != RHS[idxCenter]) {
            //The RHS here is a nan, so simply makes the result at this point a nan as well
            Result[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_xm = idxCenter - 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_ym = idxCenter - GridX;
            long long idx_zp = idxCenter + GridX * GridY;
            long long idx_zm = idxCenter - GridX * GridY;

            varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 80) [i.e. isnan(X) is the same as X!=X]
            varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
            varfloat bzp = ((zz + 1) >= GridZ) || (RHS[idx_zp] != RHS[idx_zp]);
            varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
            varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);
            varfloat bzm = ((zz - 1) < 0) || (RHS[idx_zm] != RHS[idx_zm]);

            if (SP->Kernel == FACE_CROSSING){
                //Face crossing Kernel uses less points.
                
                //Adds the pressure values to right-hand side for this cell 
                varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm); //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;

                varfloat R = PressureField[idxCenter];
                R -= bxp ? 0 : SP->cx * PressureField[idx_xp] / w_in_1; //done this way to prevent access outside allocated memory 
                R -= bxm ? 0 : SP->cx * PressureField[idx_xm] / w_in_1;
                R -= byp ? 0 : SP->cy * PressureField[idx_yp] / w_in_1;
                R -= bym ? 0 : SP->cy * PressureField[idx_ym] / w_in_1;
                R -= bzp ? 0 : SP->cz * PressureField[idx_zp] / w_in_1;
                R -= bzm ? 0 : SP->cz * PressureField[idx_zm] / w_in_1;
                Result[idxCenter] = R;
            }
            else if (SP->Kernel == CELL_CENTERED){
                //Cell centered Kernel 
                long long idx_xpyp = idxCenter + GridX + 1; //xy edges
                long long idx_xpym = idxCenter - GridX + 1;
                long long idx_xmyp = idxCenter + GridX - 1;
                long long idx_xmym = idxCenter - GridX - 1;
                
                long long idx_xpzp = idxCenter + GridX * GridY + 1; //xz edges
                long long idx_xpzm = idxCenter - GridX * GridY + 1;
                long long idx_xmzp = idxCenter + GridX * GridY - 1;
                long long idx_xmzm = idxCenter - GridX * GridY - 1;

                long long idx_ypzp = idxCenter + GridX * GridY + GridX; //yz edges
                long long idx_ypzm = idxCenter - GridX * GridY + GridX;
                long long idx_ymzp = idxCenter + GridX * GridY - GridX;
                long long idx_ymzm = idxCenter - GridX * GridY - GridX;

                long long idx_xpypzp = idxCenter + 1 + GridX + GridX * GridY; //x+ corners
                long long idx_xpypzm = idxCenter + 1 + GridX - GridX * GridY;
                long long idx_xpymzp = idxCenter + 1 - GridX + GridX * GridY;
                long long idx_xpymzm = idxCenter + 1 - GridX - GridX * GridY;

                long long idx_xmypzp = idxCenter - 1 + GridX + GridX * GridY; //x- corners
                long long idx_xmypzm = idxCenter - 1 + GridX - GridX * GridY;
                long long idx_xmymzp = idxCenter - 1 - GridX + GridX * GridY;
                long long idx_xmymzm = idxCenter - 1 - GridX - GridX * GridY;
            
                //Computes boolean values for each index
                varfloat bxpyp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || (RHS[idx_xpyp] != RHS[idx_xpyp]);  //xy edges
                varfloat bxpym = ((xx + 1) >= GridX) || ((yy - 1) < 0) || (RHS[idx_xpym] != RHS[idx_xpym]);
                varfloat bxmyp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || (RHS[idx_xmyp] != RHS[idx_xmyp]);
                varfloat bxmym = ((xx - 1) < 0) || ((yy - 1) < 0) || (RHS[idx_xmym] != RHS[idx_xmym]);
            
                varfloat bxpzp = ((xx + 1) >= GridX) || ((zz + 1) >= GridZ) || (RHS[idx_xpzp] != RHS[idx_xpzp]);  //xz edges
                varfloat bxpzm = ((xx + 1) >= GridX) || ((zz - 1) < 0) || (RHS[idx_xpzm] != RHS[idx_xpzm]);
                varfloat bxmzp = ((xx - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_xmzp] != RHS[idx_xmzp]);
                varfloat bxmzm = ((xx - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_xmzm] != RHS[idx_xmzm]);
            
                varfloat bypzp = ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (RHS[idx_ypzp] != RHS[idx_ypzp]);  //yz edges
                varfloat bypzm = ((yy + 1) >= GridY) || ((zz - 1) < 0) || (RHS[idx_ypzm] != RHS[idx_ypzm]);
                varfloat bymzp = ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_ymzp] != RHS[idx_ymzp]);
                varfloat bymzm = ((yy - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_ymzm] != RHS[idx_ymzm]);

                varfloat bxpypzp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (RHS[idx_xpypzp] != RHS[idx_xpypzp]);  //x+ corners
                varfloat bxpypzm = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (RHS[idx_xpypzm] != RHS[idx_xpypzm]); 
                varfloat bxpymzp = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_xpymzp] != RHS[idx_xpymzp]); 
                varfloat bxpymzm = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_xpymzm] != RHS[idx_xpymzm]); 

                varfloat bxmypzp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (RHS[idx_xmypzp] != RHS[idx_xmypzp]);  //x- corners
                varfloat bxmypzm = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (RHS[idx_xmypzm] != RHS[idx_xmypzm]); 
                varfloat bxmymzp = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_xmymzp] != RHS[idx_xmymzp]); 
                varfloat bxmymzm = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_xmymzm] != RHS[idx_xmymzm]); 

                //Adds the pressure values to right-hand side for this cell 
                varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm) + 
                                SP->cxy * (bxpyp + bxpym + bxmyp + bxmym) + SP->cxz * (bxpzp + bxpzm + bxmzp + bxmzm) + 
                                SP->cyz * (bypzp + bypzm + bymzp + bymzm) + SP->cxyz * (bxpypzp + bxpypzm + bxpymzp + bxpymzm + bxmypzp + bxmypzm + bxmymzp + bxmymzm); 
                                //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;
                
                //Computes the pressure
                varfloat R = PressureField[idxCenter];
                R -= bxp ? 0 : SP->cx * PressureField[idx_xp] / w_in_1; //Faces
                R -= bxm ? 0 : SP->cx * PressureField[idx_xm] / w_in_1;
                R -= byp ? 0 : SP->cy * PressureField[idx_yp] / w_in_1;
                R -= bym ? 0 : SP->cy * PressureField[idx_ym] / w_in_1;
                R -= bzp ? 0 : SP->cz * PressureField[idx_zp] / w_in_1;
                R -= bzm ? 0 : SP->cz * PressureField[idx_zm] / w_in_1;
                
                R -= bxpyp ? 0 : SP->cxy * PressureField[idx_xpyp] / w_in_1; //Edges xy
                R -= bxpym ? 0 : SP->cxy * PressureField[idx_xpym] / w_in_1;
                R -= bxmyp ? 0 : SP->cxy * PressureField[idx_xmyp] / w_in_1;
                R -= bxmym ? 0 : SP->cxy * PressureField[idx_xmym] / w_in_1;
                
                R -= bxpzp ? 0 : SP->cxz * PressureField[idx_xpzp] / w_in_1; //Edges xz
                R -= bxpzm ? 0 : SP->cxz * PressureField[idx_xpzm] / w_in_1;
                R -= bxmzp ? 0 : SP->cxz * PressureField[idx_xmzp] / w_in_1;
                R -= bxmzm ? 0 : SP->cxz * PressureField[idx_xmzm] / w_in_1;
                
                R -= bypzp ? 0 : SP->cyz * PressureField[idx_ypzp] / w_in_1; //Edges yz
                R -= bypzm ? 0 : SP->cyz * PressureField[idx_ypzm] / w_in_1;
                R -= bymzp ? 0 : SP->cyz * PressureField[idx_ymzp] / w_in_1;
                R -= bymzm ? 0 : SP->cyz * PressureField[idx_ymzm] / w_in_1;
                
                R -= bxpypzp ? 0 : SP->cxyz * PressureField[idx_xpypzp] / w_in_1; //Corners x+
                R -= bxpypzm ? 0 : SP->cxyz * PressureField[idx_xpypzm] / w_in_1;
                R -= bxpymzp ? 0 : SP->cxyz * PressureField[idx_xpymzp] / w_in_1;
                R -= bxpymzm ? 0 : SP->cxyz * PressureField[idx_xpymzm] / w_in_1;
                
                R -= bxmypzp ? 0 : SP->cxyz * PressureField[idx_xmypzp] / w_in_1; //Corners x-
                R -= bxmypzm ? 0 : SP->cxyz * PressureField[idx_xmypzm] / w_in_1;
                R -= bxmymzp ? 0 : SP->cxyz * PressureField[idx_xmymzp] / w_in_1;
                R -= bxmymzm ? 0 : SP->cxyz * PressureField[idx_xmymzm] / w_in_1;
                Result[idxCenter] = R;
            }
        }

    }

}

template <typename varfloat>
__global__ void UpdateRHS_Vector_GPU(varfloat* PressureField, varfloat* RHS, varfloat* SourceX, varfloat* SourceY, varfloat* SourceZ, SolverParameters<varfloat>* SP) {
    //Computes the right-hand side vector based on the values of the pressures for all cells, considering boundaries, etc.
    long long xx = blockIdx.x * blockDim.x + threadIdx.x;
    long long yy = blockIdx.y * blockDim.y + threadIdx.y;
    long long zz = blockIdx.z * blockDim.z + threadIdx.z;

    if ((xx >= SP->BoxGridPoints.x) || (yy >= SP->BoxGridPoints.y) || (zz >= SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }

    if (SP->BoxGridPoints.z == 1) {
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long idxCenter = xx + GridX * (yy + GridY * zz);

        //dx and dy for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;

        if (SourceX[idxCenter] != SourceX[idxCenter]) {
            //The source value here is a nan, so simply makes the RHS at this point a nan as well
            RHS[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_xm = idxCenter - 1;
            long long idx_ym = idxCenter - GridX;

            if (SP->Kernel == FACE_CROSSING){
                //Face crossing Kernel uses less points.

                //Computes the boolean values for each index
                varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]);
                varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
                varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
                varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);

                //Computes the weights for the [n] coefficients
                varfloat rhs_cx = SP->cx;
                varfloat rhs_cy = SP->cy;

                varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym); //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;

                //Adds the pressure values to right-hand side for this cell
                varfloat R = 0.0;
                R += bxp ? 0.0 : (- rhs_cx * (SourceX[idx_xp] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                R += bxm ? 0.0 : (rhs_cx * (SourceX[idx_xm] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                R += byp ? 0.0 : (- rhs_cy * (SourceY[idx_yp] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                R += bym ? 0.0 : (rhs_cy * (SourceY[idx_ym] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                RHS[idxCenter] = R;
            }
            else if(SP->Kernel == CELL_CENTERED){
                //Cell centered Kernel 
                long long idx_xpyp = idxCenter + GridX + 1;
                long long idx_xmyp = idxCenter + GridX - 1;
                long long idx_xpym = idxCenter - GridX + 1;
                long long idx_xmym = idxCenter - GridX - 1;

                varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
                varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
                varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);
                
                varfloat bxpyp = (((xx + 1) >= GridX) || ((yy + 1) >= GridY)) || (SourceX[idx_xpyp] != SourceX[idx_xpyp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                varfloat bxmyp = (((yy + 1) >= GridY) || ((xx - 1) < 0)) || (SourceX[idx_xmyp] != SourceX[idx_xmyp]);
                varfloat bxpym = (((yy - 1) < 0) || ((xx + 1) >= GridX)) || (SourceX[idx_xpym] != SourceX[idx_xpym]);
                varfloat bxmym = (((yy - 1) < 0) || ((xx - 1) < 0)) || (SourceX[idx_xmym] != SourceX[idx_xmym]);

                varfloat rhs_cx = SP->cx;
                varfloat rhs_cy = SP->cy;
                varfloat rhs_cxy = SP->cxy;

                varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym) + rhs_cxy * (bxpyp + bxmyp + bxpym + bxmym); //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;

                //Adds the pressure values to right-hand side for this cell
                varfloat R = 0.0;
                R += bxp ? 0.0 : (- rhs_cx * (SourceX[idx_xp] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                R += bxm ? 0.0 : (rhs_cx * (SourceX[idx_xm] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                R += byp ? 0.0 : (- rhs_cy * (SourceY[idx_yp] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                R += bym ? 0.0 : (rhs_cy * (SourceY[idx_ym] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                
                R += bxpyp ? 0.0 : (- rhs_cxy * (+(SourceX[idx_xpyp] + SourceX[idxCenter]) * (GridDX / 2.0) + (SourceY[idx_xpyp] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                R += bxmyp ? 0.0 : (- rhs_cxy * (-(SourceX[idx_xmyp] + SourceX[idxCenter]) * (GridDX / 2.0) + (SourceY[idx_xmyp] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                R += bxpym ? 0.0 : (- rhs_cxy * (+(SourceX[idx_xpym] + SourceX[idxCenter]) * (GridDX / 2.0) - (SourceY[idx_xpym] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                R += bxmym ? 0.0 : (- rhs_cxy * (-(SourceX[idx_xmym] + SourceX[idxCenter]) * (GridDX / 2.0) - (SourceY[idx_xmym] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                RHS[idxCenter] = R;
            }
        }
    }
    else {
        //3D case
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long GridZ = (long long)SP->BoxGridPoints.z;
        long long idxCenter = xx + GridX * (yy + GridY * zz);

        //dx and dy and dz for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;
        varfloat GridDZ = SP->GridDelta.z;

        if (SourceX[idxCenter] != SourceX[idxCenter]) {
            //The source value here is a nan, so simply makes the RHS at this point a nan as well
            RHS[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_xm = idxCenter - 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_ym = idxCenter - GridX;
            long long idx_zp = idxCenter + GridX * GridY;
            long long idx_zm = idxCenter - GridX * GridY;

            //Computes the boolean values for each index
            varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]);
            varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
            varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
            varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);
            varfloat bzp = ((zz + 1) >= GridZ) || (SourceX[idx_zp] != SourceX[idx_zp]);
            varfloat bzm = ((zz - 1) < 0) || (SourceX[idx_zm] != SourceX[idx_zm]);

            if (SP->Kernel == FACE_CROSSING){
                //Face crossing Kernel uses less points.

                //Computes the weights for the [n] coefficients
                varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm); //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;

                //Adds the pressure values to right-hand side for this cell   
                varfloat R = 0.0;
                R += bxp ? 0.0 : (-SP->cx * (SourceX[idx_xp] + SourceX[idxCenter]) * GridDX / 2) / w_in_1;
                R += bxm ? 0.0 : (SP->cx * (SourceX[idx_xm] + SourceX[idxCenter]) * GridDX / 2) / w_in_1;
                R += byp ? 0.0 : (-SP->cy * (SourceY[idx_yp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                R += bym ? 0.0 : (SP->cy * (SourceY[idx_ym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                R += bzp ? 0.0 : (-SP->cz * (SourceZ[idx_zp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;
                R += bzm ? 0.0 : (SP->cz * (SourceZ[idx_zm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;
                RHS[idxCenter] = R;
            }
            else if (SP->Kernel == CELL_CENTERED){
                //Cell centered Kernel 
                long long idx_xpyp = idxCenter + GridX + 1; //xy edges
                long long idx_xpym = idxCenter - GridX + 1;
                long long idx_xmyp = idxCenter + GridX - 1;
                long long idx_xmym = idxCenter - GridX - 1;
                
                long long idx_xpzp = idxCenter + GridX * GridY + 1; //xz edges
                long long idx_xpzm = idxCenter - GridX * GridY + 1;
                long long idx_xmzp = idxCenter + GridX * GridY - 1;
                long long idx_xmzm = idxCenter - GridX * GridY - 1;

                long long idx_ypzp = idxCenter + GridX * GridY + GridX; //yz edges
                long long idx_ypzm = idxCenter - GridX * GridY + GridX;
                long long idx_ymzp = idxCenter + GridX * GridY - GridX;
                long long idx_ymzm = idxCenter - GridX * GridY - GridX;

                long long idx_xpypzp = idxCenter + 1 + GridX + GridX * GridY; //x+ corners
                long long idx_xpypzm = idxCenter + 1 + GridX - GridX * GridY;
                long long idx_xpymzp = idxCenter + 1 - GridX + GridX * GridY;
                long long idx_xpymzm = idxCenter + 1 - GridX - GridX * GridY;

                long long idx_xmypzp = idxCenter - 1 + GridX + GridX * GridY; //x- corners
                long long idx_xmypzm = idxCenter - 1 + GridX - GridX * GridY;
                long long idx_xmymzp = idxCenter - 1 - GridX + GridX * GridY;
                long long idx_xmymzm = idxCenter - 1 - GridX - GridX * GridY;
            
                //Computes boolean values for each index
                varfloat bxpyp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || (SourceX[idx_xpyp] != SourceX[idx_xpyp]);  //xy edges
                varfloat bxpym = ((xx + 1) >= GridX) || ((yy - 1) < 0) || (SourceX[idx_xpym] != SourceX[idx_xpym]);
                varfloat bxmyp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || (SourceX[idx_xmyp] != SourceX[idx_xmyp]);
                varfloat bxmym = ((xx - 1) < 0) || ((yy - 1) < 0) || (SourceX[idx_xmym] != SourceX[idx_xmym]);
            
                varfloat bxpzp = ((xx + 1) >= GridX) || ((zz + 1) >= GridZ) || (SourceX[idx_xpzp] != SourceX[idx_xpzp]);  //xz edges
                varfloat bxpzm = ((xx + 1) >= GridX) || ((zz - 1) < 0) || (SourceX[idx_xpzm] != SourceX[idx_xpzm]);
                varfloat bxmzp = ((xx - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_xmzp] != SourceX[idx_xmzp]);
                varfloat bxmzm = ((xx - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_xmzm] != SourceX[idx_xmzm]);
            
                varfloat bypzp = ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (SourceX[idx_ypzp] != SourceX[idx_ypzp]);  //yz edges
                varfloat bypzm = ((yy + 1) >= GridY) || ((zz - 1) < 0) || (SourceX[idx_ypzm] != SourceX[idx_ypzm]);
                varfloat bymzp = ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_ymzp] != SourceX[idx_ymzp]);
                varfloat bymzm = ((yy - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_ymzm] != SourceX[idx_ymzm]);

                varfloat bxpypzp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (SourceX[idx_xpypzp] != SourceX[idx_xpypzp]);  //x+ corners
                varfloat bxpypzm = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (SourceX[idx_xpypzm] != SourceX[idx_xpypzm]); 
                varfloat bxpymzp = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_xpymzp] != SourceX[idx_xpymzp]); 
                varfloat bxpymzm = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_xpymzm] != SourceX[idx_xpymzm]); 

                varfloat bxmypzp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (SourceX[idx_xmypzp] != SourceX[idx_xmypzp]);  //x- corners
                varfloat bxmypzm = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (SourceX[idx_xmypzm] != SourceX[idx_xmypzm]); 
                varfloat bxmymzp = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_xmymzp] != SourceX[idx_xmymzp]); 
                varfloat bxmymzm = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_xmymzm] != SourceX[idx_xmymzm]); 

                //Adds the pressure values to right-hand side for this cell 
                varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm) + 
                                SP->cxy * (bxpyp + bxpym + bxmyp + bxmym) + SP->cxz * (bxpzp + bxpzm + bxmzp + bxmzm) + 
                                SP->cyz * (bypzp + bypzm + bymzp + bymzm) + SP->cxyz * (bxpypzp + bxpypzm + bxpymzp + bxpymzm + bxmypzp + bxmypzm + bxmymzp + bxmymzm); 
                                //Weight for the center coefficient
                varfloat w_in_1 = 1.0 - w_in;

                //Adds the pressure values to right-hand side for this cell   
                varfloat R = 0.0;
                R -= bxp ? 0.0 : SP->cx * (+(SourceX[idx_xp] + SourceX[idxCenter]) * GridDX / 2) / w_in_1; //Faces
                R -= bxm ? 0.0 : SP->cx * (-(SourceX[idx_xm] + SourceX[idxCenter]) * GridDX / 2) / w_in_1;
                R -= byp ? 0.0 : SP->cy * (+(SourceY[idx_yp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                R -= bym ? 0.0 : SP->cy * (-(SourceY[idx_ym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                R -= bzp ? 0.0 : SP->cz * (+(SourceZ[idx_zp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;
                R -= bzm ? 0.0 : SP->cz * (-(SourceZ[idx_zm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;

                R -= bxpyp ? 0.0 : SP->cxy * (+(SourceX[idx_xpyp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xpyp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; //Edges xy
                R -= bxpym ? 0.0 : SP->cxy * (+(SourceX[idx_xpym] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xpym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; 
                R -= bxmyp ? 0.0 : SP->cxy * (-(SourceX[idx_xmyp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xmyp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; 
                R -= bxmym ? 0.0 : SP->cxy * (-(SourceX[idx_xmym] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xmym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; 

                R -= bxpzp ? 0.0 : SP->cxz * (+(SourceX[idx_xpzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceZ[idx_xpzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Edges xz
                R -= bxpzm ? 0.0 : SP->cxz * (+(SourceX[idx_xpzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceZ[idx_xpzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bxmzp ? 0.0 : SP->cxz * (-(SourceX[idx_xmzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceZ[idx_xmzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bxmzm ? 0.0 : SP->cxz * (-(SourceX[idx_xmzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceZ[idx_xmzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 

                R -= bypzp ? 0.0 : SP->cyz * (+(SourceY[idx_ypzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_ypzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Edges yz
                R -= bypzm ? 0.0 : SP->cyz * (+(SourceY[idx_ypzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_ypzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bymzp ? 0.0 : SP->cyz * (-(SourceY[idx_ymzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_ymzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bymzm ? 0.0 : SP->cyz * (-(SourceY[idx_ymzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_ymzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 

                R -= bxpypzp ? 0.0 : SP->cxyz * (+(SourceX[idx_xpypzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xpypzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xpypzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Corners x+
                R -= bxpypzm ? 0.0 : SP->cxyz * (+(SourceX[idx_xpypzm] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xpypzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xpypzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bxpymzp ? 0.0 : SP->cxyz * (+(SourceX[idx_xpymzp] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xpymzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xpymzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bxpymzm ? 0.0 : SP->cxyz * (+(SourceX[idx_xpymzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xpymzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xpymzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 

                R -= bxmypzp ? 0.0 : SP->cxyz * (-(SourceX[idx_xmypzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xmypzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xmypzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Corners x-
                R -= bxmypzm ? 0.0 : SP->cxyz * (-(SourceX[idx_xmypzm] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xmypzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xmypzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bxmymzp ? 0.0 : SP->cxyz * (-(SourceX[idx_xmymzp] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xmymzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xmymzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                R -= bxmymzm ? 0.0 : SP->cxyz * (-(SourceX[idx_xmymzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xmymzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xmymzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                RHS[idxCenter] = R;
            }
        }
    }
}

template <typename varfloat>
void ConjugateGradientSolver_GPU(varfloat* PressureField, varfloat* RHS, SolverParameters<varfloat> SolverConfig, BoxContents<varfloat> VTK_Contents) {
    // Allocate GPU memory for source field and pressure field
    cudaFree(0); //Initializes GPU context
    
    //Creates concurrent streams so processing can occur in parallel
    const int nStreams = 4;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    //Memory allocation
    long long boxArraySize = sizeof(varfloat) * VTK_Contents.totalBoxElements;
    varfloat* d_PressureField; varfloat* d_RHS; 
    varfloat* d_rk; varfloat* d_rkp1; varfloat* d_pk; varfloat* d_temp;
    cudaMalloc(&d_PressureField, boxArraySize); cudaMalloc(&d_RHS, boxArraySize);
    cudaMalloc(&d_rk, boxArraySize); cudaMalloc(&d_rkp1, boxArraySize);
    cudaMalloc(&d_pk, boxArraySize); cudaMalloc(&d_temp, boxArraySize);

    cudaMemcpyAsync(d_PressureField, PressureField, boxArraySize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_RHS, RHS, boxArraySize, cudaMemcpyHostToDevice, stream[1]);

    //Allocate GPU memory for the source terms
    varfloat* d_SourceX; varfloat* d_SourceY; varfloat* d_SourceZ;
    cudaMalloc(&d_SourceX, boxArraySize); cudaMalloc(&d_SourceY, boxArraySize); cudaMalloc(&d_SourceZ, boxArraySize);
    cudaMemcpyAsync(d_SourceX, VTK_Contents.SourceFn_Field_X, boxArraySize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_SourceY, VTK_Contents.SourceFn_Field_Y, boxArraySize, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(d_SourceZ, VTK_Contents.SourceFn_Field_Z, boxArraySize, cudaMemcpyHostToDevice, stream[2]);

    //Allocates auxiliary variables
    SolverParameters<varfloat>* d_SolverConfig;
    cudaMalloc(&d_SolverConfig, sizeof(SolverParameters<varfloat>));
    cudaMemcpyAsync(d_SolverConfig, &SolverConfig, sizeof(SolverParameters<varfloat>), cudaMemcpyHostToDevice, stream[1]);

    //Allocates scalars
    varfloat* d_beta; varfloat* d_alpha; varfloat* d_r_norm; varfloat* d_r_norm_old; varfloat* d_temp_scal; 
    cudaMalloc((void**)&d_beta, sizeof(varfloat)); cudaMalloc((void**)&d_alpha, sizeof(varfloat)); cudaMalloc((void**)&d_r_norm, sizeof(varfloat));
    cudaMalloc((void**)&d_r_norm_old, sizeof(varfloat)); cudaMalloc((void**)&d_temp_scal, sizeof(varfloat));

    varfloat* d_sum; varfloat* d_numel;
    cudaMalloc((void**)&d_sum, sizeof(varfloat)); cudaMalloc((void**)&d_numel, sizeof(varfloat));

    dim3 threadsPerBlock3D; dim3 numBlocks3D; //3D for the matrix kernel
    if (VTK_Contents.BoxGridSize.z == 1) {
        threadsPerBlock3D = dim3(16, 16, 1);
        numBlocks3D = dim3(ceil(VTK_Contents.BoxGridSize.x / (varfloat)threadsPerBlock3D.x), ceil(VTK_Contents.BoxGridSize.y / (varfloat)threadsPerBlock3D.y), 1);
    }
    else {
        threadsPerBlock3D = dim3(4, 8, 8);
        numBlocks3D = dim3(ceil(VTK_Contents.BoxGridSize.x / (varfloat)threadsPerBlock3D.x), ceil(VTK_Contents.BoxGridSize.y / (varfloat)threadsPerBlock3D.y), ceil(VTK_Contents.BoxGridSize.z / (varfloat)threadsPerBlock3D.z));
    }
    dim3 threadsPerBlock1D = dim3(BLOCKDIM_VEC, 1, 1); //1D for the vector kernels
    dim3 numBlocks1D = dim3(ceil(VTK_Contents.totalBoxElements / (varfloat)threadsPerBlock1D.x), 1, 1);

    ClockTic();
    double IdleClock = 0.0; // to keep track of the time so every 10 seconds it still prints something


    bool canExit = false;
    
    while (!canExit){
        //will only exit the loop if the CG hasn't diverged
        //=====Updates RHS terms=====
        cudaDeviceSynchronize();
        UpdateRHS_Vector_GPU << <numBlocks3D, threadsPerBlock3D, 0, stream[0] >> > (d_PressureField, d_RHS, d_SourceX, d_SourceY, d_SourceZ, d_SolverConfig); //b

        //=====Starts CG solver computations=====
        cudaDeviceSynchronize();
        MatrixMul_Omnidirectional_GPU << <numBlocks3D, threadsPerBlock3D, 0, stream[0] >> > (d_temp, d_PressureField, d_RHS, d_SolverConfig); //temp=A*x_0
        cudaDeviceSynchronize();
        subtractVectors_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_RHS, d_temp, d_rk, d_SolverConfig); //r_0=b-A*x_0

        cudaDeviceSynchronize();
        cudaMemcpyAsync(d_pk, d_rk, boxArraySize, cudaMemcpyDeviceToDevice, stream[0]); //p_0=r_0
        vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_rk, d_rk, d_r_norm_old, d_SolverConfig); //r_k dot r_kvarfloat r_norm; 
        cudaDeviceSynchronize();

        varfloat r_norm_init;
        Progress P_cgs;
        cudaMemcpy(&r_norm_init, d_r_norm_old, sizeof(varfloat), cudaMemcpyDeviceToHost); // initial residual norm
        cudaDeviceSynchronize();
        r_norm_init = sqrt(r_norm_init);
        varfloat r_norm_min = 1e9; //To restart the CG if divergence is detected

        if (SolverConfig.Verbose){ mexPrintf("Initial Residual Norm=%f\n", r_norm_init);}
        CGS_Progress.clear();
        P_cgs.Iteration = 0; P_cgs.Residual = 1.0f;  P_cgs.TimeSeconds = ClockToc(); CGS_Progress.push_back(P_cgs);

        for (int cgs_iter = 0; cgs_iter < VTK_Contents.totalBoxElements; cgs_iter++) {
            //Iterations of the Conjugate Gradient Solver here
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_rk, d_rk, d_r_norm_old, d_SolverConfig); //r_k dot r_k
            MatrixMul_Omnidirectional_GPU << <numBlocks3D, threadsPerBlock3D, 0, stream[1] >> > (d_temp, d_pk, d_RHS, d_SolverConfig); //temp=A*p_k
            cudaDeviceSynchronize();
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_pk, d_temp, d_temp_scal, d_SolverConfig); //temp_scal = p_k dot temp
            cudaDeviceSynchronize();
            divide << <1, 1, 0, stream[0] >> > (d_r_norm_old, d_temp_scal, d_alpha);//alpha = (rk dot rk) / (pk dot A*pk)
            cudaDeviceSynchronize(); 

            //Implicit residual update
            scalarVectorMult_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_alpha, d_temp, d_temp, d_SolverConfig); //temp=alphak*temp
            subtractVectors_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[2] >> > (d_rk, d_temp, d_rkp1, d_SolverConfig); //r_k+1=rk-temp (i.e. rk-A*temp)

            cudaDeviceSynchronize();
            scalarVectorMult_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_alpha, d_pk, d_temp, d_SolverConfig); //temp = alphak*pk
            cudaDeviceSynchronize();
            addVectors_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_PressureField, d_temp, d_PressureField, d_SolverConfig); //xk+1=xk+alphak*pk
            cudaDeviceSynchronize();
            //printVector1_GPU << <1, 1 >> > (d_PressureField); cudaDeviceSynchronize();
            //printVector2_GPU << <1, 1 >> > (d_PressureField);

            //Explicit residual update
                //MatrixMul_Omnidirectional_GPU << <numBlocks3D, threadsPerBlock3D >> > (d_temp, d_PressureField, d_RHS, d_SolverConfig); //temp=A*x_k+1
                //cudaDeviceSynchronize();
                //subtractVectors_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_RHS, d_temp, d_rkp1, d_SolverConfig); //r_k+1=b-A*xk+1
                //cudaDeviceSynchronize();

            cudaMemcpyAsync(d_rk, d_rkp1, boxArraySize, cudaMemcpyDeviceToDevice, stream[0]); //rk=rk+1
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_rkp1, d_rkp1, d_r_norm, d_SolverConfig); //r_k+1 dot r_k+1
            cudaDeviceSynchronize();

            varfloat r_norm; cudaMemcpy(&r_norm, d_r_norm, sizeof(varfloat), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            r_norm = sqrt(r_norm);
            
            //Updates the lowest rnorm
            if (r_norm_min > r_norm){
                r_norm_min = r_norm;
            }

            //Handles output to the user
            if (cgs_iter % 10 == 0) {
                if (SolverConfig.Verbose){ mexPrintf("CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e; \n", cgs_iter, r_norm / r_norm_init, r_norm);fflush(stdout);}
            }        
            if (!SolverConfig.Verbose){
                //Even if not verbose, still prints every 10 seconds
                if (ClockToc() > (IdleClock + 10.0)) {
                    mexPrintf("CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e; \n", cgs_iter, r_norm / r_norm_init, r_norm);
                    fflush(stdout);
                    IdleClock += 10.0;
                }
            }

            //Stores iteration info on memory
            P_cgs.Iteration = cgs_iter+1; P_cgs.Residual = r_norm / r_norm_init; P_cgs.TimeSeconds = ClockToc(); CGS_Progress.push_back(P_cgs);

            if ((r_norm / r_norm_init > SolverConfig.solverToleranceRel) && (r_norm > SolverConfig.solverToleranceAbs)) {
                //Only continues if not yet within tolerance
                divide << <1, 1 >> > (d_r_norm, d_r_norm_old, d_beta);//beta = (rk+1 dot rk+1) / (rk dot rk)
                cudaDeviceSynchronize(); 
                scalarVectorMult_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_beta, d_pk, d_temp, d_SolverConfig); //temp=beta*pk
                cudaDeviceSynchronize();
                addVectors_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_temp, d_rkp1, d_pk, d_SolverConfig); //pk+1=rk+1 + beta*pk 
                cudaDeviceSynchronize();
            }
            else {
                if (SolverConfig.Verbose){ mexPrintf("CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e [Converged]\n", cgs_iter, r_norm / r_norm_init, r_norm);} 

                if (isnan(r_norm)) {
                    mexPrintf("======== Result was NAN! ========\n");
                    mexPrintf("Make sure your coordinate system is correct. This code expects a coordinate in the 'ND grid' format, i.e., dimension order is (x, y) or (x, y, z). DO NOT USE the 'MESHGRID' format to build the arrays in Matlab!\n");
                }

                canExit = true;
                break;
            }

            if (r_norm > (1e2 * r_norm_min)) { //factor of 1e2 arbitrary
                //CG is diverging, restarts
                //GPU_FillNan << <numBlocks3D, threadsPerBlock3D, 0, stream[0] >> > (d_PressureField, d_SolverConfig);
                //cudaDeviceSynchronize();
                SolverConfig.solverToleranceAbs = r_norm_min * 1.001; //Sets the new tolerance to a slightly higher value than the best achieved so we can exit next time
                mexPrintf("==========CG Diverged! Restarting with AbsTol = %0.2e=========.\=\n", SolverConfig.solverToleranceAbs);
                break; // restarts the whole CG
            }        
        }
    }

    //Extracts 3D array from GPU Memory
    //cudaMemcpy(PressureField, d_RHS, boxArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(PressureField, d_PressureField, boxArraySize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //Frees memory 
    if (SolverConfig.Verbose){ mexPrintf("==========================================================\n");}
    cudaFree(d_PressureField); cudaFree(d_RHS); cudaFree(d_rk); cudaFree(d_rkp1); cudaFree(d_pk); cudaFree(d_temp);
    cudaFree(d_SourceX); cudaFree(d_SourceY); cudaFree(d_SourceZ);
    cudaFree(d_SolverConfig);
    cudaFree(d_beta); cudaFree(d_alpha); cudaFree(d_r_norm); cudaFree(d_r_norm_old); cudaFree(d_temp_scal);
    cudaFree(d_sum); cudaFree(d_numel);

    //Destroy parallel streams
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

#pragma endregion

//========CPU OpenMP Functions======
#pragma region
template <typename varfloat>
 void scalarVectorMult_CPU(varfloat* scalar, varfloat* a, varfloat* out, SolverParameters<varfloat>* SP) {
    #pragma omp parallel for
    for (long long i = 0; i < SP->totalBoxElements; i++) {
        out[i] = a[i]* *scalar;
    }
}

template <typename varfloat>
void vectorDot_CPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    //Performs dot product    
    varfloat sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < SP->totalBoxElements; i++) {
        if ((a[i] == a[i]) && (b[i] == b[i])) {
            sum += a[i] * b[i];
            //printf("i=%lld; a=%f, b=%f, tmp=%f, \n", i, a[i], b[i],tmp);
        }
    }
    *out = sum;
}

template <typename varfloat>
void addVectors_CPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    #pragma omp parallel for
    for (long long i = 0; i < SP->totalBoxElements; i++) {
        out[i] = a[i] + b[i];
    }
}

template <typename varfloat>
void subtractVectors_CPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    #pragma omp parallel for
    for (long long i = 0; i < SP->totalBoxElements; i++) {
        out[i] = a[i] - b[i];
    }
}

template <typename varfloat>
void MatrixMul_Omnidirectional_CPU(varfloat* Result, varfloat* PressureField, varfloat* RHS, SolverParameters<varfloat>* SP) {
    //This is the bit of code that performs the matrix multiplication Result=A*x (where A is the weight matrix and x is the PressureField)
    //The RHS of the equation is also provided so we can find the points where we have NAN's    

    if (SP->BoxGridPoints.z == 1) {
        //2D Case
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;

        //dx and dy for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;
        long long zz = 0;

        #pragma omp parallel for
        for (long long xx = 0; xx < SP->BoxGridPoints.x; xx++) {
            for (long long yy = 0; yy < SP->BoxGridPoints.y; yy++) {
                long long idxCenter = xx + GridX * (yy + GridY * zz);

                if ((RHS[idxCenter] != RHS[idxCenter])) {
                    //The RHS here is a nan, so simply makes the result at this point a nan as well
                    Result[idxCenter] = NAN;
                }
                else {
                    long long idx_xp = idxCenter + 1;
                    long long idx_xm = idxCenter - 1;
                    long long idx_yp = idxCenter + GridX;
                    long long idx_ym = idxCenter - GridX;

                    if (SP->Kernel == FACE_CROSSING){
                        //Face crossing Kernel uses less points.
                        varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                        varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
                        varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
                        varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);

                        varfloat rhs_cx = SP->cx;
                        varfloat rhs_cy = SP->cy;

                        //Adds the pressure values to right-hand side for this cell 
                        varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym); //Weight for the center coefficient
                        varfloat w_in_1 = 1.0 - w_in;

                        varfloat R = PressureField[idxCenter];
                        R -= bxp ? 0.0 : rhs_cx * PressureField[idx_xp] / w_in_1; //done this way to prevent access outside allocated memory 
                        R -= bxm ? 0.0 : rhs_cx * PressureField[idx_xm] / w_in_1;
                        R -= byp ? 0.0 : rhs_cy * PressureField[idx_yp] / w_in_1;
                        R -= bym ? 0.0 : rhs_cy * PressureField[idx_ym] / w_in_1;
                        Result[idxCenter] = R;
                    }
                    else if(SP->Kernel == CELL_CENTERED){
                        //Cell centered Kernel 
                        long long idx_xpyp = idxCenter + GridX + 1;
                        long long idx_xmyp = idxCenter + GridX - 1;
                        long long idx_xpym = idxCenter - GridX + 1;
                        long long idx_xmym = idxCenter - GridX - 1;

                        varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                        varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
                        varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
                        varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);
                        
                        varfloat bxpyp = (((xx + 1) >= GridX) || ((yy + 1) >= GridY)) || (RHS[idx_xpyp] != RHS[idx_xpyp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                        varfloat bxmyp = (((yy + 1) >= GridY) || ((xx - 1) < 0)) || (RHS[idx_xmyp] != RHS[idx_xmyp]);
                        varfloat bxpym = (((yy - 1) < 0) || ((xx + 1) >= GridX)) || (RHS[idx_xpym] != RHS[idx_xpym]);
                        varfloat bxmym = (((yy - 1) < 0) || ((xx - 1) < 0)) || (RHS[idx_xmym] != RHS[idx_xmym]);

                        varfloat rhs_cx = SP->cx;
                        varfloat rhs_cy = SP->cy;
                        varfloat rhs_cxy = SP->cxy;

                        //Adds the pressure values to right-hand side for this cell 
                        varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym) + rhs_cxy * (bxpyp + bxmyp + bxpym + bxmym); //Weight for the center coefficient
                        varfloat w_in_1 = 1.0 - w_in;

                        varfloat R = PressureField[idxCenter];
                        R -= bxp ? 0.0 : rhs_cx * PressureField[idx_xp] / w_in_1; //done this way to prevent access outside allocated memory 
                        R -= bxm ? 0.0 : rhs_cx * PressureField[idx_xm] / w_in_1;
                        R -= byp ? 0.0 : rhs_cy * PressureField[idx_yp] / w_in_1;
                        R -= bym ? 0.0 : rhs_cy * PressureField[idx_ym] / w_in_1;

                        R -= bxpyp ? 0.0 : rhs_cxy * PressureField[idx_xpyp] / w_in_1; //corners
                        R -= bxmyp ? 0.0 : rhs_cxy * PressureField[idx_xmyp] / w_in_1;
                        R -= bxpym ? 0.0 : rhs_cxy * PressureField[idx_xpym] / w_in_1;
                        R -= bxmym ? 0.0 : rhs_cxy * PressureField[idx_xmym] / w_in_1;
                        Result[idxCenter] = R;
                    }
                }
            }
        }
    }
    else {
        //3D Case
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long GridZ = (long long)SP->BoxGridPoints.z;

        //dx and dy and dz for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;
        varfloat GridDZ = SP->GridDelta.z;

        #pragma omp parallel for
        for (long long xx = 0; xx < SP->BoxGridPoints.x; xx++) {
            for (long long yy = 0; yy < SP->BoxGridPoints.y; yy++) {
                for (long long zz = 0; zz < SP->BoxGridPoints.z; zz++) {
                    long long idxCenter = xx + GridX * (yy + GridY * zz);

                    if (RHS[idxCenter] != RHS[idxCenter]) {
                        //The RHS here is a nan, so simply makes the result at this point a nan as well
                        Result[idxCenter] = NAN;
                    }
                    else {
                        long long idx_xp = idxCenter + 1;
                        long long idx_xm = idxCenter - 1;
                        long long idx_yp = idxCenter + GridX;
                        long long idx_ym = idxCenter - GridX;
                        long long idx_zp = idxCenter + GridX * GridY;
                        long long idx_zm = idxCenter - GridX * GridY;
                        
                        varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 80) [i.e. isnan(X) is the same as X!=X]
                        varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
                        varfloat bzp = ((zz + 1) >= GridZ) || (RHS[idx_zp] != RHS[idx_zp]);
                        varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
                        varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);
                        varfloat bzm = ((zz - 1) < 0) || (RHS[idx_zm] != RHS[idx_zm]);

                        if (SP->Kernel == FACE_CROSSING){
                            //Face crossing Kernel uses less points.

                            //Adds the pressure values to right-hand side for this cell 
                            varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm); //Weight for the center coefficient
                            varfloat w_in_1 = 1.0 - w_in;

                            varfloat R = PressureField[idxCenter];
                            R -= bxp ? 0 : SP->cx * PressureField[idx_xp] / w_in_1; //done this way to prevent access outside allocated memory 
                            R -= bxm ? 0 : SP->cx * PressureField[idx_xm] / w_in_1;
                            R -= byp ? 0 : SP->cy * PressureField[idx_yp] / w_in_1;
                            R -= bym ? 0 : SP->cy * PressureField[idx_ym] / w_in_1;
                            R -= bzp ? 0 : SP->cz * PressureField[idx_zp] / w_in_1;
                            R -= bzm ? 0 : SP->cz * PressureField[idx_zm] / w_in_1;
                            Result[idxCenter] = R;
                        }
                        else if(SP->Kernel == CELL_CENTERED){
                            //Cell centered Kernel 
                            long long idx_xpyp = idxCenter + GridX + 1; //xy edges
                            long long idx_xpym = idxCenter - GridX + 1;
                            long long idx_xmyp = idxCenter + GridX - 1;
                            long long idx_xmym = idxCenter - GridX - 1;
                            
                            long long idx_xpzp = idxCenter + GridX * GridY + 1; //xz edges
                            long long idx_xpzm = idxCenter - GridX * GridY + 1;
                            long long idx_xmzp = idxCenter + GridX * GridY - 1;
                            long long idx_xmzm = idxCenter - GridX * GridY - 1;

                            long long idx_ypzp = idxCenter + GridX * GridY + GridX; //yz edges
                            long long idx_ypzm = idxCenter - GridX * GridY + GridX;
                            long long idx_ymzp = idxCenter + GridX * GridY - GridX;
                            long long idx_ymzm = idxCenter - GridX * GridY - GridX;

                            long long idx_xpypzp = idxCenter + 1 + GridX + GridX * GridY; //x+ corners
                            long long idx_xpypzm = idxCenter + 1 + GridX - GridX * GridY;
                            long long idx_xpymzp = idxCenter + 1 - GridX + GridX * GridY;
                            long long idx_xpymzm = idxCenter + 1 - GridX - GridX * GridY;

                            long long idx_xmypzp = idxCenter - 1 + GridX + GridX * GridY; //x- corners
                            long long idx_xmypzm = idxCenter - 1 + GridX - GridX * GridY;
                            long long idx_xmymzp = idxCenter - 1 - GridX + GridX * GridY;
                            long long idx_xmymzm = idxCenter - 1 - GridX - GridX * GridY;
                        
                            //Computes boolean values for each index
                            varfloat bxpyp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || (RHS[idx_xpyp] != RHS[idx_xpyp]);  //xy edges
                            varfloat bxpym = ((xx + 1) >= GridX) || ((yy - 1) < 0) || (RHS[idx_xpym] != RHS[idx_xpym]);
                            varfloat bxmyp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || (RHS[idx_xmyp] != RHS[idx_xmyp]);
                            varfloat bxmym = ((xx - 1) < 0) || ((yy - 1) < 0) || (RHS[idx_xmym] != RHS[idx_xmym]);
                        
                            varfloat bxpzp = ((xx + 1) >= GridX) || ((zz + 1) >= GridZ) || (RHS[idx_xpzp] != RHS[idx_xpzp]);  //xz edges
                            varfloat bxpzm = ((xx + 1) >= GridX) || ((zz - 1) < 0) || (RHS[idx_xpzm] != RHS[idx_xpzm]);
                            varfloat bxmzp = ((xx - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_xmzp] != RHS[idx_xmzp]);
                            varfloat bxmzm = ((xx - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_xmzm] != RHS[idx_xmzm]);
                        
                            varfloat bypzp = ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (RHS[idx_ypzp] != RHS[idx_ypzp]);  //yz edges
                            varfloat bypzm = ((yy + 1) >= GridY) || ((zz - 1) < 0) || (RHS[idx_ypzm] != RHS[idx_ypzm]);
                            varfloat bymzp = ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_ymzp] != RHS[idx_ymzp]);
                            varfloat bymzm = ((yy - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_ymzm] != RHS[idx_ymzm]);

                            varfloat bxpypzp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (RHS[idx_xpypzp] != RHS[idx_xpypzp]);  //x+ corners
                            varfloat bxpypzm = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (RHS[idx_xpypzm] != RHS[idx_xpypzm]); 
                            varfloat bxpymzp = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_xpymzp] != RHS[idx_xpymzp]); 
                            varfloat bxpymzm = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_xpymzm] != RHS[idx_xpymzm]); 

                            varfloat bxmypzp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (RHS[idx_xmypzp] != RHS[idx_xmypzp]);  //x- corners
                            varfloat bxmypzm = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (RHS[idx_xmypzm] != RHS[idx_xmypzm]); 
                            varfloat bxmymzp = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (RHS[idx_xmymzp] != RHS[idx_xmymzp]); 
                            varfloat bxmymzm = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz - 1) < 0) || (RHS[idx_xmymzm] != RHS[idx_xmymzm]); 

                            //Adds the pressure values to right-hand side for this cell 
                            varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm) + 
                                            SP->cxy * (bxpyp + bxpym + bxmyp + bxmym) + SP->cxz * (bxpzp + bxpzm + bxmzp + bxmzm) + 
                                            SP->cyz * (bypzp + bypzm + bymzp + bymzm) + SP->cxyz * (bxpypzp + bxpypzm + bxpymzp + bxpymzm + bxmypzp + bxmypzm + bxmymzp + bxmymzm); 
                                            //Weight for the center coefficient
                            varfloat w_in_1 = 1.0 - w_in;
                            
                            //Computes the pressure
                            varfloat R = PressureField[idxCenter];
                            R -= bxp ? 0 : SP->cx * PressureField[idx_xp] / w_in_1; //Faces
                            R -= bxm ? 0 : SP->cx * PressureField[idx_xm] / w_in_1;
                            R -= byp ? 0 : SP->cy * PressureField[idx_yp] / w_in_1;
                            R -= bym ? 0 : SP->cy * PressureField[idx_ym] / w_in_1;
                            R -= bzp ? 0 : SP->cz * PressureField[idx_zp] / w_in_1;
                            R -= bzm ? 0 : SP->cz * PressureField[idx_zm] / w_in_1;
                            
                            R -= bxpyp ? 0 : SP->cxy * PressureField[idx_xpyp] / w_in_1; //Edges xy
                            R -= bxpym ? 0 : SP->cxy * PressureField[idx_xpym] / w_in_1;
                            R -= bxmyp ? 0 : SP->cxy * PressureField[idx_xmyp] / w_in_1;
                            R -= bxmym ? 0 : SP->cxy * PressureField[idx_xmym] / w_in_1;
                            
                            R -= bxpzp ? 0 : SP->cxz * PressureField[idx_xpzp] / w_in_1; //Edges xz
                            R -= bxpzm ? 0 : SP->cxz * PressureField[idx_xpzm] / w_in_1;
                            R -= bxmzp ? 0 : SP->cxz * PressureField[idx_xmzp] / w_in_1;
                            R -= bxmzm ? 0 : SP->cxz * PressureField[idx_xmzm] / w_in_1;
                            
                            R -= bypzp ? 0 : SP->cyz * PressureField[idx_ypzp] / w_in_1; //Edges yz
                            R -= bypzm ? 0 : SP->cyz * PressureField[idx_ypzm] / w_in_1;
                            R -= bymzp ? 0 : SP->cyz * PressureField[idx_ymzp] / w_in_1;
                            R -= bymzm ? 0 : SP->cyz * PressureField[idx_ymzm] / w_in_1;
                            
                            R -= bxpypzp ? 0 : SP->cxyz * PressureField[idx_xpypzp] / w_in_1; //Corners x+
                            R -= bxpypzm ? 0 : SP->cxyz * PressureField[idx_xpypzm] / w_in_1;
                            R -= bxpymzp ? 0 : SP->cxyz * PressureField[idx_xpymzp] / w_in_1;
                            R -= bxpymzm ? 0 : SP->cxyz * PressureField[idx_xpymzm] / w_in_1;
                            
                            R -= bxmypzp ? 0 : SP->cxyz * PressureField[idx_xmypzp] / w_in_1; //Corners x-
                            R -= bxmypzm ? 0 : SP->cxyz * PressureField[idx_xmypzm] / w_in_1;
                            R -= bxmymzp ? 0 : SP->cxyz * PressureField[idx_xmymzp] / w_in_1;
                            R -= bxmymzm ? 0 : SP->cxyz * PressureField[idx_xmymzm] / w_in_1;
                            Result[idxCenter] = R;
                        }
                    }
                }
            }
        }

    }

}

template <typename varfloat>
void UpdateRHS_Vector_CPU(varfloat* PressureField, varfloat* RHS, varfloat* SourceX, varfloat* SourceY, varfloat* SourceZ, SolverParameters<varfloat>* SP) {
    //Computes the right-hand side vector based on the values of the pressures for all cells, considering boundaries, etc.
    
    if (SP->BoxGridPoints.z == 1) {
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;

        //dx and dy for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;

        #pragma omp parallel for
        for (long long xx = 0; xx < SP->BoxGridPoints.x; xx++) {
            for (long long yy = 0; yy < SP->BoxGridPoints.y; yy++) {
                long long zz = 0;
                long long idxCenter = xx + GridX * (yy + GridY * zz);

                if (SourceX[idxCenter] != SourceX[idxCenter]) {
                    //The source value here is a nan, so simply makes the RHS at this point a nan as well
                    RHS[idxCenter] = NAN;
                }
                else {
                        long long idx_xp = idxCenter + 1;
                        long long idx_yp = idxCenter + GridX;
                        long long idx_xm = idxCenter - 1;
                        long long idx_ym = idxCenter - GridX;

                        if (SP->Kernel == FACE_CROSSING){
                            //Face crossing Kernel uses less points.

                            //Computes the boolean values for each index
                            varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]);
                            varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
                            varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
                            varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);

                            //Computes the weights for the [n] coefficients
                            varfloat rhs_cx = SP->cx;
                            varfloat rhs_cy = SP->cy;

                            varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym); //Weight for the center coefficient
                            varfloat w_in_1 = 1.0 - w_in;

                            //Adds the pressure values to right-hand side for this cell
                            varfloat R = 0.0;
                            R += bxp ? 0.0 : (-rhs_cx * (SourceX[idx_xp] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                            R += bxm ? 0.0 : (rhs_cx * (SourceX[idx_xm] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                            R += byp ? 0.0 : (-rhs_cy * (SourceY[idx_yp] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                            R += bym ? 0.0 : (rhs_cy * (SourceY[idx_ym] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                            RHS[idxCenter] = R;
                        }
                        else if(SP->Kernel == CELL_CENTERED){
                            //Cell centered Kernel 
                            long long idx_xpyp = idxCenter + GridX + 1;
                            long long idx_xmyp = idxCenter + GridX - 1;
                            long long idx_xpym = idxCenter - GridX + 1;
                            long long idx_xmym = idxCenter - GridX - 1;

                            varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                            varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
                            varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
                            varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);
                            
                            varfloat bxpyp = (((xx + 1) >= GridX) || ((yy + 1) >= GridY)) || (SourceX[idx_xpyp] != SourceX[idx_xpyp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
                            varfloat bxmyp = (((yy + 1) >= GridY) || ((xx - 1) < 0)) || (SourceX[idx_xmyp] != SourceX[idx_xmyp]);
                            varfloat bxpym = (((yy - 1) < 0) || ((xx + 1) >= GridX)) || (SourceX[idx_xpym] != SourceX[idx_xpym]);
                            varfloat bxmym = (((yy - 1) < 0) || ((xx - 1) < 0)) || (SourceX[idx_xmym] != SourceX[idx_xmym]);

                            varfloat rhs_cx = SP->cx;
                            varfloat rhs_cy = SP->cy;
                            varfloat rhs_cxy = SP->cxy;

                            varfloat w_in = rhs_cx * (bxp + bxm) + rhs_cy * (byp + bym) + rhs_cxy * (bxpyp + bxmyp + bxpym + bxmym); //Weight for the center coefficient
                            varfloat w_in_1 = 1.0 - w_in;

                            //Adds the pressure values to right-hand side for this cell
                            varfloat R = 0.0;
                            R += bxp ? 0.0 : (- rhs_cx * (SourceX[idx_xp] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                            R += bxm ? 0.0 : (rhs_cx * (SourceX[idx_xm] + SourceX[idxCenter]) * (GridDX / 2.0)) / w_in_1;
                            R += byp ? 0.0 : (- rhs_cy * (SourceY[idx_yp] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                            R += bym ? 0.0 : (rhs_cy * (SourceY[idx_ym] + SourceY[idxCenter]) * (GridDY / 2.0)) / w_in_1;
                            
                            R += bxpyp ? 0.0 : (- rhs_cxy * (+(SourceX[idx_xpyp] + SourceX[idxCenter]) * (GridDX / 2.0) + (SourceY[idx_xpyp] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                            R += bxmyp ? 0.0 : (- rhs_cxy * (-(SourceX[idx_xmyp] + SourceX[idxCenter]) * (GridDX / 2.0) + (SourceY[idx_xmyp] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                            R += bxpym ? 0.0 : (- rhs_cxy * (+(SourceX[idx_xpym] + SourceX[idxCenter]) * (GridDX / 2.0) - (SourceY[idx_xpym] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                            R += bxmym ? 0.0 : (- rhs_cxy * (-(SourceX[idx_xmym] + SourceX[idxCenter]) * (GridDX / 2.0) - (SourceY[idx_xmym] + SourceY[idxCenter]) * (GridDY / 2.0))) / w_in_1;
                            
                            RHS[idxCenter] = R;
                        }
                    }
            }
        }
    }
    else {
        //3D case
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long GridZ = (long long)SP->BoxGridPoints.z;

        //dx and dy and dz for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;
        varfloat GridDZ = SP->GridDelta.z;

        #pragma omp parallel for
        for (long long xx = 0; xx < SP->BoxGridPoints.x; xx++) {
            for (long long yy = 0; yy < SP->BoxGridPoints.y; yy++) {
                for (long long zz = 0; zz < SP->BoxGridPoints.z; zz++) {
                    long long idxCenter = xx + GridX * (yy + GridY * zz);

                    if (SourceX[idxCenter] != SourceX[idxCenter]) {
                        //The source value here is a nan, so simply makes the RHS at this point a nan as well
                        RHS[idxCenter] = NAN;
                    }
                    else {
                        long long idx_xp = idxCenter + 1;
                        long long idx_xm = idxCenter - 1;
                        long long idx_yp = idxCenter + GridX;
                        long long idx_ym = idxCenter - GridX;
                        long long idx_zp = idxCenter + GridX * GridY;
                        long long idx_zm = idxCenter - GridX * GridY;

                        //Computes the boolean values for each index
                        varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]);
                        varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
                        varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
                        varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);
                        varfloat bzp = ((zz + 1) >= GridZ) || (SourceX[idx_zp] != SourceX[idx_zp]);
                        varfloat bzm = ((zz - 1) < 0) || (SourceX[idx_zm] != SourceX[idx_zm]);

                        if (SP->Kernel == FACE_CROSSING){
                            //Face crossing Kernel uses less points.

                            //Computes the weights for the [n] coefficients
                            varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm); //Weight for the center coefficient
                            varfloat w_in_1 = 1.0 - w_in;

                            //Adds the pressure values to right-hand side for this cell   
                            varfloat R = 0.0;
                            R += bxp ? 0.0 : (-SP->cx * (SourceX[idx_xp] + SourceX[idxCenter]) * GridDX / 2) / w_in_1;
                            R += bxm ? 0.0 : (SP->cx * (SourceX[idx_xm] + SourceX[idxCenter]) * GridDX / 2) / w_in_1;
                            R += byp ? 0.0 : (-SP->cy * (SourceY[idx_yp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                            R += bym ? 0.0 : (SP->cy * (SourceY[idx_ym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                            R += bzp ? 0.0 : (-SP->cz * (SourceZ[idx_zp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;
                            R += bzm ? 0.0 : (SP->cz * (SourceZ[idx_zm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;
                            RHS[idxCenter] = R;
                        }
                        else if(SP->Kernel == CELL_CENTERED){
                            //Cell centered Kernel 
                            long long idx_xpyp = idxCenter + GridX + 1; //xy edges
                            long long idx_xpym = idxCenter - GridX + 1;
                            long long idx_xmyp = idxCenter + GridX - 1;
                            long long idx_xmym = idxCenter - GridX - 1;
                            
                            long long idx_xpzp = idxCenter + GridX * GridY + 1; //xz edges
                            long long idx_xpzm = idxCenter - GridX * GridY + 1;
                            long long idx_xmzp = idxCenter + GridX * GridY - 1;
                            long long idx_xmzm = idxCenter - GridX * GridY - 1;

                            long long idx_ypzp = idxCenter + GridX * GridY + GridX; //yz edges
                            long long idx_ypzm = idxCenter - GridX * GridY + GridX;
                            long long idx_ymzp = idxCenter + GridX * GridY - GridX;
                            long long idx_ymzm = idxCenter - GridX * GridY - GridX;

                            long long idx_xpypzp = idxCenter + 1 + GridX + GridX * GridY; //x+ corners
                            long long idx_xpypzm = idxCenter + 1 + GridX - GridX * GridY;
                            long long idx_xpymzp = idxCenter + 1 - GridX + GridX * GridY;
                            long long idx_xpymzm = idxCenter + 1 - GridX - GridX * GridY;

                            long long idx_xmypzp = idxCenter - 1 + GridX + GridX * GridY; //x- corners
                            long long idx_xmypzm = idxCenter - 1 + GridX - GridX * GridY;
                            long long idx_xmymzp = idxCenter - 1 - GridX + GridX * GridY;
                            long long idx_xmymzm = idxCenter - 1 - GridX - GridX * GridY;
                        
                            //Computes boolean values for each index
                            varfloat bxpyp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || (SourceX[idx_xpyp] != SourceX[idx_xpyp]);  //xy edges
                            varfloat bxpym = ((xx + 1) >= GridX) || ((yy - 1) < 0) || (SourceX[idx_xpym] != SourceX[idx_xpym]);
                            varfloat bxmyp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || (SourceX[idx_xmyp] != SourceX[idx_xmyp]);
                            varfloat bxmym = ((xx - 1) < 0) || ((yy - 1) < 0) || (SourceX[idx_xmym] != SourceX[idx_xmym]);
                        
                            varfloat bxpzp = ((xx + 1) >= GridX) || ((zz + 1) >= GridZ) || (SourceX[idx_xpzp] != SourceX[idx_xpzp]);  //xz edges
                            varfloat bxpzm = ((xx + 1) >= GridX) || ((zz - 1) < 0) || (SourceX[idx_xpzm] != SourceX[idx_xpzm]);
                            varfloat bxmzp = ((xx - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_xmzp] != SourceX[idx_xmzp]);
                            varfloat bxmzm = ((xx - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_xmzm] != SourceX[idx_xmzm]);
                        
                            varfloat bypzp = ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (SourceX[idx_ypzp] != SourceX[idx_ypzp]);  //yz edges
                            varfloat bypzm = ((yy + 1) >= GridY) || ((zz - 1) < 0) || (SourceX[idx_ypzm] != SourceX[idx_ypzm]);
                            varfloat bymzp = ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_ymzp] != SourceX[idx_ymzp]);
                            varfloat bymzm = ((yy - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_ymzm] != SourceX[idx_ymzm]);

                            varfloat bxpypzp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (SourceX[idx_xpypzp] != SourceX[idx_xpypzp]);  //x+ corners
                            varfloat bxpypzm = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (SourceX[idx_xpypzm] != SourceX[idx_xpypzm]); 
                            varfloat bxpymzp = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_xpymzp] != SourceX[idx_xpymzp]); 
                            varfloat bxpymzm = ((xx + 1) >= GridX) || ((yy - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_xpymzm] != SourceX[idx_xpymzm]); 

                            varfloat bxmypzp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || (SourceX[idx_xmypzp] != SourceX[idx_xmypzp]);  //x- corners
                            varfloat bxmypzm = ((xx - 1) < 0) || ((yy + 1) >= GridY) || ((zz - 1) < 0) || (SourceX[idx_xmypzm] != SourceX[idx_xmypzm]); 
                            varfloat bxmymzp = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz + 1) >= GridZ) || (SourceX[idx_xmymzp] != SourceX[idx_xmymzp]); 
                            varfloat bxmymzm = ((xx - 1) < 0) || ((yy - 1) < 0) || ((zz - 1) < 0) || (SourceX[idx_xmymzm] != SourceX[idx_xmymzm]); 

                            //Adds the pressure values to right-hand side for this cell 
                            varfloat w_in = SP->cx * (bxp + bxm) + SP->cy * (byp + bym) + SP->cz * (bzp + bzm) + 
                                            SP->cxy * (bxpyp + bxpym + bxmyp + bxmym) + SP->cxz * (bxpzp + bxpzm + bxmzp + bxmzm) + 
                                            SP->cyz * (bypzp + bypzm + bymzp + bymzm) + SP->cxyz * (bxpypzp + bxpypzm + bxpymzp + bxpymzm + bxmypzp + bxmypzm + bxmymzp + bxmymzm); 
                                            //Weight for the center coefficient
                            varfloat w_in_1 = 1.0 - w_in;

                            //Adds the pressure values to right-hand side for this cell   
                            varfloat R = 0.0;
                            R -= bxp ? 0.0 : SP->cx * (+(SourceX[idx_xp] + SourceX[idxCenter]) * GridDX / 2) / w_in_1; //Faces
                            R -= bxm ? 0.0 : SP->cx * (-(SourceX[idx_xm] + SourceX[idxCenter]) * GridDX / 2) / w_in_1;
                            R -= byp ? 0.0 : SP->cy * (+(SourceY[idx_yp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                            R -= bym ? 0.0 : SP->cy * (-(SourceY[idx_ym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1;
                            R -= bzp ? 0.0 : SP->cz * (+(SourceZ[idx_zp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;
                            R -= bzm ? 0.0 : SP->cz * (-(SourceZ[idx_zm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1;

                            R -= bxpyp ? 0.0 : SP->cxy * (+(SourceX[idx_xpyp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xpyp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; //Edges xy
                            R -= bxpym ? 0.0 : SP->cxy * (+(SourceX[idx_xpym] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xpym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; 
                            R -= bxmyp ? 0.0 : SP->cxy * (-(SourceX[idx_xmyp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xmyp] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; 
                            R -= bxmym ? 0.0 : SP->cxy * (-(SourceX[idx_xmym] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xmym] + SourceY[idxCenter]) * GridDY / 2) / w_in_1; 

                            R -= bxpzp ? 0.0 : SP->cxz * (+(SourceX[idx_xpzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceZ[idx_xpzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Edges xz
                            R -= bxpzm ? 0.0 : SP->cxz * (+(SourceX[idx_xpzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceZ[idx_xpzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bxmzp ? 0.0 : SP->cxz * (-(SourceX[idx_xmzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceZ[idx_xmzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bxmzm ? 0.0 : SP->cxz * (-(SourceX[idx_xmzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceZ[idx_xmzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 

                            R -= bypzp ? 0.0 : SP->cyz * (+(SourceY[idx_ypzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_ypzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Edges yz
                            R -= bypzm ? 0.0 : SP->cyz * (+(SourceY[idx_ypzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_ypzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bymzp ? 0.0 : SP->cyz * (-(SourceY[idx_ymzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_ymzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bymzm ? 0.0 : SP->cyz * (-(SourceY[idx_ymzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_ymzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 

                            R -= bxpypzp ? 0.0 : SP->cxyz * (+(SourceX[idx_xpypzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xpypzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xpypzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Corners x+
                            R -= bxpypzm ? 0.0 : SP->cxyz * (+(SourceX[idx_xpypzm] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xpypzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xpypzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bxpymzp ? 0.0 : SP->cxyz * (+(SourceX[idx_xpymzp] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xpymzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xpymzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bxpymzm ? 0.0 : SP->cxyz * (+(SourceX[idx_xpymzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xpymzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xpymzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 

                            R -= bxmypzp ? 0.0 : SP->cxyz * (-(SourceX[idx_xmypzp] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xmypzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xmypzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; //Corners x-
                            R -= bxmypzm ? 0.0 : SP->cxyz * (-(SourceX[idx_xmypzm] + SourceX[idxCenter]) * GridDX / 2 + (SourceY[idx_xmypzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xmypzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bxmymzp ? 0.0 : SP->cxyz * (-(SourceX[idx_xmymzp] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xmymzp] + SourceY[idxCenter]) * GridDY / 2 + (SourceZ[idx_xmymzp] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            R -= bxmymzm ? 0.0 : SP->cxyz * (-(SourceX[idx_xmymzm] + SourceX[idxCenter]) * GridDX / 2 - (SourceY[idx_xmymzm] + SourceY[idxCenter]) * GridDY / 2 - (SourceZ[idx_xmymzm] + SourceZ[idxCenter]) * GridDZ / 2) / w_in_1; 
                            RHS[idxCenter] = R;
                        }
                    }
                }
            }
        }
    }
}

template <typename varfloat>
void ConjugateGradientSolver_CPU(varfloat* PressureField, varfloat* RHS, SolverParameters<varfloat> SolverConfig, BoxContents<varfloat> VTK_Contents) {
    // CPU Solver version of the conjugate gradient

    //Allocate memory
    long long boxArraySize = sizeof(varfloat) * VTK_Contents.totalBoxElements;
    varfloat* rk; varfloat* rkp1; varfloat* pk; varfloat* temp;
    rk = (varfloat*)malloc(boxArraySize); rkp1 = (varfloat*)malloc(boxArraySize);
    pk = (varfloat*)malloc(boxArraySize); temp = (varfloat*)malloc(boxArraySize);

    varfloat* beta; varfloat* alpha; varfloat* r_norm; varfloat* r_norm_old; varfloat* temp_scal;
    beta = (varfloat*)malloc(sizeof(varfloat)); alpha = (varfloat*)malloc(sizeof(varfloat)); r_norm = (varfloat*)malloc(sizeof(varfloat));
    r_norm_old = (varfloat*)malloc(sizeof(varfloat)); temp_scal = (varfloat*)malloc(sizeof(varfloat));

    //Start CG solver here [see wikipedia page on Conjugate Gradient to see the steps implemented]
    ClockTic();
    double IdleClock = 0.0; // to keep track of the time so every 10 seconds it still prints something

    bool canExit = false;

    while (!canExit){
        //will only exit the loop if the CG hasn't diverged
        UpdateRHS_Vector_CPU(PressureField, RHS, VTK_Contents.SourceFn_Field_X, VTK_Contents.SourceFn_Field_Y, VTK_Contents.SourceFn_Field_Z, &SolverConfig); //b
        MatrixMul_Omnidirectional_CPU(temp, PressureField, RHS, &SolverConfig); //temp=A*x_0
        
        subtractVectors_CPU(RHS, temp, rk, &SolverConfig); //r_0=b-A*x_0
        memcpy(pk, rk, boxArraySize);
        vectorDot_CPU(rk, rk, r_norm_old, &SolverConfig); //r_k dot r_k

        varfloat r_norm_init; varfloat r_norm_sqrt; 
        varfloat r_norm_min = 1e9; //To restart the CG if divergence is detected
        Progress P_cgs;
        r_norm_init = sqrt(*r_norm_old);
        if (SolverConfig.Verbose){ mexPrintf("Initial Residual Norm=%f\n", r_norm_init);}
        CGS_Progress.clear();
        P_cgs.Iteration = 0; P_cgs.Residual = 1.0f; P_cgs.TimeSeconds = ClockToc(); CGS_Progress.push_back(P_cgs);

        for (int cgs_iter = 0; cgs_iter < VTK_Contents.totalBoxElements; cgs_iter++) {
            //Iterations of the Conjugate Gradient Solver here
            vectorDot_CPU(rk, rk, r_norm_old, &SolverConfig); //r_k dot r_k
            MatrixMul_Omnidirectional_CPU(temp, pk, RHS, &SolverConfig); //temp=A*p_k
            vectorDot_CPU(pk, temp, temp_scal, &SolverConfig); //temp_scal = p_k dot temp
            *alpha = *r_norm_old / *temp_scal;//alpha = (rk dot rk) / (pk dot A*pk)

            //Implicit residual update
            scalarVectorMult_CPU (alpha, temp, temp, &SolverConfig); //temp=alphak*temp
            subtractVectors_CPU (rk, temp, rkp1, &SolverConfig); //r_k+1=rk-temp (i.e. rk-A*temp)

            scalarVectorMult_CPU (alpha, pk, temp, &SolverConfig); //temp = alphak*pk
            addVectors_CPU(PressureField, temp, PressureField, &SolverConfig); //xk+1=xk+alphak*pk

            //Explicit residual update
                //MatrixMul_Omnidirectional_CPU (temp, PressureField, RHS, &SolverConfig); //temp=A*x_k+1
                //subtractVectors_CPU (RHS, temp, rkp1, &SolverConfig); //r_k+1=b-A*xk+1

            memcpy(rk, rkp1, boxArraySize);//rk=rk+1
            vectorDot_CPU (rkp1, rkp1, r_norm, &SolverConfig); //r_k+1 dot r_k+1
            r_norm_sqrt = sqrt(*r_norm);

            //Updates the lowest rnorm
            if (r_norm_min > r_norm_sqrt){
                r_norm_min = r_norm_sqrt;
            }

            if (cgs_iter % 10 == 0) {
                if (SolverConfig.Verbose){ mexPrintf("CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e; \n", cgs_iter, r_norm_sqrt / r_norm_init, r_norm_sqrt);}
            }
            if (!SolverConfig.Verbose){
                //Even if not verbose, still prints every 10 seconds
                if (ClockToc() > (IdleClock + 10.0)) {
                    mexPrintf("CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e; \n", cgs_iter, r_norm_sqrt / r_norm_init, r_norm_sqrt);
                    IdleClock += 10.0;
                }
            }

            //Stores iteration info on memory
            P_cgs.Iteration = cgs_iter+1; P_cgs.Residual = r_norm_sqrt / r_norm_init; P_cgs.TimeSeconds = ClockToc(); CGS_Progress.push_back(P_cgs);

            if ((r_norm_sqrt / r_norm_init > SolverConfig.solverToleranceRel) && (r_norm_sqrt > SolverConfig.solverToleranceAbs)) {
                //Only continues if not yet within tolerance
                *beta = *r_norm / *r_norm_old;//beta = (rk+1 dot rk+1) / (rk dot rk)
                scalarVectorMult_CPU (beta, pk, temp, &SolverConfig); //temp=beta*pk
                addVectors_CPU(temp, rkp1, pk, &SolverConfig); //pk+1=rk+1 + beta*pk 
            }
            else {
                if (SolverConfig.Verbose){ mexPrintf("CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e [Converged]\n", cgs_iter, r_norm_sqrt / r_norm_init, r_norm_sqrt);}

                if (isnan(r_norm_sqrt)) {
                    mexPrintf("======== Result was NAN! ========\n");
                    mexPrintf("Make sure your coordinate system is correct. This code expects a coordinate in the 'ND grid' format, i.e., dimension order is (x, y) or (x, y, z). DO NOT USE the 'MESHGRID' format to build the arrays in Matlab!\n");
                }
                canExit = true; // technically not required but it reads better
                return;
            }

            if (r_norm_sqrt > (r_norm_min * 1e2)) { //factor of 1e2 arbitrary
                //CG is diverging, restarts
                //FillBox(PressureField, ZEROS, SolverConfig);
                SolverConfig.solverToleranceAbs = r_norm_min * 1.001; //Sets the new tolerance to a slightly higher value than the best achieved so we can exit next time
                mexPrintf("==========CG Diverged! Restarting with AbsTol = %0.2e=========.\=\n", SolverConfig.solverToleranceAbs);
                break; // restarts the whole CG
            }
        }
    }

    free(rk); free(rkp1);
    free(pk); free(temp);
}

#pragma endregion


//===================Main Matlab function================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //Function prototypes:
    //[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz); %Basic form, source term Sx, Sy, Sz are either 2D matrices or 3D matrices. We use **ND grid** format here. Uses default options, and delta=1.
    //[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, delta); %Also provides a grid spacing delta which is the same in all directions.
    //[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, [dx dy dz]); %Provides a grid spacing that is different for x, y, z but is still constant for each direction.
    //[P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, delta, options); %Also provides options as a struct. See below.

    //=====Initializes and checks arguments=====
    //Check for the correct number of input and output arguments
    if (nrhs < 3) {
        mexErrMsgIdAndTxt("OSMODI:mexFunction:invalidNumInputs", "At least 3 input arguments required: [P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, [dx dy dz], options);");
    }
    if (nrhs > 5) {
        mexErrMsgIdAndTxt("OSMODI:mexFunction:invalidNumInputs", "At most 5 input arguments possible: [P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, [dx dy dz], options);");
    }
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("OSMODI:mexFunction:invalidNumOutputs", "Two output arguments required: [P, CGS_Residuals] = OSMODI(Sx, Sy, Sz, [dx dy dz], options);");
    }

    //Ensures all input arguments are matching the varfloat
	if (sizeof(varfloat) == sizeof(float)){
		//Inputs must be singles if code was compiled with functions inputs as singles
		if ((!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]) || !mxIsSingle(prhs[2])) || ((nrhs > 3) && !mxIsSingle(prhs[3]))) {
			mexErrMsgIdAndTxt("OSMODI:mexFunction:inputNotSingle", "All inputs must be single precision. This code was implemented only in single precision to maximize GPU usage. If the inputs are double, please use: \n [P, CGS_Residuals] = OSMODI(single(Sx), single(Sy), single(Sz), single([dx dy dz]));");
		}
	}
	else if (sizeof(varfloat) == sizeof(double)){
		//Inputs must be doubles if code was compiled with functions inputs as doubles
		if ((!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2])) || ((nrhs > 3) && !mxIsDouble(prhs[3]))) {
			mexErrMsgIdAndTxt("OSMODI:mexFunction:inputNotDouble", "All inputs must be double precision. If the inputs are single, please use: \n [P, CGS_Residuals] = OSMODI(double(Sx), double(Sy), double(Sz), double([dx dy dz]));");
		}
	}
	else {
		mexErrMsgIdAndTxt("OSMODI:mexFunction:varfloatIncorrect", "This OSMODI code was compiled incorrectly. the template variable varfloat must be either float or double.");
	}

    // Starts filling the arguments
    SolverParameters<varfloat> SolverConfig;
    BoxContents<varfloat> VTK_Contents;
    if (nrhs == 3) {
        SolverConfig.GridDelta = {1.0f, 1.0f, 1.0f};
        VTK_Contents.GridDelta = {1.0f, 1.0f, 1.0f};
    }

    //Ensures Sx, Sy, Sz are of the same size
    int numDimensions1 = mxGetNumberOfDimensions(prhs[0]);
    int numDimensions2 = mxGetNumberOfDimensions(prhs[1]);
    int numDimensions3 = mxGetNumberOfDimensions(prhs[2]);

    if (numDimensions1!=numDimensions2 || numDimensions1!=numDimensions3 || numDimensions2!=numDimensions3) {
        mexErrMsgIdAndTxt("OSMODI:mexFunction:dimensionMismatch", "Sx, Sy and Sz must have the same number of dimensions!");
    }
    if (!(numDimensions1 == 2 || numDimensions1 == 3)) {
        mexErrMsgIdAndTxt("OSMODI:mexFunction:dimensionCount", "Sx, Sy and Sz must be either 2D or 3D matrices!");
    }

    const mwSize *dims1 = mxGetDimensions(prhs[0]);
    const mwSize *dims2 = mxGetDimensions(prhs[1]);
    const mwSize *dims3 = mxGetDimensions(prhs[2]);

    if (numDimensions1==2) {
        if ((dims1[0]!=dims2[0]) || (dims1[0]!=dims3[0]) || (dims2[0]!=dims3[0]) || 
            (dims1[1]!=dims2[1]) || (dims1[1]!=dims3[1]) || (dims2[1]!=dims3[1])) {
            mexErrMsgIdAndTxt("OSMODI:mexFunction:dimensionCountMismatch", "Sx, Sy and Sz must have exactly the same number of dimensions!");
        }
    }
    else if(numDimensions1==3){
        if ((dims1[0]!=dims2[0]) || (dims1[0]!=dims3[0]) || (dims2[0]!=dims3[0]) || 
            (dims1[1]!=dims2[1]) || (dims1[1]!=dims3[1]) || (dims2[1]!=dims3[1]) || 
            (dims1[2]!=dims2[2]) || (dims1[2]!=dims3[2]) || (dims2[2]!=dims3[2])) {
            mexErrMsgIdAndTxt("OSMODI:mexFunction:dimensionCountMismatch", "Sx, Sy and Sz must have exactly the same number of dimensions!");
        }
    }

    //If we're here then Sx, Sy and Sz are 2D or 3D and have the same number of dimensions.
    //Loads the sizes on the configuration variables
    if (numDimensions1==2) {
        SolverConfig.BoxGridPoints = {static_cast<int> (dims1[0]), static_cast<int> (dims1[1]), 1};
        VTK_Contents.BoxGridSize = {static_cast<int> (dims1[0]), static_cast<int> (dims1[1]), 1};
        SolverConfig.totalBoxElements = static_cast<long long> (dims1[0]) * static_cast<long long> (dims1[1]);
        VTK_Contents.totalBoxElements = static_cast<long long> (dims1[0]) * static_cast<long long> (dims1[1]);
    }
    else if(numDimensions1==3){
        SolverConfig.BoxGridPoints = {static_cast<int> (dims1[0]), static_cast<int> (dims1[1]), static_cast<int> (dims1[2])};
        VTK_Contents.BoxGridSize = {static_cast<int> (dims1[0]), static_cast<int> (dims1[1]), static_cast<int> (dims1[2])};
        SolverConfig.totalBoxElements = static_cast<long long> (dims1[0]) * static_cast<long long> (dims1[1]) * static_cast<long long> (dims1[2]);
        VTK_Contents.totalBoxElements = static_cast<long long> (dims1[0]) * static_cast<long long> (dims1[1]) * static_cast<long long> (dims1[2]);
    }
    

    //Loads the actual field pointers to memory
    VTK_Contents.SourceFn_Field_X = (varfloat*) mxGetPr(prhs[0]);
    VTK_Contents.SourceFn_Field_Y = (varfloat*) mxGetPr(prhs[1]);
    VTK_Contents.SourceFn_Field_Z = (varfloat*) mxGetPr(prhs[2]);

    //Fills in the gridDeltas if provided
    if (nrhs > 3) {
        varfloat* gridDeltas = (varfloat*) mxGetPr(prhs[3]);
        int gridDeltasSize = max(static_cast<int>(mxGetM(prhs[3])), static_cast<int>(mxGetN(prhs[3])));

        if (gridDeltasSize == 1){
            SolverConfig.GridDelta = {gridDeltas[0], gridDeltas[0], gridDeltas[0]};
            VTK_Contents.GridDelta = {gridDeltas[0], gridDeltas[0], gridDeltas[0]};        
        } 
        else if (gridDeltasSize >= 3){
            SolverConfig.GridDelta = {gridDeltas[0], gridDeltas[1], gridDeltas[2]};
            VTK_Contents.GridDelta = {gridDeltas[0], gridDeltas[1], gridDeltas[2]};        
        } 
        else if (gridDeltasSize == 2 && numDimensions1 == 2){
            //Unused dimension z
            SolverConfig.GridDelta = {gridDeltas[0], gridDeltas[1], 1.0f};
            VTK_Contents.GridDelta = {gridDeltas[0], gridDeltas[1], 1.0f};        
        } 
        else{
            mexErrMsgIdAndTxt("OSMODI:mexFunction:gridDeltaError", "Wrong size for the grid delta. If the grid is 2D, delta can be any size, but for a 3D grid delta has to be 1-long or 3-long. If delta is longer than 3, only the first 3 elements are considered.");
        }  
    }

    //Fills in the options
    if (nrhs == 5) {
        const mxArray *inputOptions = prhs[4];

        int fieldNo;
        fieldNo = mxGetFieldNumber(inputOptions, "Verbose");
        if (fieldNo != -1){
            const mxArray *mxVerbose = mxGetField(inputOptions, 0, "Verbose");
            varfloat verbose = (varfloat) mxGetScalar(mxVerbose);
            if (verbose == 1.0f){
                SolverConfig.Verbose = true;
                mexPrintf("SolverConfig.Verbose = TRUE\n");
                mexPrintf("Welcome to OSMODI! (Rev Nov 05 2024)\n");
            }else{
                SolverConfig.Verbose = false;
            }            
        }

        fieldNo = mxGetFieldNumber(inputOptions, "SolverToleranceRel");
        if (fieldNo != -1){
            const mxArray *mxSolverTolRel = mxGetField(inputOptions, 0, "SolverToleranceRel");
            SolverConfig.solverToleranceRel = (varfloat) mxGetScalar(mxSolverTolRel);
            if (SolverConfig.Verbose){
                mexPrintf("SolverConfig.solverToleranceRel = %f\n", SolverConfig.solverToleranceRel);
            }
        }
        
        fieldNo = mxGetFieldNumber(inputOptions, "SolverToleranceAbs");
        if (fieldNo != -1){
            const mxArray *mxSolverTolAbs = mxGetField(inputOptions, 0, "SolverToleranceAbs");
            SolverConfig.solverToleranceAbs = (varfloat) mxGetScalar(mxSolverTolAbs);
            if (SolverConfig.Verbose){
                mexPrintf("SolverConfig.solverToleranceAbs = %f\n", SolverConfig.solverToleranceAbs);
            }
        }
        
        fieldNo = mxGetFieldNumber(inputOptions, "SolverDevice");
        if (fieldNo != -1){
            const mxArray *mxSolverDev = mxGetField(inputOptions, 0, "SolverDevice");
            if (!mxIsChar(mxSolverDev)) {
                mexErrMsgIdAndTxt("OSMODI:mexFunction:deviceMustBeString", "SolverDevice must be a string, either 'GPU' or 'CPU'.");
            }
            else{
                string buffer = mxArrayToString(mxSolverDev);
                if (iequals(buffer, "CPU")){
                    SolverConfig.SolverDevice = CPU;
                    if (SolverConfig.Verbose){
                        mexPrintf("SolverConfig.SolverDevice = CPU\n");
                    }
                }
                else if(iequals(buffer, "GPU")){
                    SolverConfig.SolverDevice = GPU;
                    if (SolverConfig.Verbose){
                        mexPrintf("SolverConfig.SolverDevice = GPU\n");
                    }
                }
                else{
                    mexErrMsgIdAndTxt("OSMODI:mexFunction:deviceError", "SolverDevice must be either 'GPU' or 'CPU'.");
                }
            }            
        }   
        
        fieldNo = mxGetFieldNumber(inputOptions, "Kernel");
        if (fieldNo != -1){
            const mxArray *mxKernel = mxGetField(inputOptions, 0, "Kernel");
            if (!mxIsChar(mxKernel)) {
                mexErrMsgIdAndTxt("OSMODI:mexFunction:deviceMustBeString", "Kernel must be a string, either 'face-crossing' or 'cell-centered'.");
            }
            else{
                string buffer = mxArrayToString(mxKernel);
                if (iequals(buffer, "face-crossing")){
                    SolverConfig.Kernel = FACE_CROSSING;
                    if (SolverConfig.Verbose){
                        mexPrintf("SolverConfig.Kernel = FACE_CROSSING\n");
                    }
                }
                else if(iequals(buffer, "cell-centered")){
                    SolverConfig.Kernel = CELL_CENTERED;
                    if (SolverConfig.Verbose){
                        mexPrintf("SolverConfig.Kernel = CELL_CENTERED\n");
                    }
                }
                else{
                    mexErrMsgIdAndTxt("OSMODI:mexFunction:deviceError", "Kernel must be a string, either 'face-crossing' or 'cell-centered'.");
                }
            }            
        }       
    }

    //Calculates constants for kernels
    if (numDimensions1==2) {
        if (SolverConfig.Kernel == FACE_CROSSING){
            varfloat cx = 2.0 * SolverConfig.GridDelta.y;
            varfloat cy = 2.0 * SolverConfig.GridDelta.x;
            varfloat ctot = 2.0 * (cx + cy); 
            SolverConfig.cx = cx / ctot;
            SolverConfig.cy = cy / ctot;
        }
        else if (SolverConfig.Kernel == CELL_CENTERED){
            varfloat dx = SolverConfig.GridDelta.y;
            varfloat dy = SolverConfig.GridDelta.x;
            varfloat dd = sqrt(dx * dx + dy * dy);
            varfloat ctot = 4.0 * dd; 
            SolverConfig.cx = 2.0 * (dd - dx) / ctot;
            SolverConfig.cy = 2.0 * (dd - dy) / ctot;
            SolverConfig.cxy = (dx + dy - dd) / ctot;
        }
    }
    else if(numDimensions1==3){
        if (SolverConfig.Kernel == FACE_CROSSING){
            varfloat cx = 1.0 * SolverConfig.GridDelta.y * SolverConfig.GridDelta.z; //That 1.0 is supposed to be a pi in the paper. It cancels out, so there's no point in adding it here.
            varfloat cy = 1.0 * SolverConfig.GridDelta.x * SolverConfig.GridDelta.z;
            varfloat cz = 1.0 * SolverConfig.GridDelta.x * SolverConfig.GridDelta.y;
            varfloat ctot = 2.0 * (cx + cy + cz); 
            SolverConfig.cx = cx / ctot;
            SolverConfig.cy = cy / ctot;
            SolverConfig.cz = cz / ctot;
        }
        else if (SolverConfig.Kernel == CELL_CENTERED){
            PrecomputeConstants(&SolverConfig);
            //mexPrintf("(cx, cxy, cxyz)=%f,%f,%f\n\n", SolverConfig.cx, SolverConfig.cxy, SolverConfig.cxyz);
        }
    }

    if (SolverConfig.Verbose){
        mexPrintf("Input Box Size = [%d, %d, %d]\n", SolverConfig.BoxGridPoints.x, SolverConfig.BoxGridPoints.y, SolverConfig.BoxGridPoints.z);
        mexPrintf("Total Box Size = %lld\n", SolverConfig.totalBoxElements);
        mexPrintf("Input Box Deltas = [%f, %f, %f]\n", SolverConfig.GridDelta.x, SolverConfig.GridDelta.y, SolverConfig.GridDelta.z);
    }

    //Prepares the output variable for Pressure in memory
    mxArray *PressureOut;
    if (numDimensions1==2) {
        mwSize arrayDim[2] = {dims1[0], dims1[1]};
		if (sizeof(varfloat) == sizeof(float)){
			PressureOut = mxCreateNumericArray(numDimensions1,arrayDim, mxSINGLE_CLASS, mxREAL);			
		}
		else if(sizeof(varfloat) == sizeof(double)){
			PressureOut = mxCreateNumericArray(numDimensions1,arrayDim, mxDOUBLE_CLASS, mxREAL);
		}
    }
    else if(numDimensions1==3){
        mwSize arrayDim[3] = {dims1[0], dims1[1], dims1[2]};
		if (sizeof(varfloat) == sizeof(float)){
			PressureOut = mxCreateNumericArray(numDimensions1,arrayDim, mxSINGLE_CLASS, mxREAL);			
		}
		else if(sizeof(varfloat) == sizeof(double)){
			PressureOut = mxCreateNumericArray(numDimensions1,arrayDim, mxDOUBLE_CLASS, mxREAL);
		}
    }
    varfloat *PressureField = static_cast<varfloat*>(mxGetData(PressureOut));

    //Initializes the GPU
    bool GPUSuccess;
    GPUSuccess = InitializeGPU(SolverConfig);
    if (!GPUSuccess) {
        mexPrintf("***Error - GPU Not Initialized. Defaulting to CPU code.***");
    }

    //===Starts the solver====
    if(SolverConfig.Verbose){mexPrintf("========Starting up the solver...========\n");}

    ClockTic();
    // Allocate CPU memory for the result and fills with zeros
    FillBox(PressureField, ZEROS, SolverConfig); // Initializes with zeros

    varfloat* RHS; 
    RHS = (varfloat*)malloc(sizeof(varfloat) * VTK_Contents.totalBoxElements);
    FillBox(RHS, ZEROS, SolverConfig); // Initializes with zeros

    //Solves the equations with the solver
    if ((SolverConfig.SolverDevice == GPU) && GPUSuccess) {
        if(SolverConfig.Verbose){mexPrintf("Starting GPU Solver...\n");}
        ConjugateGradientSolver_GPU(PressureField, RHS, SolverConfig, VTK_Contents);
    }
    else {
        if(SolverConfig.Verbose){mexPrintf("Starting CPU Solver...\n");}
        ConjugateGradientSolver_CPU(PressureField, RHS, SolverConfig, VTK_Contents);
    }

    free(RHS); //Fixes memory leak

    //Outputs the results
    plhs[0] = PressureOut; //Outputs pressure
        
    int nRows = CGS_Progress.size();//Iterates through the contents of the vector CGS_Progress to output the progress as a matrix
	
	mxArray *CGS_Progress_mx;
	if (sizeof(varfloat) == sizeof(float)){
		CGS_Progress_mx = mxCreateNumericMatrix(nRows, 3, mxSINGLE_CLASS, mxREAL);		
	}
	else if(sizeof(varfloat) == sizeof(double)){
		CGS_Progress_mx = mxCreateNumericMatrix(nRows, 3, mxDOUBLE_CLASS, mxREAL);
	}    
	
    varfloat *CGS_Progress_f = static_cast<varfloat*>(mxGetData(CGS_Progress_mx));
    for (int i=0; i<nRows; i++){
        CGS_Progress_f[0 * nRows + i] = (varfloat) CGS_Progress[i].Iteration;
        CGS_Progress_f[1 * nRows + i] = (varfloat) CGS_Progress[i].Residual;
        CGS_Progress_f[2 * nRows + i] = (varfloat) CGS_Progress[i].TimeSeconds;
    }    
    plhs[1] = CGS_Progress_mx;

    //Happy ending
    if (SolverConfig.Verbose){mexPrintf("OSMODI completed successfully!\n");}
}

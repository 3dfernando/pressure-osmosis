#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <omp.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;
using namespace std;

// Constants
#define PI 3.141592653589793
#define GPU 1
#define CPU 0
#define FACE_CROSSING 0
#define CELL_CENTERED 1
#define ZEROS 0
#define ONES 1
#define RANDOMS 2

// Structs
struct int3 {
    int x, y, z;
};

template <typename varfloat>
struct varfloat3 {
    varfloat x;
    varfloat y;
    varfloat z;
};

template <typename varfloat>
struct SolverParameters {
    bool Verbose = false; //Whether to send messages to console
    int SolverDevice = CPU; //CPU or GPU
    int Kernel = FACE_CROSSING; //FACE_CROSSING, CELL_CENTERED
    varfloat solverToleranceRel = 1e-4; //Relative error allowed for the solver
    varfloat solverToleranceAbs = 1e-4; //Absolute error allowed for the solver
    int3 BoxGridPoints = {100, 100, 1}; //Number of grid points in the box
    varfloat3<varfloat> GridDelta = {1,1,1}; //Delta value for derivative approximation
    long long totalBoxElements = 100*100*1; //Tracks the size of the box
    
    varfloat cx = 1; //Constant for the x-face
    varfloat cy = 1; //Constant for the y-face
    varfloat cz = 1; //Constant for the z-face
    varfloat cxy = 1; //Constant for the xy-edge
    varfloat cxz = 1; //Constant for the xz-edge
    varfloat cyz = 1; //Constant for the yz-edge
    varfloat cxyz = 1; //Constant for the xyz-corner
};

template <typename varfloat>
struct BoxContents {
    varfloat* SourceFn_Field_X = nullptr; //Stores the Source Function here
    varfloat* SourceFn_Field_Y = nullptr; //Stores the Source Function here
    varfloat* SourceFn_Field_Z = nullptr; //Stores the Source Function here
    int3 BoxGridSize = {100, 100, 1}; //Number of grid points in the box
    varfloat3<varfloat> GridDelta = {1,1,1}; //Delta value for derivative approximation
    long long totalBoxElements = 100*100*1; //Tracks the size of the box
};

struct Progress { //Struct to store the progress of the CG solver
    int Iteration;
    float Residual;
    double TimeSeconds;
};

vector<Progress> CGS_Progress;

#pragma region

bool iequals(const string& a, const string& b)
{
    return std::equal(a.begin(), a.end(),
                      b.begin(), b.end(),
                      [](char a, char b) {
                          return tolower(a) == tolower(b);
                      });
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
    //Function Axyz from ProjectionCoefficients Writeup (Zigunov&Charonko, MST, 2024, Supp. Material, Eq 101)
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

//========CPU OpenMP Functions======
#pragma region

template <typename varfloat>
void scalarVectorMult_CPU(varfloat* scalar, varfloat* a, varfloat* out, SolverParameters<varfloat>* SP) {
    #pragma omp parallel for
    for (long long i = 0; i < SP->totalBoxElements; i++) {
        out[i] = a[i] * *scalar;
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
        //2D case
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
                    printf("%lld," ,idxCenter);
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
    rk = (varfloat*)malloc(boxArraySize); 
    rkp1 = (varfloat*)malloc(boxArraySize);
    pk = (varfloat*)malloc(boxArraySize); 
    temp = (varfloat*)malloc(boxArraySize);

    varfloat* beta; varfloat* alpha; varfloat* r_norm; varfloat* r_norm_old; varfloat* temp_scal;
    beta = (varfloat*)malloc(sizeof(varfloat)); 
    alpha = (varfloat*)malloc(sizeof(varfloat)); 
    r_norm = (varfloat*)malloc(sizeof(varfloat));
    r_norm_old = (varfloat*)malloc(sizeof(varfloat)); 
    temp_scal = (varfloat*)malloc(sizeof(varfloat));

    //Start CG solver here [see wikipedia page on Conjugate Gradient to see the steps implemented]
    bool canExit = false;

    while (!canExit){
        //will only exit the loop if the CG hasn't diverged
        UpdateRHS_Vector_CPU(PressureField, RHS, VTK_Contents.SourceFn_Field_X, VTK_Contents.SourceFn_Field_Y, VTK_Contents.SourceFn_Field_Z, &SolverConfig); //b
        MatrixMul_Omnidirectional_CPU(temp, PressureField, RHS, &SolverConfig); //temp=A*x_0
        
        subtractVectors_CPU(RHS, temp, rk, &SolverConfig); //r_0=b-A*x_0
        memcpy(pk, rk, boxArraySize);
        vectorDot_CPU(rk, rk, r_norm_old, &SolverConfig); //r_k dot r_k

        varfloat r_norm_init; 
        varfloat r_norm_sqrt; 
        varfloat r_norm_min = 1e9; //To restart the CG if divergence is detected
        Progress P_cgs;
        r_norm_init = sqrt(*r_norm_old);
        
        if (SolverConfig.Verbose){ 
            cout << "Initial Residual Norm=" << r_norm_init << endl;
        }
        
        CGS_Progress.clear();
        P_cgs.Iteration = 0; 
        P_cgs.Residual = 1.0f; 
        P_cgs.TimeSeconds = 0.0;
        CGS_Progress.push_back(P_cgs);

        for (int cgs_iter = 0; cgs_iter < VTK_Contents.totalBoxElements; cgs_iter++) {
            //Iterations of the Conjugate Gradient Solver here
            vectorDot_CPU(rk, rk, r_norm_old, &SolverConfig); //r_k dot r_k
            MatrixMul_Omnidirectional_CPU(temp, pk, RHS, &SolverConfig); //temp=A*p_k
            vectorDot_CPU(pk, temp, temp_scal, &SolverConfig); //temp_scal = p_k dot temp
            *alpha = *r_norm_old / *temp_scal;//alpha = (rk dot rk) / (pk dot A*pk)

            //Implicit residual update
            scalarVectorMult_CPU(alpha, temp, temp, &SolverConfig); //temp=alphak*temp
            subtractVectors_CPU(rk, temp, rkp1, &SolverConfig); //r_k+1=rk-temp (i.e. rk-A*temp)

            scalarVectorMult_CPU(alpha, pk, temp, &SolverConfig); //temp = alphak*pk
            addVectors_CPU(PressureField, temp, PressureField, &SolverConfig); //xk+1=xk+alphak*pk

            //Explicit residual update
                //MatrixMul_Omnidirectional_CPU(temp, PressureField, RHS, &SolverConfig); //temp=A*x_k+1
                //subtractVectors_CPU(RHS, temp, rkp1, &SolverConfig); //r_k+1=b-A*xk+1

            memcpy(rk, rkp1, boxArraySize);//rk=rk+1
            vectorDot_CPU(rkp1, rkp1, r_norm, &SolverConfig); //r_k+1 dot r_k+1
            r_norm_sqrt = sqrt(*r_norm);

            //Updates the lowest rnorm
            if (r_norm_min > r_norm_sqrt){
                r_norm_min = r_norm_sqrt;
            }

            if (cgs_iter % 10 == 0) {
                if (SolverConfig.Verbose){ 
                    cout << "CG Iteration=" << cgs_iter 
                         << "; RelRes=" << scientific << r_norm_sqrt / r_norm_init 
                         << "; AbsRes=" << r_norm_sqrt << endl;
                }
            }

            //Stores iteration info on memory
            P_cgs.Iteration = cgs_iter+1; 
            P_cgs.Residual = r_norm_sqrt / r_norm_init; 
            P_cgs.TimeSeconds = 0.0;
            CGS_Progress.push_back(P_cgs);

            if ((r_norm_sqrt / r_norm_init > SolverConfig.solverToleranceRel) && (r_norm_sqrt > SolverConfig.solverToleranceAbs)) {
                //Only continues if not yet within tolerance
                *beta = *r_norm / *r_norm_old;//beta = (rk+1 dot rk+1) / (rk dot rk)
                scalarVectorMult_CPU(beta, pk, temp, &SolverConfig); //temp=beta*pk
                addVectors_CPU(temp, rkp1, pk, &SolverConfig); //pk+1=rk+1 + beta*pk 
            }
            else {
                if (SolverConfig.Verbose){ 
                    cout << "CG Iteration=" << cgs_iter 
                         << "; RelRes=" << scientific << r_norm_sqrt / r_norm_init 
                         << "; AbsRes=" << r_norm_sqrt << " [Converged]" << endl;
                }

                if (isnan(r_norm_sqrt)) {
                    cout << "======== Result was NAN! ========" << endl;
                    cout << "Make sure your coordinate system is correct." << endl;
                }
                canExit = true; // technically not required but it reads better
                break;
            }

            if (r_norm_sqrt > (r_norm_min * 1e2)) { //factor of 1e2 arbitrary
                //CG is diverging, restarts
                //FillBox(PressureField, ZEROS, SolverConfig);
                SolverConfig.solverToleranceAbs = r_norm_min * 1.001; //Sets the new tolerance to a slightly higher value than the best achieved so we can exit next time
                cout << "==========CG Diverged! Restarting with AbsTol = " 
                     << scientific << SolverConfig.solverToleranceAbs << "==========" << endl;
                break; // restarts the whole CG
            }
        }
    }

    free(rk); 
    free(rkp1);
    free(pk); 
    free(temp);
    free(beta);
    free(alpha);
    free(r_norm);
    free(r_norm_old);
    free(temp_scal);
}

#pragma endregion

// Main solver wrapper
template <typename varfloat>
py::tuple call_solver(py::array_t<varfloat, py::array::f_style | py::array::forcecast> Sx,
                      py::array_t<varfloat, py::array::f_style | py::array::forcecast> Sy,
                      py::array_t<varfloat, py::array::f_style | py::array::forcecast> Sz,
                      py::array_t<varfloat> delta,
                      py::dict options) 
{
    // Basic input presence check
    if (Sx.size() == 0 || Sy.size() == 0 || Sz.size() == 0)
        throw std::runtime_error("Error: Sx, Sy, and Sz cannot be empty.");

    // Dimension checks
    if (Sx.ndim() < 2 || Sx.ndim() > 3)
        throw std::runtime_error("Error: Input arrays must be 2D or 3D only.");

    if (!(Sx.ndim() == Sy.ndim() && Sx.ndim() == Sz.ndim()))
        throw std::runtime_error("Error: Sx, Sy, Sz must have the same number of dimensions.");

    for (ssize_t i = 0; i < Sx.ndim(); ++i) {
        if (Sx.shape(i) != Sy.shape(i) || Sx.shape(i) != Sz.shape(i))
            throw std::runtime_error("Error: Sx, Sy, Sz must have the same shape.");
    }

    // Initialize configs
    SolverParameters<varfloat> SolverConfig;
    BoxContents<varfloat> VTK_Contents;

    int dim0, dim1, dim2;
    std::vector<py::ssize_t> output_shape;

    if (Sx.ndim() == 2) {
        dim0 = Sx.shape(0);  
        dim1 = Sx.shape(1);  
        dim2 = 1;
        output_shape = {Sx.shape(0), Sx.shape(1)}; 
    } else if (Sx.ndim() == 3) {
        dim0 = Sx.shape(0);  
        dim1 = Sx.shape(1);  
        dim2 = Sx.shape(2);  
        output_shape = {Sx.shape(0), Sx.shape(1), Sx.shape(2)}; 
    } else {
        throw std::runtime_error("Arrays must be 2D or 3D");
    }

    SolverConfig.BoxGridPoints = { dim0, dim1, dim2 };
    VTK_Contents.BoxGridSize = SolverConfig.BoxGridPoints;

    // Compute total elements
    if (Sx.ndim() == 2) {
        SolverConfig.totalBoxElements = static_cast<long long>(dim0) * static_cast<long long>(dim1);
        VTK_Contents.totalBoxElements = SolverConfig.totalBoxElements;
    } else {
        SolverConfig.totalBoxElements = static_cast<long long>(dim0) * static_cast<long long>(dim1) * static_cast<long long>(dim2);
        VTK_Contents.totalBoxElements = SolverConfig.totalBoxElements;
    }

    // Grid delta handling
    varfloat3<varfloat> grid_delta = {1.0, 1.0, 1.0};

    if (delta.size() > 0) {
        auto ptr = (varfloat*)delta.mutable_data();
        size_t sz = delta.size();

        if (sz == 1) {
            grid_delta = {ptr[0], ptr[0], ptr[0]};
        }
        else if (Sx.ndim() == 2 && sz >= 2) {
            grid_delta = {ptr[0], ptr[1], 1.0};  
        }
        else if (Sx.ndim() == 3 && sz >= 3) {
            grid_delta = {ptr[0], ptr[1], ptr[2]};  
        }
        else {
            throw std::runtime_error("Invalid grid delta size.");
        }
    }

    SolverConfig.GridDelta = grid_delta;
    VTK_Contents.GridDelta = grid_delta;

    // Options handling
    if (!options.is_none()) {
        if (options.contains("Verbose")) {
            SolverConfig.Verbose = py::cast<bool>(options["Verbose"]);
            if (SolverConfig.Verbose) {
                cout << "SolverConfig.Verbose = TRUE" << endl;
                cout << "Welcome to OSMODI!" << endl;
            }
        }

        if (options.contains("SolverToleranceRel")) {
            SolverConfig.solverToleranceRel = py::cast<varfloat>(options["SolverToleranceRel"]);
            if (SolverConfig.Verbose) {
                cout << "SolverConfig.solverToleranceRel = " << SolverConfig.solverToleranceRel << endl;
            }
        }

        if (options.contains("SolverToleranceAbs")) {
            SolverConfig.solverToleranceAbs = py::cast<varfloat>(options["SolverToleranceAbs"]);
            if (SolverConfig.Verbose) {
                cout << "SolverConfig.solverToleranceAbs = " << SolverConfig.solverToleranceAbs << endl;
            }
        }

        if (options.contains("SolverDevice")) {
            string device = py::cast<string>(options["SolverDevice"]);
            if (iequals(device, "CPU")) {
                SolverConfig.SolverDevice = CPU;
                if (SolverConfig.Verbose) cout << "SolverConfig.SolverDevice = CPU" << endl;
            } else if (iequals(device, "GPU")) {
                SolverConfig.SolverDevice = GPU;
                if (SolverConfig.Verbose) cout << "This is the cpu solver. Use osmodi.solve_gpu for gpu solver" << endl;
                SolverConfig.SolverDevice = CPU;
            } else {
                throw std::runtime_error("SolverDevice must be either 'GPU' or 'CPU'.");
            }
        }

        if (options.contains("Kernel")) {
            string kernel = py::cast<string>(options["Kernel"]);
            if (iequals(kernel, "face-crossing")) {
                SolverConfig.Kernel = FACE_CROSSING;
                if (SolverConfig.Verbose) cout << "SolverConfig.Kernel = FACE_CROSSING" << endl;
            } else if (iequals(kernel, "cell-centered")) {
                SolverConfig.Kernel = CELL_CENTERED;
                if (SolverConfig.Verbose) cout << "SolverConfig.Kernel = CELL_CENTERED" << endl;
            } else {
                throw std::runtime_error("Kernel must be either 'face-crossing' or 'cell-centered'.");
            }
        }
    }

    // Calculate kernel constants
    if (Sx.ndim() == 2) {
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
    else if(Sx.ndim() == 3){
        if (SolverConfig.Kernel == FACE_CROSSING){
            varfloat cx = 1.0 * SolverConfig.GridDelta.y * SolverConfig.GridDelta.z;
            varfloat cy = 1.0 * SolverConfig.GridDelta.x * SolverConfig.GridDelta.z;
            varfloat cz = 1.0 * SolverConfig.GridDelta.x * SolverConfig.GridDelta.y;
            varfloat ctot = 2.0 * (cx + cy + cz); 
            SolverConfig.cx = cx / ctot;
            SolverConfig.cy = cy / ctot;
            SolverConfig.cz = cz / ctot;
        }
        else if (SolverConfig.Kernel == CELL_CENTERED){
            PrecomputeConstants(&SolverConfig);
        }
    }

    if (SolverConfig.Verbose){
        cout << "Input Box Size = [" << SolverConfig.BoxGridPoints.x << ", " 
             << SolverConfig.BoxGridPoints.y << ", " << SolverConfig.BoxGridPoints.z << "]" << endl;
        cout << "Total Box Size = " << SolverConfig.totalBoxElements << endl;
        cout << "Input Box Deltas = [" << SolverConfig.GridDelta.x << ", " 
             << SolverConfig.GridDelta.y << ", " << SolverConfig.GridDelta.z << "]" << endl;
    }

    // Allocate output arrays
    auto result = py::array_t<varfloat, py::array::f_style>(output_shape);

    varfloat* PressureField = (varfloat*)result.mutable_data();
    varfloat* RHS = new varfloat[VTK_Contents.totalBoxElements];

    // Assign source fields
    VTK_Contents.SourceFn_Field_X = (varfloat*)Sx.mutable_data();
    VTK_Contents.SourceFn_Field_Y = (varfloat*)Sy.mutable_data();
    VTK_Contents.SourceFn_Field_Z = (varfloat*)Sz.mutable_data();

    // for (int i =0;i< SolverConfig.totalBoxElements ; i++){
    //     printf("%f, ",VTK_Contents.SourceFn_Field_X[i]);
    // }

    // Initialize arrays
    FillBox(PressureField, ZEROS, SolverConfig);
    FillBox(RHS, ZEROS, SolverConfig);

    // Run solver
    if (SolverConfig.Verbose){
        cout << "========Starting CPU Solver...========" << endl;
    }
    
    ConjugateGradientSolver_CPU(PressureField, RHS, SolverConfig, VTK_Contents);

    delete[] RHS;

    if (SolverConfig.Verbose){
        cout << "OSMODI completed successfully!" << endl;
    }

    // Create progress matrix
    int nRows = CGS_Progress.size();
    py::array_t<varfloat> progress_matrix({nRows, 3});
    auto progress_ptr = (varfloat*)progress_matrix.mutable_data();
    
    for (int i = 0; i < nRows; i++){
        progress_ptr[i * 3 + 0] = (varfloat) CGS_Progress[i].Iteration;
        progress_ptr[i * 3 + 1] = (varfloat) CGS_Progress[i].Residual;
        progress_ptr[i * 3 + 2] = (varfloat) CGS_Progress[i].TimeSeconds;
    }

    // Return both pressure field and progress matrix
    return py::make_tuple(result, progress_matrix);
}

// PyBind11 Module 
PYBIND11_MODULE(osmodi_cpu_bind, m) {
    m.doc() = "OSMODI solver interface (CPU-only, float/double, 2D/3D)";

    m.def("solver", [](py::array Sx, py::array Sy, py::array Sz,
                       py::array delta = py::array(),
                       py::dict options = py::dict()) -> py::object
    {
        // Validate dtype consistency
        if (Sx.dtype().num() != Sy.dtype().num() ||
            Sx.dtype().num() != Sz.dtype().num())
            throw std::runtime_error("All inputs (Sx, Sy, Sz) must have the same dtype.");

        // Enforce float32 or float64 only
        if (Sx.dtype().kind() != 'f')
            throw std::runtime_error("Only float32 and float64 arrays are supported.");

        if (Sx.itemsize() == 4)
            return call_solver<float>(Sx.cast<py::array_t<float>>(),
                                      Sy.cast<py::array_t<float>>(),
                                      Sz.cast<py::array_t<float>>(),
                                      delta.cast<py::array_t<float>>(),
                                      options);
        else if (Sx.itemsize() == 8)
            return call_solver<double>(Sx.cast<py::array_t<double>>(),
                                       Sy.cast<py::array_t<double>>(),
                                       Sz.cast<py::array_t<double>>(),
                                       delta.cast<py::array_t<double>>(),
                                       options);
        else
            throw std::runtime_error("Unsupported float precision: only 32-bit or 64-bit allowed.");
    }, py::arg("Sx"), py::arg("Sy"), py::arg("Sz"), 
       py::arg("delta") = py::array(), 
       py::arg("options") = py::dict(),
       "Solve the omnidirectional pressure Poisson equation\n\n"
       "Parameters:\n"
       "  Sx, Sy, Sz: Source term arrays (2D or 3D)\n"
       "  delta: Grid spacing array\n"
       "  options: Dictionary with solver options\n"
       "    - Verbose: bool\n"
       "    - SolverToleranceRel: float\n"
       "    - SolverToleranceAbs: float\n"
       "    - SolverDevice: 'CPU' or 'GPU'\n"
       "    - Kernel: 'face-crossing' or 'cell-centered'\n");
}
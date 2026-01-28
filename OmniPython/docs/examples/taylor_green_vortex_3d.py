import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import osmodi

# Parameters
t = 0.1  # Time, s
H = 1e-6  # Vortex strength, m^2
nu = 1e-6  # Kinematic Viscosity, m^2/s
rho = 1000  # Density, kg/m^3

# Characteristic scales
L0 = np.sqrt(H)  # Characteristic Length, m
U0 = nu / np.sqrt(H)  # Characteristic Velocity, m/s
T0 = L0 / U0  # Characteristic Time, s
P0 = rho * U0**2  # Characteristic Pressure, Pa

# Grid setup
NptsX = 300
NptsY = 500
NptsZ = 250

x = np.linspace(-2*L0, 2*L0, NptsX)
y = np.linspace(-2*L0, 2*L0, NptsY)
z = np.linspace(-2*L0, 2*L0, NptsZ)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

print(f"Grid: {NptsX} x {NptsY} x {NptsZ}")
print(f"Grid spacing: dx={dx:.6e}, dy={dy:.6e}, dz={dz:.6e}")


X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Cylindrical coordinates 
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Velocity field 
UR = np.zeros_like(X)
UTheta = (H / (8*np.pi)) * (R / (nu * t**2)) * np.exp(-(R**2) / (4*nu*t))

# Pressure field
P = -(rho * H**2) / (64 * (np.pi**2) * nu * t**3) * np.exp(-(R**2) / (2*nu*t))
NormPmax = ((H**2) / (64 * (np.pi**2) * nu * t**3)) / (U0**2)

# Convert to Cartesian velocities
Ux = UR * np.cos(Theta) - UTheta * np.sin(Theta)
Uy = UR * np.sin(Theta) + UTheta * np.cos(Theta)
Uz = np.zeros_like(Ux)

# Time derivatives
dudt = np.zeros_like(Ux)
dvdt = np.zeros_like(Ux)
dwdt = np.zeros_like(Ux)

# Compute gradients
DUDX, DUDY, DUDZ = np.gradient(Ux, dx, dy, dz)
DVDX, DVDY, DVDZ = np.gradient(Uy, dx, dy, dz)
DWDX, DWDY, DWDZ = np.gradient(Uz, dx, dy, dz)

# Source terms 
Sx = (-rho * (dudt + Ux*DUDX + Uy*DUDY + Uz*DUDZ)) / (P0/L0)
Sy = (-rho * (dvdt + Ux*DVDX + Uy*DVDY + Uz*DVDZ)) / (P0/L0)
Sz = (-rho * (dwdt + Ux*DWDX + Uy*DWDY + Uz*DWDZ)) / (P0/L0)

# Solver options
options = {
    'SolverToleranceRel': 1e-4,
    'SolverToleranceAbs': 1e-6,
    'Verbose': True,
    'SolverDevice': 'GPU',
    'Kernel': 'cell-centered'
}

delta = np.array([dx/L0, dy/L0, dz/L0], dtype=np.float64)

print("="*60)
print("CALLING SOLVER")
print("="*60)

print("Starting OSMODI...")
start_time = time.time()
P_OSMODI, CGS = osmodi.solve_gpu(Sx, Sy, Sz, delta, options)
end_time = time.time()

print(f"\nTime Taken for computation: {end_time - start_time:.4f}s")
print(f"P_OSMODI.shape: {P_OSMODI.shape}")

# Remove corner value 
P_OSMODI = P_OSMODI - P_OSMODI[1, 1, 1]

# Ground truth normalized pressure
Ptruth = P / (rho * U0**2)

# Compute error
Err = (P_OSMODI - Ptruth) / NormPmax
ErrorPercent = 100 * np.sqrt(np.sum(Err**2) / Err.size)
print(f"\nError = {ErrorPercent:.3f}%")
# print(f"\nError =" , Err)

# Calculate mid point
mid_x = NptsX // 2
mid_y = NptsY // 2
mid_z = NptsZ // 2

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'3D Taylor Vortex - All Slices - Error: {ErrorPercent:.3f}%', fontsize=16)

# TOP ROW: Solver Results
# XY slice at mid-Z
im1 = axes[0, 0].imshow(P_OSMODI[:, :, mid_z].T, origin='lower', aspect='equal', cmap='RdBu_r')
axes[0, 0].set_title(f'Solver - XY slice (z={mid_z})')
axes[0, 0].set_xlabel('x index')
axes[0, 0].set_ylabel('y index')
plt.colorbar(im1, ax=axes[0, 0])

# XZ slice at mid-Y
im2 = axes[0, 1].imshow(P_OSMODI[:, mid_y, :].T, origin='lower', aspect='equal', cmap='RdBu_r')
axes[0, 1].set_title(f'Solver - XZ slice (y={mid_y})')
axes[0, 1].set_xlabel('x index')
axes[0, 1].set_ylabel('z index')
plt.colorbar(im2, ax=axes[0, 1])

# YZ slice at mid-X 
im3 = axes[0, 2].imshow(P_OSMODI[mid_x, :, :].T, origin='lower', aspect='equal', cmap='RdBu_r')
axes[0, 2].set_title(f'Solver - YZ slice (x={mid_x})')
axes[0, 2].set_xlabel('y index')
axes[0, 2].set_ylabel('z index')
plt.colorbar(im3, ax=axes[0, 2])

# BOTTOM ROW: Ground Truth 
# XY slice at mid-Z
im4 = axes[1, 0].imshow(Ptruth[:, :, mid_z].T, origin='lower', aspect='equal', cmap='RdBu_r')
axes[1, 0].set_title(f'Truth - XY slice (z={mid_z})')
axes[1, 0].set_xlabel('x index')
axes[1, 0].set_ylabel('y index')
plt.colorbar(im4, ax=axes[1, 0])

# XZ slice at mid-Y
im5 = axes[1, 1].imshow(Ptruth[:, mid_y, :].T, origin='lower', aspect='equal', cmap='RdBu_r')
axes[1, 1].set_title(f'Truth - XZ slice (y={mid_y})')
axes[1, 1].set_xlabel('x index')
axes[1, 1].set_ylabel('z index')
plt.colorbar(im5, ax=axes[1, 1])

# YZ slice at mid-X
im6 = axes[1, 2].imshow(Ptruth[mid_x, :, :].T, origin='lower', aspect='equal', cmap='RdBu_r')
axes[1, 2].set_title(f'Truth - YZ slice (x={mid_x})')
axes[1, 2].set_xlabel('y index')
axes[1, 2].set_ylabel('z index')
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('taylor_vortex_3d_all_slices.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'taylor_vortex_3d_all_slices.png'")
plt.show()

# Convergence plot
if CGS.shape[0] > 1:
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(CGS[:, 0], CGS[:, 1], 'b-', linewidth=2)
    ax.set_xlabel('CG Iteration')
    ax.set_ylabel('Relative Residual')
    ax.set_title('CG Solver Convergence')
    ax.grid(True, alpha=0.3)
    plt.savefig('taylor_vortex_3d_convergence.png', dpi=150)

plt.show()

print("\nDone!")
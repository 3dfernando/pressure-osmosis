import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import osmodi as oss

# ------------------------------
# Physical parameters
# ------------------------------
t = 0.2
dt = 0.001
H = 1e-6
nu = 1e-6
rho = 1000

L0 = np.sqrt(H)
U0 = nu / np.sqrt(H)
T0 = L0 / U0
P0 = rho * U0**2

# ------------------------------
# Grid 
# ------------------------------
NptsX = 250
NptsY = 400
DomainSize = 2

x = np.linspace(-DomainSize*L0, DomainSize*L0, NptsX)
y = np.linspace(-DomainSize*L0, DomainSize*L0, NptsY)
dx = x[1] - x[0]
dy = y[1] - y[0]

# ndgrid
X, Y = np.meshgrid(x, y, indexing='ij')

R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

UR = np.zeros_like(X)
UTheta = (H / (8*np.pi)) * (R / (nu * t**2)) * np.exp(-(R**2) / (4*nu*t))

P = -(rho * H**2) / (64 * (np.pi**2) * nu * t**3) * np.exp(-(R**2) / (2*nu*t))
NormPmax = ((H**2) / (64 * (np.pi**2) * nu * t**3)) / (U0**2)

Ux = UR * np.cos(Theta) - UTheta * np.sin(Theta)
Uy = UR * np.sin(Theta) + UTheta * np.cos(Theta)

dudt = np.zeros_like(Ux)
dvdt = np.zeros_like(Ux)

# Compute gradients
DUDX, DUDY = np.gradient(Ux, dx, dy)  
DVDX, DVDY = np.gradient(Uy, dx, dy)

# Source terms
Sx = (-rho * (dudt + Ux*DUDX + Uy*DUDY)) / (P0/L0)
Sy = (-rho * (dvdt + Ux*DVDX + Uy*DVDY)) / (P0/L0)
Sz = np.zeros_like(Sx)

# ------------------------------
# Plot Gradients and Source Terms
# ------------------------------
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle('Velocity Gradients', fontsize=14)

im1 = axes1[0, 0].imshow(DUDX.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes1[0, 0].set_title('DUDX (∂u/∂x)')
axes1[0, 0].set_xlabel('x')
axes1[0, 0].set_ylabel('y')
plt.colorbar(im1, ax=axes1[0, 0])

im2 = axes1[0, 1].imshow(DUDY.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes1[0, 1].set_title('DUDY (∂u/∂y)')
axes1[0, 1].set_xlabel('x')
axes1[0, 1].set_ylabel('y')
plt.colorbar(im2, ax=axes1[0, 1])

im3 = axes1[0, 2].imshow(Ux.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes1[0, 2].set_title('Ux')
axes1[0, 2].set_xlabel('x')
axes1[0, 2].set_ylabel('y')
plt.colorbar(im3, ax=axes1[0, 2])

im4 = axes1[1, 0].imshow(DVDX.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes1[1, 0].set_title('DVDX (∂v/∂x)')
axes1[1, 0].set_xlabel('x')
axes1[1, 0].set_ylabel('y')
plt.colorbar(im4, ax=axes1[1, 0])

im5 = axes1[1, 1].imshow(DVDY.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes1[1, 1].set_title('DVDY (∂v/∂y)')
axes1[1, 1].set_xlabel('x')
axes1[1, 1].set_ylabel('y')
plt.colorbar(im5, ax=axes1[1, 1])

im6 = axes1[1, 2].imshow(Uy.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes1[1, 2].set_title('Uy')
axes1[1, 2].set_xlabel('x')
axes1[1, 2].set_ylabel('y')
plt.colorbar(im6, ax=axes1[1, 2])

plt.tight_layout()
plt.savefig('velocity_gradients.png', dpi=150)
print("Velocity gradients plot saved")

# ------------------------------
# Plot Source Terms
# ------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Source Terms', fontsize=14)

im7 = axes2[0].imshow(Sx.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes2[0].set_title('Sx')
axes2[0].set_xlabel('x')
axes2[0].set_ylabel('y')
plt.colorbar(im7, ax=axes2[0])

im8 = axes2[1].imshow(Sy.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes2[1].set_title('Sy')
axes2[1].set_xlabel('x')
axes2[1].set_ylabel('y')
plt.colorbar(im8, ax=axes2[1])

plt.tight_layout()
plt.savefig('source_terms.png', dpi=150)
print("Source terms plot saved")

# ------------------------------
# Solver
# ------------------------------
print("ARRAY SHAPES AND PROPERTIES")
print("=" * 60)
print(f"Sx.shape: {Sx.shape}")
print(f"X.shape: {X.shape}")

options = {
    'SolverToleranceRel': 1e-4,
    'SolverToleranceAbs': 1e-6,
    'Verbose': True,
    'SolverDevice': 'GPU',
    'Kernel': 'cell-centered'
}

delta = np.array([dx/L0, dy/L0], dtype=np.float64)

print("="*60)
print("CALLING SOLVER")
print("="*60)

P_OSMODI, CGS = oss.solve_gpu(Sx, Sy, Sz, delta, options)

print(f"\nP_OSMODI.shape: {P_OSMODI.shape}")
print(f"P.shape: {P.shape}")

# Remove corner value
P_OSMODI = P_OSMODI - P_OSMODI[1, 1]
Ptruth = P / (rho * U0**2)

Err1 = (P_OSMODI - Ptruth) / NormPmax
ErrorPercent1 = 100 * np.sqrt(np.sum(Err1**2) / Err1.size)
print(f"Error: {ErrorPercent1:.3f}%")

# ------------------------------
# Plot Results
# ------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle(f'Pressure Solution - Error: {ErrorPercent1:.3f}%', fontsize=14)

im9 = axes3[0].imshow(P_OSMODI.T, origin='lower', aspect='equal')
axes3[0].set_title('Solver')
axes3[0].set_xlabel('x')
axes3[0].set_ylabel('y')
plt.colorbar(im9, ax=axes3[0])

im10 = axes3[1].imshow(Ptruth.T, origin='lower', aspect='equal')
axes3[1].set_title('Truth')
axes3[1].set_xlabel('x')
axes3[1].set_ylabel('y')
plt.colorbar(im10, ax=axes3[1])

im11 = axes3[2].imshow(Err1.T, origin='lower', aspect='equal', cmap='RdBu_r')
axes3[2].set_title(f'Error')
axes3[2].set_xlabel('x')
axes3[2].set_ylabel('y')
plt.colorbar(im11, ax=axes3[2])

plt.tight_layout()
plt.savefig('taylor_vortex_2d.png', dpi=150)
print("\nPressure solution plot saved as 'taylor_vortex_2d.png'")

plt.show()



































































































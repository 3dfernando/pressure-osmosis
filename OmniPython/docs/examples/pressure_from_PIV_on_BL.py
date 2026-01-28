"""
TRIPV Boundary Layer Example
Demonstrates OSMODI solver on real PIV experimental data
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os
import osmodi

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory
output_dir = os.path.join(script_dir, 'TRIPV_BL_output')
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading TRIPV boundary layer data...")
data_path = os.path.join(script_dir, 'TRPIV_BL_Sample.mat')
reference_path = os.path.join(script_dir, 'P.mat')

data = scipy.io.loadmat(data_path)
reference_data = scipy.io.loadmat(reference_path)

U = data['U']
V = data['V']
P_truth = reference_data['P']
dx = float(data['dx'][0, 0])
dy = float(data['dy'][0, 0])
dt = float(data['dt'][0, 0])

print(f"Data shape: {U.shape}")
nx, ny, nt = U.shape

# Compute spatial derivatives
print("Computing velocity gradients...")
dUdx = np.zeros_like(U)
dUdy = np.zeros_like(U)
dVdx = np.zeros_like(V)
dVdy = np.zeros_like(V)

for t in range(nt):
    U_t = U[:, :, t].T
    V_t = V[:, :, t].T
    dU_dy, dU_dx = np.gradient(U_t, dx, dy)
    dV_dy, dV_dx = np.gradient(V_t, dx, dy)
    dUdx[:, :, t] = dU_dx.T
    dUdy[:, :, t] = dU_dy.T
    dVdx[:, :, t] = dV_dx.T
    dVdy[:, :, t] = dV_dy.T

# Temporal derivatives
dUdt = np.zeros_like(U)
dVdt = np.zeros_like(V)
for x in range(nx):
    for y in range(ny):
        dUdt[x, y, :] = np.gradient(U[x, y, :], dt)
        dVdt[x, y, :] = np.gradient(V[x, y, :], dt)

# Second derivatives for viscous terms
mu = 18e-6  # Dynamic viscosity
d2Udx2 = np.zeros_like(U)
d2Udy2 = np.zeros_like(U)
d2Vdx2 = np.zeros_like(V)
d2Vdy2 = np.zeros_like(V)

for t in range(nt):
    dUdx_t = dUdx[:, :, t].T
    dUdy_t = dUdy[:, :, t].T
    dVdx_t = dVdx[:, :, t].T
    dVdy_t = dVdy[:, :, t].T
    _, d2U_dx2 = np.gradient(dUdx_t, dx, dy)
    d2U_dy2, _ = np.gradient(dUdy_t, dx, dy)
    _, d2V_dx2 = np.gradient(dVdx_t, dx, dy)
    d2V_dy2, _ = np.gradient(dVdy_t, dx, dy)
    d2Udx2[:, :, t] = d2U_dx2.T
    d2Udy2[:, :, t] = d2U_dy2.T
    d2Vdx2[:, :, t] = d2V_dx2.T
    d2Vdy2[:, :, t] = d2V_dy2.T

# Calculate pressure gradient source terms from Navier-Stokes
rho = 1.2  # Air density
dPdx = -rho * (dUdt + U * dUdx + V * dUdy) + mu * (d2Udx2 + d2Udy2)
dPdy = -rho * (dVdt + U * dVdx + V * dVdy) + mu * (d2Vdx2 + d2Vdy2)

# Match NaN masks
nanMask = np.isnan(dPdx) | np.isnan(dPdy)
dPdx[nanMask] = np.nan
dPdy[nanMask] = np.nan

print(f"Valid measurement points: {np.sum(~nanMask[:,:,0])} out of {nx*ny}")

# Solve for pressure using OSMODI
print("\nSolving for pressure field...")

solver_options = {
    'SolverToleranceRel': 1e-4,
    'SolverToleranceAbs': 1e-4,
    'Kernel': 'cell-centered',
    'Verbose': True
}

delta = np.array([dx, dy], dtype=np.float64)
P = np.zeros((nx, ny, nt))

# Reference point for normalization
xinf = 236
yinf = 162
Uinf = np.sqrt(np.nanmean(U[xinf, yinf, :])**2 + np.nanmean(V[xinf, yinf, :])**2)
qinf = 0.5 * rho * Uinf**2

print(f"Reference velocity: {Uinf:.6f} m/s")
print(f"Reference dynamic pressure: {qinf:.6f} Pa")

# Process each time frame
start_time = time.time()

# Choose solver based on availability
if osmodi.GPU_AVAILABLE:
    print("Using GPU solver")
    solver_func = osmodi.solve_gpu
else:
    print("Using CPU solver")
    solver_func = osmodi.solve_cpu

for i in range(nt):
    # Prepare source terms for this frame
    dPdx_frame = np.asfortranarray(dPdx[:, :, i])
    dPdy_frame = np.asfortranarray(dPdy[:, :, i])
    ones_array = np.ones_like(dPdx_frame, dtype=np.float64)
    ones_array[nanMask[:, :, i]] = np.nan
    ones_array = np.asfortranarray(ones_array)
    
    # Solve
    p, progress = solver_func(dPdx_frame, dPdy_frame, ones_array, delta, solver_options)
    
    P[:, :, i] = p
    print(f"Frame {i+1}/{nt} completed")

elapsed = time.time() - start_time
print(f"\nTotal processing time: {elapsed:.2f}s ({elapsed/nt:.3f}s per frame)")

# Calculate pressure coefficient
Cp = P / qinf
Cp_truth = P_truth / qinf

# Remove mean at reference point
for i in range(nt):
    Cp[:, :, i] = Cp[:, :, i] - Cp[xinf, yinf, i]
    Cp_truth[:, :, i] = Cp_truth[:, :, i] - Cp_truth[xinf, yinf, i]

# Validation
valid_points = ~np.isnan(Cp) & ~np.isnan(Cp_truth)
diff = np.abs(Cp[valid_points] - Cp_truth[valid_points])

print(f"\nValidation Results:")
print(f"Mean absolute error: {np.mean(diff):.6f}")
print(f"Max absolute error: {np.max(diff):.6f}")
print(f"Valid points analyzed: {np.sum(valid_points)}")

# Visualization - First frame comparison
print("\nGenerating validation plot...")
fig = plt.figure(figsize=(15, 5))

ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(Cp[:, :, 0].T, origin='lower', vmin=-0.4, vmax=0.4, cmap='jet', aspect='equal')
ax1.set_title('OSMODI Computed Pressure')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im1, ax=ax1, label='Cp')

ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(Cp_truth[:, :, 0].T, origin='lower', vmin=-0.4, vmax=0.4, cmap='jet', aspect='equal')
ax2.set_title('Reference Pressure')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(im2, ax=ax2, label='Cp')

ax3 = plt.subplot(1, 3, 3)
error = Cp[:, :, 0] - Cp_truth[:, :, 0]
im3 = ax3.imshow(error.T, origin='lower', vmin=-0.1, vmax=0.1, cmap='RdBu_r', aspect='equal')
ax3.set_title('Error (Computed - Reference)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
plt.colorbar(im3, ax=ax3, label='Error')

plt.suptitle('TRIPV Boundary Layer: OSMODI Validation (Frame 1)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'validation_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResults saved to: {output_dir}")
print("Done!")
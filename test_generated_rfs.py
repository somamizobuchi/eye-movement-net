from utils import hex_grid_torch, generate_rgc_spatial_rf
import matplotlib.pyplot as plt
import numpy as np
import math

pts = hex_grid_torch(32, 32, 5.0, "cpu")
pts = pts.numpy()

# Create kernels for each point
n_points = pts.shape[0]
n_cols = int(math.ceil(math.sqrt(n_points)))
n_rows = int(math.ceil(n_points / n_cols))

# First pass: generate all kernels to find global color limits
kernels = []
for pt in pts:
    # Use the point as the center within the 32x32 frame
    cx, cy = float(pt[0]), float(pt[1])
    _, _, kernel = generate_rgc_spatial_rf(
        cell_type="P", eccentricity=0.3, ppd=120.0, size_pixels=32, center=(cx, cy)
    )
    kernels.append(kernel)

# Calculate global color limits
all_kernels = np.array(kernels)
vmax = np.abs(all_kernels).max()
vmin = -vmax

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

for i, (ax, pt, kernel) in enumerate(zip(axes, pts, kernels)):
    # Plot the kernel with global color limits
    im = ax.imshow(kernel, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.axis("off")

# Hide extra subplots
for ax in axes[n_points:]:
    ax.axis("off")

plt.tight_layout(pad=0.1)

# Create a second figure with scatter plot of all centers
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(pts[:, 0], pts[:, 1], s=100, alpha=0.6, edgecolors="black")
ax2.set_xlim(-2, 34)
ax2.set_ylim(-2, 34)
ax2.set_aspect("equal")
ax2.invert_yaxis()  # Invert y-axis to match image coordinates
plt.tight_layout(pad=0.1)

plt.show()

#!/usr/bin/env python3
"""k-NN Decision Boundary Visualization for AML"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import make_interp_spline

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(10, 6.2))

# Class A data points (unauffällig - normal transactions)
class_a_x = np.array([1.2, 2.0, 3.5, 1.8, 0.8, 2.5])
class_a_y = np.array([1.5, 3.0, 2.5, 0.8, 3.5, 1.2])

# Class B data points (auffällig - suspicious transactions)
class_b_x = np.array([6.5, 7.8, 8.5, 6.0, 7.2, 8.0])
class_b_y = np.array([6.0, 8.0, 7.0, 7.5, 5.5, 9.0])

# Decision boundary points (smooth curve) - sorted by x
boundary_x = np.array([3.8, 4.5, 4.9, 5.3])
boundary_y = np.array([7.2, 3.0, 4.8, 0.0])

# Create smooth boundary curve
x_smooth = np.linspace(boundary_x.min(), boundary_x.max(), 300)
spl = make_interp_spline(boundary_x, boundary_y, k=3)
y_smooth = spl(x_smooth)

# Fill class regions with semi-transparent colors
# Region A (left side - normal)
region_a_x = [0, 0, 3.8] + x_smooth.tolist() + [5.3, 4.5, 0]
region_a_y = [0, 10, 7.2] + y_smooth.tolist() + [0, 0, 0]
ax.fill(region_a_x, region_a_y, color='#4DBBD5', alpha=0.15, zorder=0)

# Region B (right side - suspicious)
region_b_x = [10, 10, 10, 5.3] + x_smooth[::-1].tolist() + [3.8, 0, 0]
region_b_y = [0, 10, 10, 0] + y_smooth[::-1].tolist() + [7.2, 10, 10]
ax.fill(region_b_x, region_b_y, color='#E64B35', alpha=0.15, zorder=0)

# Plot decision boundary
ax.plot(x_smooth, y_smooth, 'r-', linewidth=2.5, label='Decision Boundary', zorder=3)

# Plot data points
ax.scatter(class_a_x, class_a_y, c='#4DBBD5', marker='o', s=80, 
           edgecolors='black', linewidths=1, label='Unauffällig', zorder=4)
ax.scatter(class_b_x, class_b_y, c='#E64B35', marker='^', s=100, 
           edgecolors='black', linewidths=1, label='Auffällig', zorder=4)

# New transaction point
new_tx_x, new_tx_y = 5.2, 5.0
ax.scatter(new_tx_x, new_tx_y, c='black', marker='x', s=200, 
           linewidths=3, label='Neue Tx', zorder=5)

# k-neighborhood circle
k_circle = Circle((new_tx_x, new_tx_y), 1.45, fill=False, 
                  edgecolor='black', linestyle='--', linewidth=1.5, zorder=2)
ax.add_patch(k_circle)

# Add dummy for legend
ax.plot([], [], 'k--', linewidth=1.5, label='k-Umgebung')

# Labels and limits
ax.set_xlabel('Feature 1 (z.B. Transaktionshöhe)', fontsize=11)
ax.set_ylabel('Feature 2 (z.B. Häufigkeit)', fontsize=11)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

# Tight layout
plt.tight_layout()

# Save as PDF
plt.savefig('knn_decision_boundary.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("✓ k-NN plot saved to knn_decision_boundary.pdf")

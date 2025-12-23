#!/usr/bin/env python3
"""k-Means Voronoi Diagram Visualization for AML"""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(10, 6.2))

# Cluster 1 data points (low-risk transactions)
cluster1_x = np.array([1.0, 2.5, 1.5, 3.0, 2.0, 2.8])
cluster1_y = np.array([1.0, 2.0, 2.5, 1.2, 0.8, 3.0])

# Cluster 2 data points (high-risk transactions)
cluster2_x = np.array([7.0, 8.5, 7.5, 6.5, 8.0, 9.0])
cluster2_y = np.array([7.0, 8.5, 6.0, 9.0, 7.8, 6.5])

# Centroids
centroid1 = (2.2, 1.6)
centroid2 = (7.7, 7.4)

# Decision boundary (perpendicular bisector)
# Line equation: y = -1.0*x + 10.6
x_boundary = np.array([0, 10])
y_boundary = -1.0 * x_boundary + 10.6

# Fill Voronoi regions with semi-transparent colors
# Cluster 1 region (bottom-left, below decision line)
region1_x = [0, 0, 10, 5]
region1_y = [0, 10.6, 0.6, 5.6]
ax.fill(region1_x, region1_y, color='#00A087', alpha=0.12, zorder=0)

# Cluster 2 region (top-right, above decision line)
region2_x = [0, 5, 10, 10]
region2_y = [10.6, 5.6, 0.6, 10]
ax.fill(region2_x, region2_y, color='#3C5488', alpha=0.12, zorder=0)

# Plot decision boundary
ax.plot(x_boundary, y_boundary, 'r-', linewidth=2.5, label='Decision Boundary', zorder=3)

# Plot data points
ax.scatter(cluster1_x, cluster1_y, c='#00A087', marker='o', s=80, 
           edgecolors='black', linewidths=1, label='Cluster 1', zorder=4)
ax.scatter(cluster2_x, cluster2_y, c='#3C5488', marker='o', s=80, 
           edgecolors='black', linewidths=1, label='Cluster 2', zorder=4)

# Plot centroids
ax.scatter([centroid1[0]], [centroid1[1]], c='#E64B35', marker='x', s=250, 
           linewidths=4, label='Zentroid', zorder=5)
ax.scatter([centroid2[0]], [centroid2[1]], c='#E64B35', marker='x', s=250, 
           linewidths=4, zorder=5)

# Labels and limits
ax.set_xlabel('Feature 1 (z.B. Betrag)', fontsize=11)
ax.set_ylabel('Feature 2 (z.B. Land-Risiko-Score)', fontsize=11)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

# Tight layout
plt.tight_layout()

# Save as PDF
plt.savefig('kmeans_voronoi.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("✓ k-Means plot saved to kmeans_voronoi.pdf")

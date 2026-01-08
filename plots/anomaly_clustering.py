"""
Visualization of Anomaly Detection algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# FMA Colors
FMA_ORANGE = '#EC6600'
FMA_BLUE = '#144F76'
GREEN = '#27AE60'
RED = '#E74C3C'

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ============ LEFT: Isolation Forest ============
ax = axes[0]

# Generate normal data (cluster)
n_normal = 150
X_normal = np.random.randn(n_normal, 2) * 0.8 + np.array([2, 2])

# Add some anomalies (outliers)
n_anomaly = 15
X_anomaly = np.random.uniform(low=-1, high=5, size=(n_anomaly, 2))

X = np.vstack([X_normal, X_anomaly])

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso_forest.fit_predict(X)

# Create mesh for decision function
x_min, x_max = -1.5, 5.5
y_min, y_max = -1.5, 5.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = iso_forest.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot decision boundary
ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.5)
ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='--')

# Plot points
ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c=GREEN, s=40, alpha=0.7, 
           label='Normal', edgecolors='white', linewidth=0.5)
ax.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c=RED, s=80, marker='X',
           label='Anomalie', edgecolors='black', linewidth=1)

ax.set_title('Isolation Forest\nFindet Ausreißer automatisch', fontsize=11, fontweight='bold', color=FMA_ORANGE)
ax.set_xlabel('Feature 1 (z.B. Transaktionshöhe)')
ax.set_ylabel('Feature 2 (z.B. Häufigkeit)')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)

# Add annotation
ax.annotate('Anomalie-\nGrenze', xy=(0, 0.5), xytext=(-0.5, 2),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='black'))

# ============ RIGHT: k-Means Clustering ============
ax = axes[1]

# Generate clustered data (3 customer segments)
np.random.seed(42)
n_per_cluster = 50
cluster1 = np.random.randn(n_per_cluster, 2) * 0.5 + np.array([1, 1])  # Low risk
cluster2 = np.random.randn(n_per_cluster, 2) * 0.6 + np.array([4, 1.5])  # Medium risk
cluster3 = np.random.randn(n_per_cluster, 2) * 0.4 + np.array([2.5, 4])  # High risk

X_clusters = np.vstack([cluster1, cluster2, cluster3])

# Fit k-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_clusters = kmeans.fit_predict(X_clusters)
centers = kmeans.cluster_centers_

# Plot
colors = [GREEN, FMA_BLUE, FMA_ORANGE]
labels = ['Segment A\n(Niedrig-Risiko)', 'Segment B\n(Standard)', 'Segment C\n(Erhöht)']

for i in range(3):
    mask = y_clusters == i
    ax.scatter(X_clusters[mask, 0], X_clusters[mask, 1], c=colors[i], s=40, alpha=0.7,
               edgecolors='white', linewidth=0.5)

# Plot centroids
for i, (center, label) in enumerate(zip(centers, labels)):
    ax.scatter(center[0], center[1], c=colors[i], s=200, marker='*', 
               edgecolors='black', linewidth=1.5, zorder=10)
    ax.annotate(label, xy=center, xytext=(center[0]+0.3, center[1]+0.4),
                fontsize=8, fontweight='bold', color=colors[i])

# Draw cluster boundaries (Voronoi-like)
x_min, x_max = 0, 5.5
y_min, y_max = 0, 5.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z, colors='gray', linewidths=1.5, linestyles='--', alpha=0.7)

ax.set_title('k-Means Clustering\nKundensegmentierung', fontsize=11, fontweight='bold', color=FMA_ORANGE)
ax.set_xlabel('Feature 1 (z.B. Umsatz)')
ax.set_ylabel('Feature 2 (z.B. Auslandstransaktionen)')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anomaly_clustering.pdf', bbox_inches='tight', dpi=150)
plt.savefig('anomaly_clustering.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved anomaly_clustering.pdf")

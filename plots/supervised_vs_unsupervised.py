"""
Visualization comparing Supervised vs Unsupervised Learning
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification

# FMA Colors
FMA_ORANGE = '#EC6600'
FMA_BLUE = '#144F76'

# Set random seed
np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Supervised Learning - Classification with labels
X_sup, y_sup = make_classification(n_samples=100, n_features=2, n_informative=2,
                                    n_redundant=0, n_clusters_per_class=1,
                                    class_sep=1.5, random_state=42)

ax = axes[0]
colors = [FMA_ORANGE, FMA_BLUE]
labels = ['Verdächtig', 'Normal']
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y_sup == i
    ax.scatter(X_sup[mask, 0], X_sup[mask, 1], c=color, s=50, alpha=0.7, 
               label=label, edgecolors='white', linewidth=0.5)

ax.set_title('Überwachtes Lernen\n(mit Labels)', fontsize=12, fontweight='bold', color=FMA_ORANGE)
ax.set_xlabel('Feature 1: Betrag')
ax.set_ylabel('Feature 2: Frequenz')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Unsupervised Learning - Same data without labels
X_unsup, _ = make_blobs(n_samples=100, centers=3, n_features=2, 
                        cluster_std=1.0, random_state=42)

ax = axes[1]
ax.scatter(X_unsup[:, 0], X_unsup[:, 1], c=FMA_BLUE, s=50, alpha=0.7,
           edgecolors='white', linewidth=0.5)
ax.set_title('Unüberwachtes Lernen\n(ohne Labels)', fontsize=12, fontweight='bold', color=FMA_ORANGE)
ax.set_xlabel('Feature 1: Betrag')
ax.set_ylabel('Feature 2: Frequenz')
ax.annotate('Cluster?', xy=(1, 3), fontsize=10, style='italic', color='#7F8C8D')
ax.annotate('Cluster?', xy=(-5, -7), fontsize=10, style='italic', color='#7F8C8D')
ax.annotate('Cluster?', xy=(-7, 1), fontsize=10, style='italic', color='#7F8C8D')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('supervised_vs_unsupervised.pdf', bbox_inches='tight', dpi=150)
plt.savefig('supervised_vs_unsupervised.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved supervised_vs_unsupervised.pdf")

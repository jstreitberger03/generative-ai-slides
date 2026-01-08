"""
Visualization of key ML algorithms for classification
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import LinearSegmentedColormap

# FMA Colors
FMA_ORANGE = '#EC6600'
FMA_BLUE = '#144F76'
GREEN = '#27AE60'

np.random.seed(42)

# Generate more complex data (two moons-like)
n = 80
theta1 = np.linspace(0, np.pi, n//2)
theta2 = np.linspace(0, np.pi, n//2)
X_class0 = np.vstack([np.cos(theta1), np.sin(theta1)]).T + np.random.randn(n//2, 2) * 0.15
X_class1 = np.vstack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5]).T + np.random.randn(n//2, 2) * 0.15
X = np.vstack([X_class0, X_class1])
y = np.array([0]*(n//2) + [1]*(n//2))

# Custom colormap
fma_cmap = LinearSegmentedColormap.from_list('fma', [FMA_ORANGE, '#F5F5F0', FMA_BLUE])

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

classifiers = [
    ('Decision Tree\n(Einzelner Baum, neigt zu Overfitting)', 
     DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('Random Forest\n(Mehrere Bäume → robuster)', 
     RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)),
    ('Gradient Boosting\n(Sequentiell, State-of-the-Art)', 
     GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
    ('k-Nearest Neighbors\n(Instanzbasiert, k=5)', 
     KNeighborsClassifier(n_neighbors=5))
]

for ax, (name, clf) in zip(axes.flat, classifiers):
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Try to get probability for smoother visualization
    try:
        Z_prob = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z_prob, alpha=0.4, cmap=fma_cmap, levels=20)
    except:
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=fma_cmap)
    
    ax.contour(xx, yy, Z, colors='gray', linewidths=1, alpha=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c=FMA_ORANGE, s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.scatter(X[y==1, 0], X[y==1, 1], c=FMA_BLUE, s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax.set_title(name, fontsize=10, fontweight='bold', color=FMA_ORANGE)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

plt.suptitle('Vergleich: Entscheidungsgrenzen verschiedener Algorithmen', 
             fontsize=13, fontweight='bold', color=FMA_ORANGE, y=0.98)
plt.tight_layout()
plt.savefig('algorithm_comparison.pdf', bbox_inches='tight', dpi=150)
plt.savefig('algorithm_comparison.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved algorithm_comparison.pdf")

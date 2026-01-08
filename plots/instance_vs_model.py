"""
Visualization comparing Instance-based vs Model-based Learning
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# Generate simple data
n = 50
X_class0 = np.random.randn(n, 2) + np.array([0, 0])
X_class1 = np.random.randn(n, 2) + np.array([3, 3])
X = np.vstack([X_class0, X_class1])
y = np.array([0]*n + [1]*n)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Instance-based: k-NN
ax = axes[0]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
ax.scatter(X[y==0, 0], X[y==0, 1], c='#E74C3C', s=40, alpha=0.7, label='Verdächtig')
ax.scatter(X[y==1, 0], X[y==1, 1], c='#27AE60', s=40, alpha=0.7, label='Normal')

# Highlight new point and neighbors
new_point = np.array([[1.5, 2]])
ax.scatter(new_point[0, 0], new_point[0, 1], c='#9B59B6', s=150, marker='*', 
           edgecolors='black', linewidth=1.5, label='Neu', zorder=10)

ax.set_title('Instanzbasiert (k-NN)\nSpeichert alle Trainingsdaten', fontsize=11, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Model-based: Logistic Regression
ax = axes[1]
lr = LogisticRegression()
lr.fit(X, y)
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
ax.scatter(X[y==0, 0], X[y==0, 1], c='#E74C3C', s=40, alpha=0.7, label='Verdächtig')
ax.scatter(X[y==1, 0], X[y==1, 1], c='#27AE60', s=40, alpha=0.7, label='Normal')

# Draw decision boundary line
w = lr.coef_[0]
b = lr.intercept_[0]
x_line = np.linspace(x_min, x_max, 100)
y_line = -(w[0] * x_line + b) / w[1]
ax.plot(x_line, y_line, 'k--', linewidth=2, label='Trennlinie')

ax.scatter(new_point[0, 0], new_point[0, 1], c='#9B59B6', s=150, marker='*', 
           edgecolors='black', linewidth=1.5, label='Neu', zorder=10)

ax.set_title('Modellbasiert (Log. Regression)\nLernt Parameter θ', fontsize=11, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('instance_vs_model.pdf', bbox_inches='tight', dpi=150)
plt.savefig('instance_vs_model.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved instance_vs_model.pdf")

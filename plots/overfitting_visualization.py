"""
Visualization of Underfitting, Good Fit, and Overfitting
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# FMA Colors
FMA_ORANGE = '#EC6600'
FMA_BLUE = '#144F76'

# Generate data
np.random.seed(42)
n_samples = 30
X = np.sort(np.random.rand(n_samples) * 10)
y = np.sin(X * 0.5) * 3 + np.random.randn(n_samples) * 0.8

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
X_train = X.reshape(-1, 1)

titles = ['Underfitting\n(Hoher Bias)', 'Gute Anpassung', 'Overfitting\n(Hohe Varianz)']
degrees = [1, 4, 15]
colors = [FMA_ORANGE, '#27AE60', FMA_ORANGE]

for ax, degree, title, color in zip(axes, degrees, titles, colors):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y)
    y_plot = model.predict(X_plot)
    
    ax.scatter(X, y, color=FMA_BLUE, s=40, alpha=0.7, label='Daten', zorder=5)
    ax.plot(X_plot, y_plot, color=color, linewidth=2.5, label=f'Modell (Grad {degree})')
    ax.set_title(title, fontsize=12, fontweight='bold', color=FMA_ORANGE)
    ax.set_xlabel('Feature X')
    ax.set_ylabel('Zielvariable Y')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('overfitting_visualization.pdf', bbox_inches='tight', dpi=150)
plt.savefig('overfitting_visualization.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved overfitting_visualization.pdf")

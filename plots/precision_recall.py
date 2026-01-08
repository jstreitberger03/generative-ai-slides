"""
Visualization of Precision-Recall Trade-off
"""
import numpy as np
import matplotlib.pyplot as plt

# FMA Colors
FMA_ORANGE = '#EC6600'
FMA_BLUE = '#144F76'

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: Confusion Matrix visualization
ax = axes[0]
matrix = np.array([[85, 15], [5, 895]])  # TP, FN, FP, TN for AML scenario
labels = [['TP\n(Erkannt)\n85', 'FN\n(Übersehen)\n15'],
          ['FP\n(Fehlalarm)\n5', 'TN\n(Korrekt neg.)\n895']]
colors = [['#27AE60', FMA_ORANGE], ['#F39C12', FMA_BLUE]]

for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, facecolor=colors[i][j], alpha=0.6))
        ax.text(j+0.5, 1.5-i, labels[i][j], ha='center', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(['Pred: Positiv', 'Pred: Negativ'])
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(['Tatsächlich: Negativ', 'Tatsächlich: Positiv'])
ax.set_title('Konfusionsmatrix (AML-Beispiel)', fontsize=12, fontweight='bold', color=FMA_ORANGE)

# Add metrics
precision = 85 / (85 + 5)
recall = 85 / (85 + 15)
f1 = 2 * precision * recall / (precision + recall)
ax.text(1, -0.3, f'Precision: {precision:.1%}  |  Recall: {recall:.1%}  |  F1: {f1:.2f}', 
        ha='center', fontsize=10, style='italic')

# Right: Precision-Recall curve
ax = axes[1]
thresholds = np.linspace(0, 1, 100)
# Simulated precision-recall curve
recall_vals = 1 - thresholds**0.5
precision_vals = 0.3 + 0.65 * thresholds**0.8

ax.plot(recall_vals, precision_vals, color=FMA_ORANGE, linewidth=2.5, label='Modell')
ax.fill_between(recall_vals, precision_vals, alpha=0.2, color=FMA_ORANGE)

# Mark different operating points
ax.scatter([0.85], [0.94], c='#27AE60', s=100, zorder=5, label='Hohe Precision\n(wenig FP)')
ax.scatter([0.95], [0.45], c=FMA_ORANGE, s=100, zorder=5, label='Hoher Recall\n(wenig FN)')
ax.scatter([0.90], [0.75], c=FMA_BLUE, s=100, zorder=5, label='Balanced')

ax.set_xlabel('Recall (Sensitivität)', fontsize=11)
ax.set_ylabel('Precision (Präzision)', fontsize=11)
ax.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold', color=FMA_ORANGE)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc='lower left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('precision_recall.pdf', bbox_inches='tight', dpi=150)
plt.savefig('precision_recall.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved precision_recall.pdf")

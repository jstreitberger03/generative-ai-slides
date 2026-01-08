"""
Visualization of Precision-Recall Trade-off - Intuitive Version
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# FMA Colors
FMA_ORANGE = '#EC6600'
FMA_BLUE = '#144F76'
GREEN = '#27AE60'
RED = '#E74C3C'
YELLOW = '#F1C40F'
GRAY = '#95A5A6'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ============ LEFT: Visual explanation of Precision & Recall ============
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(5, 7.5, 'Was bedeutet Precision & Recall?', ha='center', fontsize=13, 
        fontweight='bold', color=FMA_ORANGE)

# Legend for icons
ax.add_patch(Circle((0.5, 6.8), 0.2, color=RED, ec='black', lw=1))
ax.text(0.9, 6.8, '= Echter Geldwäschefall', va='center', fontsize=9)
ax.add_patch(Circle((5.5, 6.8), 0.2, color=GRAY, ec='black', lw=1))
ax.text(5.9, 6.8, '= Kein Geldwäschefall', va='center', fontsize=9)

# --- PRECISION Section ---
ax.text(2.5, 5.8, 'PRECISION', ha='center', fontsize=11, fontweight='bold', color=FMA_BLUE)
ax.text(2.5, 5.3, '„Von allen Alarmen..."', ha='center', fontsize=9, style='italic')

# Draw "Vom Modell alarmiert" box
alarm_box = FancyBboxPatch((0.3, 2.5), 4.4, 2.5, boxstyle="round,pad=0.1", 
                            facecolor='#E8F4FD', edgecolor=FMA_BLUE, linewidth=2)
ax.add_patch(alarm_box)
ax.text(2.5, 4.7, 'Vom Modell alarmiert', ha='center', fontsize=8, color=FMA_BLUE)

# Icons inside alarm box: 3 real cases, 2 false alarms
positions_prec = [(0.8, 3.5), (1.6, 3.5), (2.4, 3.5), (3.2, 3.5), (4.0, 3.5)]
colors_prec = [RED, RED, RED, GRAY, GRAY]  # 3 echt, 2 Fehlalarm
for pos, col in zip(positions_prec, colors_prec):
    ax.add_patch(Circle(pos, 0.3, color=col, ec='black', lw=1.5))

# Result text
ax.text(2.5, 2.0, '3 von 5 = 60%', ha='center', fontsize=10, fontweight='bold', color=FMA_BLUE)
ax.text(2.5, 1.5, '„Wie viele Alarme sind echt?"', ha='center', fontsize=8, style='italic')

# --- RECALL Section ---
ax.text(7.5, 5.8, 'RECALL', ha='center', fontsize=11, fontweight='bold', color=FMA_ORANGE)
ax.text(7.5, 5.3, '„Von allen echten Fällen..."', ha='center', fontsize=9, style='italic')

# Draw "Alle echten Fälle" box
real_box = FancyBboxPatch((5.3, 2.5), 4.4, 2.5, boxstyle="round,pad=0.1", 
                           facecolor='#FDEDEC', edgecolor=FMA_ORANGE, linewidth=2)
ax.add_patch(real_box)
ax.text(7.5, 4.7, 'Alle echten Geldwäschefälle', ha='center', fontsize=8, color=FMA_ORANGE)

# Icons: 4 real cases total, 3 found (green check), 1 missed (X)
positions_rec = [(5.8, 3.5), (6.6, 3.5), (7.4, 3.5), (8.2, 3.5)]
for i, pos in enumerate(positions_rec):
    ax.add_patch(Circle(pos, 0.3, color=RED, ec='black', lw=1.5))
    if i < 3:  # Found
        ax.text(pos[0], pos[1], '✓', ha='center', va='center', fontsize=12, 
                color='white', fontweight='bold')
    else:  # Missed
        ax.text(pos[0], pos[1], '✗', ha='center', va='center', fontsize=12, 
                color='white', fontweight='bold')

# Result text
ax.text(7.5, 2.0, '3 von 4 = 75%', ha='center', fontsize=10, fontweight='bold', color=FMA_ORANGE)
ax.text(7.5, 1.5, '„Wie viele wurden gefunden?"', ha='center', fontsize=8, style='italic')

# Bottom summary
ax.text(5, 0.5, 'F1 = Balance: Nur gut, wenn BEIDE Werte hoch sind!', 
        ha='center', fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor=YELLOW, lw=2))

# ============ RIGHT: Trade-off visualization ============
ax = axes[1]

# Create bar comparison
categories = ['Strenger\nSchwellenwert', 'Ausgewogen', 'Lockerer\nSchwellenwert']
precision_vals = [0.90, 0.75, 0.40]
recall_vals = [0.50, 0.75, 0.95]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, precision_vals, width, label='Precision', color=FMA_BLUE, alpha=0.8)
bars2 = ax.bar(x + width/2, recall_vals, width, label='Recall', color=FMA_ORANGE, alpha=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Wert', fontsize=11)
ax.set_title('Der Trade-off: Man kann nicht beides maximieren!', fontsize=12, 
             fontweight='bold', color=FMA_ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper center', ncol=2, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add annotations
ax.annotate('Wenig Fehlalarme,\naber viel übersehen', xy=(0, 0.5), xytext=(0, 0.25),
            ha='center', fontsize=8, style='italic', color=GRAY)
ax.annotate('Kaum übersehen,\naber viele Fehlalarme', xy=(2, 0.4), xytext=(2, 0.15),
            ha='center', fontsize=8, style='italic', color=GRAY)
ax.annotate('✓ Beste\nBalance', xy=(1, 0.75), xytext=(1, 0.55),
            ha='center', fontsize=9, fontweight='bold', color=GREEN)

plt.tight_layout()
plt.savefig('precision_recall.pdf', bbox_inches='tight', dpi=150)
plt.savefig('precision_recall.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved precision_recall.pdf")

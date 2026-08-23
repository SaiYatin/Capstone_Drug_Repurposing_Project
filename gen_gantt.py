"""Generate a professional Gantt chart for Phase 3 & Phase 4."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = r'C:\D_Games\capstone\implementations\outputs\visualizations'

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')

# --- Task definitions ---
# (Task name, start_week, duration, phase, category)
tasks = [
    # Phase 3
    ("Transfer to Sjögren's Syndrome",           1, 3, 'P3', 'transfer'),
    ("Transfer to Myasthenia Gravis",             2, 3, 'P3', 'transfer'),
    ("Transfer to Autoimmune Hepatitis",          3, 3, 'P3', 'transfer'),
    ("Transfer to Crohn's Disease",               4, 3, 'P3', 'transfer'),
    ("Optimize pathway match rate (≥80%)",        4, 4, 'P3', 'optimize'),
    ("Clinical expert validation (Layer 6)",      5, 6, 'P3/P4', 'validation'),
    ("Consolidate 6-disease results",             7, 2, 'P3', 'report'),
    # Phase 4
    ("Molecular docking integration",             9, 4, 'P4', 'advanced'),
    ("Dynamic KG updates (PubMed)",              10, 4, 'P4', 'advanced'),
    ("Patient-level omics exploration",          11, 3, 'P4', 'advanced'),
    ("Final report & IEEE paper revision",       13, 3, 'P4', 'report'),
    ("Presentation & defense preparation",       14, 2, 'P4', 'report'),
]

# Colors by category
cat_colors = {
    'transfer':   '#2196F3',  # blue
    'optimize':   '#FF9800',  # orange
    'validation': '#E91E63',  # pink
    'advanced':   '#4CAF50',  # green
    'report':     '#9C27B0',  # purple
}

# Plot bars
y_positions = list(range(len(tasks) - 1, -1, -1))
for i, (name, start, dur, phase, cat) in enumerate(tasks):
    y = y_positions[i]
    color = cat_colors[cat]
    bar = ax.barh(y, dur, left=start, height=0.6, color=color, 
                  edgecolor='white', linewidth=1.5, alpha=0.85, zorder=3)
    # Phase label inside bar
    ax.text(start + dur/2, y, phase, ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white', zorder=4)

# Task labels
ax.set_yticks(y_positions)
ax.set_yticklabels([t[0] for t in tasks], fontsize=10, fontfamily='sans-serif')

# X axis
ax.set_xlim(0, 16.5)
ax.set_xticks(range(1, 17))
ax.set_xticklabels([f'W{i}' for i in range(1, 17)], fontsize=9)
ax.set_xlabel('Weeks', fontsize=12, fontweight='bold')

# Phase divider
ax.axvline(x=8.5, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
ax.text(4.5, len(tasks) + 0.3, 'Phase 3', ha='center', fontsize=14, fontweight='bold', 
        color='#1565C0', fontfamily='sans-serif')
ax.text(12.5, len(tasks) + 0.3, 'Phase 4', ha='center', fontsize=14, fontweight='bold', 
        color='#2E7D32', fontfamily='sans-serif')

# Grid
ax.set_axisbelow(True)
ax.xaxis.grid(True, alpha=0.2, linestyle='-', color='gray')
for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')

# Title
ax.set_title('BioPathNet++ — Project Progress Plan (Phase 3 & Phase 4)', 
             fontsize=15, fontweight='bold', pad=25, fontfamily='sans-serif')

# Legend
legend_patches = [
    mpatches.Patch(color='#2196F3', label='Disease Transfer Learning'),
    mpatches.Patch(color='#FF9800', label='Optimization'),
    mpatches.Patch(color='#E91E63', label='Clinical Validation'),
    mpatches.Patch(color='#4CAF50', label='Advanced Features'),
    mpatches.Patch(color='#9C27B0', label='Report & Presentation'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, 
          framealpha=0.9, edgecolor='#CCCCCC')

plt.tight_layout()
plt.savefig(f'{OUT}/gantt_chart_v2.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Gantt chart saved to: {OUT}/gantt_chart_v2.png")

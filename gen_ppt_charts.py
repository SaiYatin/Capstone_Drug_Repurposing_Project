"""Generate 2-disease specific result charts for PBC + Systemic Sclerosis."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = r'C:\D_Games\capstone\implementations\outputs\visualizations'

# --- 1. SOTA Comparison (bar chart, 2 disease focus) ---
fig, ax = plt.subplots(figsize=(10, 5))
models = ['Random\nBaseline', 'TransE', 'DRKG\nBaseline', 'TxGNN\n(Stanford)', 'BioPathNet++\n(Ours)']
auprc = [0.500, 0.650, 0.700, 0.920, 0.935]
auroc = [0.500, 0.680, 0.720, 0.930, 0.941]
x = np.arange(len(models))
w = 0.35
bars1 = ax.bar(x - w/2, auprc, w, label='AUPRC', color='#2196F3', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, auroc, w, label='AUROC', color='#FF9800', edgecolor='white', linewidth=0.5)
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('BioPathNet++ vs State-of-the-Art (Test Set)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=12)
ax.bar_label(bars1, fmt='%.3f', fontsize=8, padding=2)
ax.bar_label(bars2, fmt='%.3f', fontsize=8, padding=2)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.axhline(y=0.920, color='red', linestyle='--', alpha=0.3, label='TxGNN baseline')
plt.tight_layout()
plt.savefig(f'{OUT}/sota_2disease.png', dpi=200, bbox_inches='tight')
plt.close()
print("1. SOTA comparison saved")

# --- 2. Per-Disease Metrics (only PBC + SS) ---
fig, ax = plt.subplots(figsize=(8, 4.5))
diseases = ['Primary Biliary\nCholangitis', 'Systemic\nSclerosis']
auprc_d = [0.949, 0.871]
auroc_d = [0.956, 0.907]
x = np.arange(len(diseases))
w = 0.3
b1 = ax.bar(x - w/2, auprc_d, w, label='AUPRC', color='#4CAF50', edgecolor='white')
b2 = ax.bar(x + w/2, auroc_d, w, label='AUROC', color='#9C27B0', edgecolor='white')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Per-Disease Zero-Shot Performance', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(diseases, fontsize=12)
ax.set_ylim(0.75, 1.0)
ax.legend(fontsize=12, loc='lower right')
ax.bar_label(b1, fmt='%.3f', fontsize=11, padding=3)
ax.bar_label(b2, fmt='%.3f', fontsize=11, padding=3)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/per_disease_2only.png', dpi=200, bbox_inches='tight')
plt.close()
print("2. Per-disease metrics saved")

# --- 3. Validation Summary (5 layers, 2 diseases) ---
fig, ax = plt.subplots(figsize=(10, 5))
layers = ['L1: Benchmark\nAUPRC', 'L2: Zero-Shot\nTransfer', 'L3: Few-Shot\nOptimality', 'L4: Literature\nValidation', 'L5: Immune Pathway\nMatch']
scores = [0.935, 0.910, 0.949, 0.786, 0.750]
targets = [0.92, 0.80, None, 0.70, 0.80]
colors = ['#4CAF50' if (t is None or s >= t) else '#FF9800' for s, t in zip(scores, targets)]
bars = ax.barh(layers, scores, color=colors, edgecolor='white', height=0.6)
ax.set_xlim(0, 1.1)
ax.set_xlabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Five-Layer Validation Framework (PBC + Systemic Sclerosis)', fontsize=14, fontweight='bold')
for bar, score, target in zip(bars, scores, targets):
    ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
            f'{score:.1%}', va='center', fontsize=11, fontweight='bold')
    if target:
        ax.axvline(x=target, color='red', linestyle='--', alpha=0.3)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.legend(['Target threshold'], loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/validation_2disease.png', dpi=200, bbox_inches='tight')
plt.close()
print("3. Validation summary saved")

# --- 4. Phase 3/4 Gantt Chart ---
fig, ax = plt.subplots(figsize=(12, 5))
tasks = [
    'Transfer to remaining 4 diseases',
    'Optimize pathway match rate',
    'Clinical expert validation (Layer 6)',
    'Molecular docking integration',
    'Dynamic KG updates',
    'Final report & publication',
]
# Phase 3: weeks 1-8, Phase 4: weeks 9-16
starts =  [0, 2, 4, 6, 10, 13]
durations = [4, 4, 6, 5, 4, 3]
phases = ['P3', 'P3', 'P3/P4', 'P4', 'P4', 'P4']
colors_g = ['#2196F3', '#2196F3', '#FF9800', '#4CAF50', '#4CAF50', '#9C27B0']

for i, (task, start, dur, phase, col) in enumerate(zip(tasks, starts, durations, phases, colors_g)):
    ax.barh(i, dur, left=start, height=0.6, color=col, edgecolor='white', alpha=0.85)
    ax.text(start + dur/2, i, f'{phase}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(tasks, fontsize=10)
ax.set_xlabel('Weeks', fontsize=12, fontweight='bold')
ax.set_title('Project Progress Plan — Phase 3 & Phase 4', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 17, 2))
ax.set_xticklabels([f'W{i}' for i in range(0, 17, 2)])
ax.axvline(x=8, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(8, len(tasks)-0.3, 'Phase 3 → Phase 4', ha='center', fontsize=9, color='red', fontweight='bold')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/gantt_chart.png', dpi=200, bbox_inches='tight')
plt.close()
print("4. Gantt chart saved")

print("\nAll 2-disease charts generated!")

"""
BioPathNet++ Publication-Quality Visualization Suite (v2 Results)
==================================================================
Generates 11 publication-ready figures for the BioPathNet++ project.
Styled for IEEE papers and academic presentations (300 DPI, clean typography,
consistent color schemes, exact data labels).

Output Directory: c:\\D_Games\\capstone\\implementations\\outputs\\visualizations\\
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------------------------------------------
# Configuration & Global Styling
# -----------------------------------------------------------------------------
OUTPUT_DIR = r"c:\D_Games\capstone\implementations\outputs\visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set clean aesthetic style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except IOError:
    try:
        plt.style.use('seaborn-whitegrid')
    except IOError:
        plt.style.use('default')

# Publication-quality typography & visual settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
})


# -----------------------------------------------------------------------------
# 1. SOTA Comparison Bar Chart
# -----------------------------------------------------------------------------
def plot_1_sota_comparison():
    """Figure 1: SOTA Comparison Bar Chart with AUPRC and AUROC side by side."""
    models = [
        'Random Baseline',
        'TransE',
        'DRKG Baseline',
        'TxGNN (Stanford)',
        'BioPathNet++ (Val)',
        'BioPathNet++ (Test)'
    ]
    auprc = [0.500, 0.650, 0.700, 0.920, 0.918, 0.935]
    auroc = [0.500, 0.680, 0.720, 0.930, 0.926, 0.941]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6.5))

    rects1 = ax.bar(x - width/2, auprc, width, label='AUPRC', color='#2b6cb0',
                    edgecolor='black', linewidth=0.8, alpha=0.9)
    rects2 = ax.bar(x + width/2, auroc, width, label='AUROC', color='#38a169',
                    edgecolor='black', linewidth=0.8, alpha=0.9)

    # Highlight BioPathNet++ bars with distinctive edges
    rects1[-2].set_edgecolor('#d69e2e')
    rects1[-2].set_linewidth(2.0)
    rects2[-2].set_edgecolor('#d69e2e')
    rects2[-2].set_linewidth(2.0)
    rects1[-1].set_edgecolor('#c53030')
    rects1[-1].set_linewidth(2.2)
    rects2[-1].set_edgecolor('#c53030')
    rects2[-1].set_linewidth(2.2)

    # TxGNN SOTA Benchmark Reference Line
    txgnn_auprc = 0.920
    ax.axhline(y=txgnn_auprc, color='#e53e3e', linestyle='--', linewidth=1.8,
               alpha=0.85, label=f'TxGNN SOTA AUPRC Benchmark ({txgnn_auprc:.3f})')

    # Value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Metric Score', fontsize=13, fontweight='bold')
    ax.set_title('State-of-the-Art Model Performance Comparison (v2 Results)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='semibold')
    ax.set_ylim(0.40, 1.04)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white')

    # Add SOTA badge annotation
    ax.annotate('★ +1.5% AUPRC over SOTA',
                xy=(5 - width/2, 0.935),
                xytext=(4.2, 0.99),
                arrowprops=dict(facecolor='#c53030', shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, fontweight='bold', color='#c53030',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#feb2b2", lw=1.5))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'sota_comparison_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 2. Per-Disease Zero-Shot Performance
# -----------------------------------------------------------------------------
def plot_2_per_disease_zero_shot():
    """Figure 2: Per-Disease Zero-Shot Performance (Horizontal Grouped Bar Chart)."""
    diseases = [
        "Sjögren's Syndrome",
        "Autoimmune Hepatitis",
        "Prim. Biliary Cholangitis",
        "Myasthenia Gravis",
        "Crohn's Disease (LODO)",
        "Systemic Sclerosis"
    ]
    auprc = [1.000, 0.975, 0.949, 0.907, 0.891, 0.871]
    auroc = [1.000, 0.970, 0.957, 0.932, 0.929, 0.907]

    # Invert so highest AUPRC is at top
    diseases_rev = diseases[::-1]
    auprc_rev = auprc[::-1]
    auroc_rev = auroc[::-1]

    y = np.arange(len(diseases_rev))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.5))

    rects1 = ax.barh(y + height/2, auprc_rev, height, label='AUPRC',
                     color='#2b6cb0', edgecolor='black', linewidth=0.8, alpha=0.9)
    rects2 = ax.barh(y - height/2, auroc_rev, height, label='AUROC',
                     color='#38a169', edgecolor='black', linewidth=0.8, alpha=0.9)

    # Average lines
    avg_auprc = np.mean(auprc)
    avg_auroc = np.mean(auroc)
    ax.axvline(x=avg_auprc, color='#2b6cb0', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'Mean AUPRC ({avg_auprc:.3f})')
    ax.axvline(x=avg_auroc, color='#38a169', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'Mean AUROC ({avg_auroc:.3f})')

    # Value labels
    for rect in rects1:
        w = rect.get_width()
        ax.annotate(f'{w:.3f}',
                    xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, fontweight='bold')

    for rect in rects2:
        w = rect.get_width()
        ax.annotate(f'{w:.3f}',
                    xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, fontweight='bold')

    ax.set_xlabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Per-Disease Zero-Shot Transfer Performance (AUPRC & AUROC)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(diseases_rev, fontsize=11, fontweight='semibold')
    ax.set_xlim(0.80, 1.08)
    ax.legend(loc='lower left', frameon=True, framealpha=0.95, facecolor='white')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'per_disease_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 3. Ablation Study
# -----------------------------------------------------------------------------
def plot_3_ablation_study():
    """Figure 3: Ablation Study showing Delta AUPRC from Full Model."""
    variants = [
        'Full Model (Baseline)',
        'No Metapaths',
        'No BRG',
        'Single NBF Layer',
        'No Fusion',
        'No BioBERT Features'
    ]
    auprc_values = [0.9350, 0.9376, 0.9350, 0.9350, 0.9061, 0.6028]
    full_model_score = 0.9350
    deltas = [val - full_model_score for val in auprc_values]

    # Invert so baseline is at top
    variants_rev = variants[::-1]
    deltas_rev = deltas[::-1]
    auprc_rev = auprc_values[::-1]

    colors = ['#e53e3e' if d < -0.001 else '#38a169' if d > 0.001 else '#4a5568' for d in deltas_rev]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(variants_rev))
    bars = ax.barh(y, deltas_rev, color=colors, edgecolor='black', linewidth=1.0, height=0.55)

    ax.axvline(x=0.0, color='black', linestyle='-', linewidth=1.2)

    # Value labels with delta and absolute AUPRC
    for i, (bar, d, val) in enumerate(zip(bars, deltas_rev, auprc_rev)):
        sign = '+' if d > 0 else ''
        text_label = f"{sign}{d:.4f}  (AUPRC: {val:.4f})"
        if d < 0:
            ax.annotate(text_label,
                        xy=(d, bar.get_y() + bar.get_height()/2),
                        xytext=(-6, 0), textcoords="offset points",
                        ha='right', va='center', fontsize=9.5, fontweight='bold', color='#9b2c2c')
        else:
            ax.annotate(text_label,
                        xy=(d, bar.get_y() + bar.get_height()/2),
                        xytext=(6, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9.5, fontweight='bold',
                        color='#22543d' if d > 0 else '#2d3748')

    ax.set_xlabel('Change in AUPRC (Δ vs Full Model: 0.9350)', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: Impact of BioPathNet++ Architectural Components',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(variants_rev, fontsize=11, fontweight='semibold')
    ax.set_xlim(-0.40, 0.08)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#38a169', edgecolor='black', label='Gain / Improvement'),
        mpatches.Patch(facecolor='#4a5568', edgecolor='black', label='Neutral (Δ = 0.0000)'),
        mpatches.Patch(facecolor='#e53e3e', edgecolor='black', label='Performance Degradation')
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True, framealpha=0.95)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'ablation_study_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 4. Literature Validation
# -----------------------------------------------------------------------------
def plot_4_literature_validation():
    """Figure 4: Literature Validation Support Rates per Disease."""
    diseases = [
        "Crohn's Disease",
        "Sjögren's Syndrome",
        "Systemic Sclerosis",
        "Prim. Biliary Cholangitis",
        "Myasthenia Gravis",
        "Autoimmune Hepatitis"
    ]
    support_rates = [90.0, 80.0, 80.0, 75.0, 60.0, 60.0]
    overall_mean = 74.1

    # Invert for descending order from top
    diseases_rev = diseases[::-1]
    rates_rev = support_rates[::-1]

    colors = ['#2b6cb0' if r >= 70.0 else '#dd6b20' for r in rates_rev]

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(diseases_rev))
    bars = ax.barh(y, rates_rev, color=colors, edgecolor='black', linewidth=1.0, height=0.55)

    # Vertical threshold line at 70%
    ax.axvline(x=70.0, color='#e53e3e', linestyle='--', linewidth=2.0,
               alpha=0.85, label='Clinical Target Threshold (70%)')
    ax.axvline(x=overall_mean, color='#2b6cb0', linestyle=':', linewidth=2.0,
               alpha=0.85, label=f'Overall Model Support Mean ({overall_mean:.1f}%)')

    # Value labels
    for bar, rate in zip(bars, rates_rev):
        ax.annotate(f'{rate:.1f}%',
                    xy=(rate, bar.get_y() + bar.get_height()/2),
                    xytext=(6, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Literature Validation Support Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('PubMed Literature Validation Rates Across Target Indications',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(diseases_rev, fontsize=11, fontweight='semibold')
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'literature_validation_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 5. Immune Pathway Match Rates
# -----------------------------------------------------------------------------
def plot_5_pathway_match():
    """Figure 5: Immune Pathway Match Rates."""
    diseases = [
        "Prim. Biliary Cholangitis",
        "Sjögren's Syndrome",
        "Systemic Sclerosis",
        "Myasthenia Gravis",
        "Autoimmune Hepatitis",
        "Crohn's Disease"
    ]
    match_pct = [95.0, 55.0, 55.0, 55.0, 55.0, 55.0]
    match_ratio = ["19/20", "11/20", "11/20", "11/20", "11/20", "11/20"]
    overall_mean = 61.7

    diseases_rev = diseases[::-1]
    pct_rev = match_pct[::-1]
    ratios_rev = match_ratio[::-1]

    colors = ['#319795' if p > 70 else '#4a5568' for p in pct_rev]

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(diseases_rev))
    bars = ax.barh(y, pct_rev, color=colors, edgecolor='black', linewidth=1.0, height=0.55)

    ax.axvline(x=overall_mean, color='#d69e2e', linestyle='--', linewidth=2.0,
               label=f'Overall Pathway Match Mean ({overall_mean:.1f}%)')

    for bar, pct, ratio in zip(bars, pct_rev, ratios_rev):
        ax.annotate(f'{pct:.1f}% ({ratio})',
                    xy=(pct, bar.get_y() + bar.get_height()/2),
                    xytext=(6, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Immune Pathway Match Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Biological Pathway Match Rates for Top Predicted Candidates',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(diseases_rev, fontsize=11, fontweight='semibold')
    ax.set_xlim(0, 110)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'pathway_match_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 6. Few-Shot vs Zero-Shot Comparison
# -----------------------------------------------------------------------------
def plot_6_fewshot_comparison():
    """Figure 6: Few-Shot vs Zero-Shot Performance Comparison."""
    diseases = [
        "Systemic Sclerosis",
        "Prim. Biliary Cholangitis",
        "Myasthenia Gravis",
        "Autoimmune Hepatitis",
        "Crohn's Disease"
    ]
    shot_0 = [0.871, 0.949, 0.907, 0.975, 0.892]
    shot_5 = [0.865, 0.945, 0.912, 0.973, 0.883]
    shot_10 = [0.865, 0.952, 0.916, 0.978, 0.890]

    x = np.arange(len(diseases))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6.5))

    rects1 = ax.bar(x - width, shot_0, width, label='0-Shot (Pure Zero-Shot)',
                    color='#3182ce', edgecolor='black', linewidth=0.8, alpha=0.9)
    rects2 = ax.bar(x, shot_5, width, label='5-Shot Adaptation',
                    color='#805ad5', edgecolor='black', linewidth=0.8, alpha=0.9)
    rects3 = ax.bar(x + width, shot_10, width, label='10-Shot Adaptation',
                    color='#38a169', edgecolor='black', linewidth=0.8, alpha=0.9)

    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold', rotation=0)

    ax.set_ylabel('AUPRC Score', fontsize=13, fontweight='bold')
    ax.set_title('Generalization Capacity: Zero-Shot vs Few-Shot Transfer Learning',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=10.5, fontweight='semibold')
    ax.set_ylim(0.80, 1.03)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'fewshot_comparison_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 7. Clinical Trials Validation
# -----------------------------------------------------------------------------
def plot_7_clinical_trials():
    """Figure 7: Clinical Trials Validation (100% match rate across indications)."""
    diseases = [
        "Crohn's Disease",
        "Sjögren's Syndrome",
        "Systemic Sclerosis",
        "Prim. Biliary Cholangitis",
        "Myasthenia Gravis",
        "Autoimmune Hepatitis"
    ]
    rates = [100.0] * len(diseases)

    diseases_rev = diseases[::-1]
    rates_rev = rates[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(diseases_rev))
    bars = ax.barh(y, rates_rev, color='#2b6cb0', edgecolor='black', linewidth=1.0, height=0.55)

    for bar in bars:
        ax.annotate('100.0% Validated',
                    xy=(bar.get_width() - 3, bar.get_y() + bar.get_height()/2),
                    ha='right', va='center', fontsize=10.5, fontweight='bold', color='white')

    ax.set_xlabel('Clinical Trials Concordance Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('ClinicalTrials.gov Real-World Evidence Concordance',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(diseases_rev, fontsize=11, fontweight='semibold')
    ax.set_xlim(0, 115)

    # Informational banner
    ax.text(0.5, -0.15,
            'Total Unique Trials Analyzed: 1,575  |  100% Agreement on Top-Tier Indications',
            transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="#ebf8ff", ec="#3182ce", lw=1.5))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'clinical_trials_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 8. Top Drug Rankings - PBC
# -----------------------------------------------------------------------------
def plot_8_top_drugs_pbc():
    """Figure 8: Top-10 Drug Rankings for Primary Biliary Cholangitis (PBC)."""
    drugs = [
        "Thalidomide",
        "Ursodeoxycholic Acid",
        "Fenofibrate",
        "Prednisolone",
        "Paclitaxel",
        "Methimazole",
        "Ibudilast",
        "Tamoxifen",
        "Sirolimus",
        "Roxithromycin"
    ]
    scores = [8.464, 8.445, 8.435, 8.427, 8.425, 8.410, 8.405, 8.389, 8.387, 8.385]
    types = ["Known", "Known", "Known", "Known", "Novel", "Novel", "Novel", "Novel", "Novel", "Novel"]

    # Invert so rank 1 is at top
    drugs_rev = drugs[::-1]
    scores_rev = scores[::-1]
    types_rev = types[::-1]

    colors = ['#38a169' if t == "Known" else '#3182ce' for t in types_rev]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(drugs_rev))
    bars = ax.barh(y, scores_rev, color=colors, edgecolor='black', linewidth=0.9, height=0.6)

    for bar, score, t in zip(bars, scores_rev, types_rev):
        ax.annotate(f'{score:.3f}  [{t}]',
                    xy=(score, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, fontweight='bold',
                    color='#22543d' if t == "Known" else '#1a365d')

    ax.set_xlabel('BioPathNet++ Affinity / Repurposing Score', fontsize=13, fontweight='bold')
    ax.set_title('Top-10 Repurposing Candidates: Primary Biliary Cholangitis (PBC)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(drugs_rev, fontsize=11, fontweight='semibold')
    ax.set_xlim(8.35, 8.52)

    # Custom Legend
    legend_elements = [
        mpatches.Patch(facecolor='#38a169', edgecolor='black', label='Known Standard-of-Care / Indication'),
        mpatches.Patch(facecolor='#3182ce', edgecolor='black', label='Novel Repurposing Candidate')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.95, facecolor='white')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'drug_ranking_pbc_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 9. Top Drug Rankings - SSc
# -----------------------------------------------------------------------------
def plot_9_top_drugs_ssc():
    """Figure 9: Top-10 Drug Rankings for Systemic Sclerosis (SSc)."""
    drugs = [
        "Sulfamethoxazole",
        "Etidronic Acid",
        "Clodronic Acid",
        "Leucovorin",
        "Griseofulvin",
        "Cyclic AMP",
        "Histamine",
        "Cholesterol",
        "Vaccine Components",
        "Spectinomycin"
    ]
    scores = [3.372, 3.334, 3.332, 3.325, 3.319, 3.314, 3.300, 3.300, 3.297, 3.296]

    drugs_rev = drugs[::-1]
    scores_rev = scores[::-1]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(drugs_rev))
    bars = ax.barh(y, scores_rev, color='#2b6cb0', edgecolor='black', linewidth=0.9, height=0.6)

    for bar, score in zip(bars, scores_rev):
        ax.annotate(f'{score:.3f}',
                    xy=(score, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, fontweight='bold')

    ax.set_xlabel('BioPathNet++ Affinity / Repurposing Score', fontsize=13, fontweight='bold')
    ax.set_title('Top-10 Repurposing Candidates: Systemic Sclerosis (SSc)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(drugs_rev, fontsize=11, fontweight='semibold')
    ax.set_xlim(3.27, 3.40)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'drug_ranking_ssc_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 10. Pathway Distribution Donut Chart
# -----------------------------------------------------------------------------
def plot_10_pathway_distribution():
    """Figure 10: Donut Chart of Immune Pathway Involvement."""
    pathways = [
        "TNF signaling",
        "Cytokine cascade",
        "Interleukins",
        "NF-κB pathway",
        "JAK-STAT",
        "T-cell markers",
        "B-cell markers"
    ]
    counts = [58, 52, 47, 34, 12, 10, 8]
    total = sum(counts)

    # Cohesive publication color palette
    colors = ['#2b6cb0', '#319795', '#38a169', '#d69e2e', '#dd6b20', '#805ad5', '#718096']

    fig, ax = plt.subplots(figsize=(9, 7.5))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[f"{p}\n(n={c})" for p, c in zip(pathways, counts)],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.0)
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    for text in texts:
        text.set_fontsize(10.5)
        text.set_fontweight('semibold')

    # Center circle annotation
    ax.text(0, 0, f"Total Hits\nN = {total}", ha='center', va='center',
            fontsize=13, fontweight='bold', color='#2d3748')

    ax.set_title('Biological Pathway Distribution Across Repurposed Candidates',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'pathway_distribution_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# 11. Multi-Layer Validation Summary (Radar / Spider Chart)
# -----------------------------------------------------------------------------
def plot_11_validation_summary():
    """Figure 11: Multi-Layer Validation Radar Chart (Achieved vs Target)."""
    categories = [
        'Benchmark\nAUPRC',
        'Zero-Shot\nTransfer',
        'Literature\nSupport',
        'Pathway\nMatch',
        'Clinical\nTrials'
    ]
    achieved = [0.935, 0.910, 0.741, 0.617, 1.000]
    target =   [0.920, 0.800, 0.700, 0.700, 0.800]

    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    achieved_c = achieved + [achieved[0]]
    target_c = target + [target[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))

    # Plot Target Line
    ax.plot(angles_c, target_c, color='#e53e3e', linewidth=2.2, linestyle='--',
            label='Target / SOTA Baseline', zorder=3)

    # Plot Achieved Polygon
    ax.plot(angles_c, achieved_c, color='#2b6cb0', linewidth=2.5, linestyle='-',
            label='BioPathNet++ Achieved', zorder=4)
    ax.fill(angles_c, achieved_c, color='#3182ce', alpha=0.30, zorder=2)

    # Markers and Data Value Annotations
    for angle, val, tgt in zip(angles, achieved, target):
        ax.plot(angle, val, 'o', color='#1a365d', markersize=7, zorder=5)
        ax.annotate(f"{val:.3f}",
                    xy=(angle, val),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1a365d')

    # Configure axes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0.40, 1.08)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.50', '0.60', '0.70', '0.80', '0.90', '1.00'],
                       fontsize=9, color='#718096')

    ax.set_title('Multi-Layer Validation Radar: BioPathNet++ vs Target Benchmarks',
                 fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True,
              framealpha=0.95, facecolor='white', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'validation_summary_v2.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Runner
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BioPathNet++ v2 Visualization Suite")
    print("Generating 11 publication-quality IEEE-ready figures...")
    print("=" * 70)

    plot_1_sota_comparison()
    plot_2_per_disease_zero_shot()
    plot_3_ablation_study()
    plot_4_literature_validation()
    plot_5_pathway_match()
    plot_6_fewshot_comparison()
    plot_7_clinical_trials()
    plot_8_top_drugs_pbc()
    plot_9_top_drugs_ssc()
    plot_10_pathway_distribution()
    plot_11_validation_summary()

    print("=" * 70)
    print(f"✓ All 11 figures successfully generated in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()

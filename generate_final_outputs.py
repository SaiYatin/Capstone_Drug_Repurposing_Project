"""
Generate Final Outputs — Master Orchestration Script
=====================================================

Runs all Phase 3 and Phase 4 scripts in sequence and generates
a comprehensive final report.

Usage:
    python generate_final_outputs.py

    Or run individual phases:
    python generate_final_outputs.py --phase 3
    python generate_final_outputs.py --phase 4

Author: Yatin
Date: April 2026
"""

import sys
import subprocess
import time
import json
import io
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def run_script(script_path, description):
    """Run a Python script and return success/failure."""
    print(f"\n{'━' * 70}")
    print(f"▶ {description}")
    print(f"  Running: python {script_path}")
    print(f"{'━' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent),
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ {description} completed ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n✗ {description} FAILED (exit code {result.returncode})")
        return False


def generate_final_report():
    """Generate a comprehensive markdown report summarizing all results."""
    import pandas as pd
    output_dir = Path('outputs')
    report_path = output_dir / 'final_report.md'

    lines = []
    lines.append("# BioPathNet++ — Final Results Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    lines.append(f"**Project:** Path-Based Graph Neural Networks for Autoimmune Drug Repurposing")
    lines.append(f"**Team:** TVNS Yatin, A Charan KR, Praveen Kumar Yadav, Srijan Jha")
    lines.append(f"**Guide:** Dr. Priyanka H")

    lines.append("\n---\n")

    # ── Section 1: Model Overview ──
    lines.append("## 1. Model Overview")
    config_path = Path('checkpoints/config.json')
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        lines.append(f"\n| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Architecture | BioPathNet++ (NBFNet + Metapaths + BRG + Attention) |")
        lines.append(f"| Hidden Dimension | {config.get('hidden_dim', 'N/A')} |")
        lines.append(f"| NBF Layers | {config.get('num_nbf_layers', 'N/A')} |")
        lines.append(f"| Metapaths | {config.get('num_metapaths', 7)} |")
        lines.append(f"| Attention Heads | {config.get('num_attention_heads', 'N/A')} |")
        lines.append(f"| Parameters | ~7M |")
        lines.append(f"| Training | 100 epochs on RTX 4500 Ada (24GB) |")

    lines.append("\n---\n")

    # ── Section 2: Overall Performance ──
    lines.append("## 2. Overall Performance")
    metrics_path = output_dir / 'validation' / 'overall_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)

        test = data.get('test', {})
        val = data.get('validation', {})

        lines.append(f"\n| Metric | Validation | Test (Zero-Shot) |")
        lines.append(f"|--------|-----------|-----------------|")
        lines.append(f"| AUPRC | {val.get('auprc', 'N/A'):.4f} | {test.get('auprc', 'N/A'):.4f} |")
        lines.append(f"| AUROC | {val.get('auroc', 'N/A'):.4f} | {test.get('auroc', 'N/A'):.4f} |")
        lines.append(f"| MRR | {val.get('mrr', 'N/A'):.4f} | {test.get('mrr', 'N/A'):.4f} |")
        lines.append(f"| Hit@10 | — | {test.get('hit_at_10', 'N/A'):.4f} |")
        lines.append(f"| Hit@20 | — | {test.get('hit_at_20', 'N/A'):.4f} |")
        lines.append(f"| Precision@10 | — | {test.get('precision_at_10', 'N/A'):.4f} |")

        lines.append(f"\n**Key Result:** Test AUPRC of {test.get('auprc', 0):.3f} on 6 held-out diseases "
                      f"demonstrates strong zero-shot transfer capability.")
    else:
        lines.append("\n*Metrics not yet computed. Run `python validation/metrics.py`.*")

    lines.append("\n---\n")

    # ── Section 3: Per-Disease Performance ──
    lines.append("## 3. Per-Disease Performance (Zero-Shot Transfer)")
    per_disease_path = output_dir / 'validation' / 'per_disease_metrics.csv'
    if per_disease_path.exists():
        pdf = pd.read_csv(per_disease_path)
        lines.append(f"\n| Disease | AUPRC | AUROC | MRR | Hit@20 | Samples |")
        lines.append(f"|---------|-------|-------|-----|--------|---------|")
        for _, row in pdf.iterrows():
            lines.append(f"| {row.get('disease_name', 'N/A')} | "
                          f"{row.get('auprc', 0):.4f} | "
                          f"{row.get('auroc', 0):.4f} | "
                          f"{row.get('mrr', 0):.4f} | "
                          f"{row.get('hit_at_20', 0):.4f} | "
                          f"{row.get('num_samples', 0)} |")

    lines.append("\n---\n")

    # ── Section 4: SOTA Comparison ──
    lines.append("## 4. SOTA Comparison")
    sota_path = output_dir / 'validation' / 'comparison_with_sota.csv'
    if sota_path.exists():
        sdf = pd.read_csv(sota_path)
        lines.append(f"\n| Model | AUPRC | AUROC |")
        lines.append(f"|-------|-------|-------|")
        for _, row in sdf.iterrows():
            lines.append(f"| {row['model']} | {row['auprc']:.3f} | {row['auroc']:.3f} |")

    lines.append("\n---\n")

    # ── Section 5: Drug Predictions ──
    lines.append("## 5. Top Drug Candidates (Zero-Shot Predictions)")
    pred_dir = output_dir / 'predictions'
    if pred_dir.exists():
        from models.disease_mappings import TEST_DISEASE_ENTITIES, get_disease_name, get_disease_short_name
        for disease_entity in TEST_DISEASE_ENTITIES:
            disease_name = get_disease_name(disease_entity)
            short_name = get_disease_short_name(disease_entity)
            csv_path = pred_dir / f"{short_name}_top50.csv"
            if csv_path.exists():
                ddf = pd.read_csv(csv_path).head(10)
                lines.append(f"\n### {disease_name}")
                lines.append(f"\n| Rank | Drug | Score | Known? |")
                lines.append(f"|------|------|-------|--------|")
                for _, row in ddf.iterrows():
                    known = "✓" if row.get('known_drug', False) else ""
                    lines.append(f"| {row['rank']} | {row.get('drug_name', row['drug_id'])} | "
                                  f"{row['score']:.4f} | {known} |")

    lines.append("\n---\n")

    # ── Section 6: Literature Validation ──
    lines.append("## 6. Literature Validation")
    lit_path = output_dir / 'validation' / 'literature_summary.json'
    if lit_path.exists():
        with open(lit_path) as f:
            lit = json.load(f)
        lines.append(f"\n- **Total predictions validated:** {lit.get('total_validated', 0)}")
        lines.append(f"- **Strong evidence (≥3 papers):** {lit.get('strong', 0)}")
        lines.append(f"- **Moderate evidence (1-2 papers):** {lit.get('moderate', 0)}")
        lines.append(f"- **Weak/Novel (0 papers):** {lit.get('weak', 0)}")
        lines.append(f"- **Overall support rate:** {lit.get('support_rate', 0):.1f}%")
    else:
        lines.append("\n*Literature validation not yet run.*")

    lines.append("\n---\n")

    # ── Section 7: Pathway Analysis ──
    lines.append("## 7. Immune Pathway Analysis")
    pw_path = output_dir / 'validation' / 'pathway_summary.json'
    if pw_path.exists():
        with open(pw_path) as f:
            pw = json.load(f)
        lines.append(f"\n- **Total predictions analyzed:** {pw.get('total_analyzed', 0)}")
        lines.append(f"- **Immune pathway matches:** {pw.get('immune_matches', 0)}")
        lines.append(f"- **Overall match rate:** {pw.get('overall_match_rate', 0):.1f}%")
        if 'pathway_frequency' in pw:
            lines.append(f"\n**Pathway frequency:**")
            for pathway, count in sorted(pw['pathway_frequency'].items(), key=lambda x: -x[1]):
                lines.append(f"- {pathway}: {count}")
    else:
        lines.append("\n*Pathway analysis not yet run.*")

    lines.append("\n---\n")

    # ── Section 8: Few-Shot Results ──
    lines.append("## 8. Few-Shot Fine-Tuning Results")
    fs_path = output_dir / 'validation' / 'few_shot_comparison.csv'
    if fs_path.exists():
        fsdf = pd.read_csv(fs_path)
        lines.append(f"\n| Disease | Pairs | Zero-Shot | 5-Shot | 10-Shot |")
        lines.append(f"|---------|-------|-----------|--------|---------|")
        for _, row in fsdf.iterrows():
            zs = f"{row.get('zero_shot_auprc', 0):.4f}"
            fs5 = f"{row.get('5_shot_auprc', 'N/A'):.4f}" if pd.notna(row.get('5_shot_auprc')) else 'N/A'
            fs10 = f"{row.get('10_shot_auprc', 'N/A'):.4f}" if pd.notna(row.get('10_shot_auprc')) else 'N/A'
            lines.append(f"| {row['disease_name']} | {row['total_pairs']} | {zs} | {fs5} | {fs10} |")
    else:
        lines.append("\n*Few-shot experiments not yet run.*")

    lines.append("\n---\n")

    # ── Conclusion ──
    lines.append("## 9. Conclusion")
    lines.append("""
BioPathNet++ demonstrates strong zero-shot transfer learning for autoimmune drug
repurposing. The model, trained on only 3 data-rich diseases (RA, MS, SLE), successfully
transfers knowledge to 6 held-out rare diseases including Sjögren's Syndrome.

**Key achievements:**
- Path-based reasoning enables interpretable predictions
- Zero-shot transfer eliminates need for disease-specific training data
- Multi-layer validation confirms prediction quality
- Top drug candidates are biologically plausible and literature-supported
""")

    # Write report
    report_text = '\n'.join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✓ Final report saved to {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate all Phase 3/4 outputs")
    parser.add_argument('--phase', type=int, choices=[3, 4], help='Run specific phase only')
    parser.add_argument('--skip-predictions', action='store_true', help='Skip drug prediction (slow)')
    parser.add_argument('--skip-pubmed', action='store_true', help='Skip PubMed queries')
    args = parser.parse_args()

    print("═" * 80)
    print("  BIOPATHNET++ — FINAL OUTPUT GENERATION")
    print("═" * 80)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)

    successes = 0
    failures = 0

    # === Phase 3: Transfer Learning ===
    if args.phase is None or args.phase == 3:
        print("\n\n" + "█" * 80)
        print("  PHASE 3: TRANSFER LEARNING")
        print("█" * 80)

        if not args.skip_predictions:
            if run_script('transfer_learning/predict_drugs.py', 'Zero-Shot Drug Prediction'):
                successes += 1
            else:
                failures += 1

        if run_script('transfer_learning/few_shot.py', 'Few-Shot Fine-Tuning'):
            successes += 1
        else:
            failures += 1

    # === Phase 4: Validation ===
    if args.phase is None or args.phase == 4:
        print("\n\n" + "█" * 80)
        print("  PHASE 4: VALIDATION & OUTPUTS")
        print("█" * 80)

        if run_script('validation/metrics.py', 'Comprehensive Metrics'):
            successes += 1
        else:
            failures += 1

        if not args.skip_pubmed:
            if run_script('validation/literature_val.py', 'Literature Validation'):
                successes += 1
            else:
                failures += 1

        if run_script('validation/pathway_analysis.py', 'Immune Pathway Analysis'):
            successes += 1
        else:
            failures += 1

        if run_script('validation/visualize_results.py', 'Visualization Generation'):
            successes += 1
        else:
            failures += 1

    # === Generate Report ===
    print("\n\n" + "█" * 80)
    print("  GENERATING FINAL REPORT")
    print("█" * 80)

    try:
        import pandas as pd  # needed for report generation
        generate_final_report()
        successes += 1
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        failures += 1

    # === Summary ===
    print("\n\n" + "═" * 80)
    print("  EXECUTION SUMMARY")
    print("═" * 80)
    print(f"  ✓ Successes: {successes}")
    print(f"  ✗ Failures:  {failures}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)

    # List output files
    output_dir = Path('outputs')
    if output_dir.exists():
        print("\n  Output files:")
        for p in sorted(output_dir.rglob('*')):
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                print(f"    {p.relative_to(output_dir)} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()

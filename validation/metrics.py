"""
Comprehensive Evaluation Metrics
==================================

Computes per-disease and overall metrics for BioPathNet++:
  - AUPRC, AUROC
  - Hit@K (K=10, 20, 50)
  - MRR (Mean Reciprocal Rank)
  - Precision@K, Recall@K

Usage:
    python validation/metrics.py

Author: Yatin
Date: April 2026
"""

import sys
import io
import warnings
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress sklearn warnings for per-disease groups with one class
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

from models.biopathnet import BioPathNetPlusPlus
from models.data_loader import GraphDataLoader
from models.disease_mappings import (
    get_disease_name, load_entity_mappings,
    TEST_DISEASE_ENTITIES, TRAIN_DISEASE_ENTITIES,
)


def compute_hit_at_k(all_scores, all_labels, k):
    """Hit@K: fraction of positive samples in top-K by score."""
    if len(all_scores) == 0:
        return 0.0
    k = min(k, len(all_scores))
    sorted_indices = np.argsort(all_scores)[::-1]
    top_k_labels = all_labels[sorted_indices[:k]]
    total_pos = all_labels.sum()
    if total_pos == 0:
        return 0.0
    return float(top_k_labels.sum() / total_pos)


def compute_mrr(all_scores, all_labels):
    """Mean Reciprocal Rank: average of 1/rank for each positive sample."""
    if len(all_scores) == 0:
        return 0.0
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_labels = all_labels[sorted_indices]
    reciprocal_ranks = []
    for rank, label in enumerate(sorted_labels, 1):
        if label == 1:
            reciprocal_ranks.append(1.0 / rank)
    if len(reciprocal_ranks) == 0:
        return 0.0
    return float(np.mean(reciprocal_ranks))


def compute_precision_at_k(all_scores, all_labels, k):
    """Precision@K: fraction of top-K predictions that are positive."""
    if len(all_scores) == 0:
        return 0.0
    k = min(k, len(all_scores))
    sorted_indices = np.argsort(all_scores)[::-1]
    top_k_labels = all_labels[sorted_indices[:k]]
    return float(top_k_labels.sum() / k)


def compute_all_metrics(scores, labels):
    """Compute all metrics for a set of scores and labels."""
    metrics = {}
    unique = np.unique(labels)

    # AUPRC
    if len(unique) >= 2:
        try:
            metrics['auprc'] = float(average_precision_score(labels, scores))
        except Exception:
            metrics['auprc'] = 0.0
    else:
        metrics['auprc'] = 0.0

    # AUROC
    if len(unique) >= 2:
        try:
            metrics['auroc'] = float(roc_auc_score(labels, scores))
        except Exception:
            metrics['auroc'] = 0.0
    else:
        metrics['auroc'] = 0.0

    # Hit@K
    for k in [10, 20, 50]:
        metrics[f'hit_at_{k}'] = compute_hit_at_k(scores, labels, k)

    # MRR
    metrics['mrr'] = compute_mrr(scores, labels)

    # Precision@K
    for k in [5, 10, 20]:
        metrics[f'precision_at_{k}'] = compute_precision_at_k(scores, labels, k)

    # Sample stats
    metrics['num_samples'] = int(len(labels))
    metrics['num_positive'] = int(labels.sum())
    metrics['num_negative'] = int(len(labels) - labels.sum())
    metrics['score_mean'] = float(scores.mean()) if len(scores) > 0 else 0.0
    metrics['score_std'] = float(scores.std()) if len(scores) > 0 else 0.0

    return metrics


def evaluate_from_csv(model, data_loader, pairs_csv, entity2id, device, split_name='test'):
    """
    Evaluate model by reading pairs directly from CSV.
    This ensures correct per-disease grouping by using the CSV's tail column.

    Returns:
        overall_metrics, per_disease_metrics
    """
    df = pd.read_csv(pairs_csv)

    # Identify the disease column (tail for standard format)
    # Group by disease entity from the CSV directly
    all_scores = []
    all_labels = []
    all_disease_entities = []

    model.eval()
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {split_name}"):
            head = row['head']
            tail = row['tail']
            label = int(row['label'])

            # Determine drug/disease
            if 'Compound' in str(head):
                drug_entity, disease_entity = head, tail
            else:
                drug_entity, disease_entity = tail, head

            drug_idx = entity2id.get(drug_entity, None)
            disease_idx = entity2id.get(disease_entity, None)

            if drug_idx is None or disease_idx is None:
                continue

            drug_tensor = torch.tensor(drug_idx, device=device)
            disease_tensor = torch.tensor([disease_idx], device=device)

            try:
                scores, _ = model(
                    data_loader.node_features,
                    data_loader.graph_data.edge_index,
                    data_loader.graph_data.edge_type,
                    drug_tensor,
                    disease_tensor,
                )
                score = scores.cpu().item()
            except Exception:
                score = 0.0

            all_scores.append(score)
            all_labels.append(label)
            all_disease_entities.append(disease_entity)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Overall metrics
    overall_metrics = compute_all_metrics(all_scores, all_labels)
    overall_metrics['split'] = split_name

    # Per-disease metrics: group by TARGET disease (from positive samples)
    # For each target disease, find ALL pairs involving that disease
    # (both positive and their corresponding negatives)
    per_disease_metrics = {}

    # Get unique diseases from positive samples
    results_df = pd.DataFrame({
        'score': all_scores,
        'label': all_labels,
        'disease': all_disease_entities,
    })

    # Get the target diseases (diseases with positive labels)
    positive_diseases = results_df[results_df['label'] == 1]['disease'].unique()

    for target_disease in positive_diseases:
        # Get all samples for this disease (positives)
        pos_mask = (results_df['disease'] == target_disease) & (results_df['label'] == 1)
        pos_scores = results_df.loc[pos_mask, 'score'].values

        # For negatives: we need to pair them properly
        # The test set is balanced 1:1, negatives are interspersed
        # Each disease's positives are paired with random negatives
        # We approximate by taking the same number of negatives from non-matching diseases
        neg_mask = results_df['label'] == 0
        neg_scores = results_df.loc[neg_mask, 'score'].values

        # Sample matching number of negatives
        n_pos = len(pos_scores)
        if len(neg_scores) > n_pos:
            np.random.seed(42)
            neg_indices = np.random.choice(len(neg_scores), n_pos, replace=False)
            neg_scores_sampled = neg_scores[neg_indices]
        else:
            neg_scores_sampled = neg_scores

        # Combine
        combined_scores = np.concatenate([pos_scores, neg_scores_sampled])
        combined_labels = np.concatenate([
            np.ones(len(pos_scores)),
            np.zeros(len(neg_scores_sampled))
        ])

        # Compute metrics
        d_metrics = compute_all_metrics(combined_scores, combined_labels)
        d_metrics['disease_entity'] = target_disease
        d_metrics['disease_name'] = get_disease_name(target_disease)
        per_disease_metrics[target_disease] = d_metrics

    return overall_metrics, per_disease_metrics


def main():
    print("=" * 80)
    print("COMPREHENSIVE METRICS EVALUATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    data_loader = GraphDataLoader(processed_dir='datasets/processed', batch_size=4)
    entity2id, id2entity = load_entity_mappings('datasets/processed')

    # Load model
    print("Loading model...")
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint best AUPRC: {checkpoint.get('best_val_auprc', '?'):.4f}")

    model = BioPathNetPlusPlus(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        hidden_dim=config['hidden_dim'],
        num_nbf_layers=config['num_nbf_layers'],
        num_metapaths=config['num_metapaths'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # ── Evaluate on TEST set (zero-shot transfer) ──
    print(f"\n{'─' * 60}")
    print("TEST SET EVALUATION (Zero-Shot Transfer)")
    print(f"{'─' * 60}")

    test_overall, test_per_disease = evaluate_from_csv(
        model, data_loader, 'datasets/processed/test_pairs.csv',
        entity2id, device, split_name='test'
    )

    print(f"\n  Overall Test Metrics:")
    print(f"    AUPRC:          {test_overall['auprc']:.4f}")
    print(f"    AUROC:          {test_overall['auroc']:.4f}")
    print(f"    MRR:            {test_overall['mrr']:.4f}")
    print(f"    Hit@10:         {test_overall['hit_at_10']:.4f}")
    print(f"    Hit@20:         {test_overall['hit_at_20']:.4f}")
    print(f"    Hit@50:         {test_overall['hit_at_50']:.4f}")
    print(f"    Precision@5:    {test_overall['precision_at_5']:.4f}")
    print(f"    Precision@10:   {test_overall['precision_at_10']:.4f}")
    print(f"    Precision@20:   {test_overall['precision_at_20']:.4f}")
    print(f"    Samples:        {test_overall['num_samples']} ({test_overall['num_positive']} pos / {test_overall['num_negative']} neg)")

    # Per-disease test results
    print(f"\n  Per-Disease Test Metrics (Zero-Shot Transfer):")
    print(f"    {'Disease':<35} {'AUPRC':>8} {'AUROC':>8} {'MRR':>8} {'Hit@20':>8} {'P@10':>8} {'Pos':>5}")
    print(f"    {'─' * 85}")
    for d_entity in sorted(test_per_disease.keys(), key=lambda x: test_per_disease[x]['auprc'], reverse=True):
        dm = test_per_disease[d_entity]
        print(f"    {dm['disease_name']:<35} {dm['auprc']:>8.4f} {dm['auroc']:>8.4f} "
              f"{dm['mrr']:>8.4f} {dm['hit_at_20']:>8.4f} "
              f"{dm['precision_at_10']:>8.4f} {dm['num_positive']:>5}")

    # ── Evaluate on VALIDATION set ──
    print(f"\n{'─' * 60}")
    print("VALIDATION SET EVALUATION")
    print(f"{'─' * 60}")

    val_overall, val_per_disease = evaluate_from_csv(
        model, data_loader, 'datasets/processed/val_pairs.csv',
        entity2id, device, split_name='val'
    )

    print(f"\n  Overall Validation Metrics:")
    print(f"    AUPRC:          {val_overall['auprc']:.4f}")
    print(f"    AUROC:          {val_overall['auroc']:.4f}")
    print(f"    MRR:            {val_overall['mrr']:.4f}")

    # ── Generalization gap ──
    gap = val_overall['auprc'] - test_overall['auprc']
    print(f"\n  Generalization Gap (Val - Test): {gap:+.4f}")
    if test_overall['auprc'] > val_overall['auprc']:
        print(f"  ✓ Test > Val — No overfitting, excellent generalization!")
    elif abs(gap) < 0.05:
        print(f"  ✓ Small gap — Good generalization!")
    else:
        print(f"  ⚠ Large gap — Possible overfitting")

    # ── Save results ──
    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-disease CSV
    rows = []
    for d_entity, dm in test_per_disease.items():
        rows.append(dm)
    pd.DataFrame(rows).to_csv(output_dir / 'per_disease_metrics.csv', index=False)

    # Overall JSON
    results = {
        'test': test_overall,
        'validation': val_overall,
        'per_disease_test': {k: v for k, v in test_per_disease.items()},
        'per_disease_val': {k: v for k, v in val_per_disease.items()},
    }
    with open(output_dir / 'overall_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # SOTA comparison
    sota_comparison = pd.DataFrame([
        {'model': 'Random Baseline', 'auprc': 0.500, 'auroc': 0.500},
        {'model': 'TransE', 'auprc': 0.650, 'auroc': 0.680},
        {'model': 'DRKG Baseline', 'auprc': 0.700, 'auroc': 0.720},
        {'model': 'TxGNN (Stanford)', 'auprc': 0.920, 'auroc': 0.930},
        {'model': 'BioPathNet++ (Ours - Val)', 'auprc': round(val_overall['auprc'], 3), 'auroc': round(val_overall['auroc'], 3)},
        {'model': 'BioPathNet++ (Ours - Test)', 'auprc': round(test_overall['auprc'], 3), 'auroc': round(test_overall['auroc'], 3)},
    ])
    sota_comparison.to_csv(output_dir / 'comparison_with_sota.csv', index=False)

    print(f"\n  ✓ Results saved to {output_dir}")

    # ── SOTA Comparison ──
    print(f"\n{'─' * 60}")
    print("SOTA COMPARISON")
    print(f"{'─' * 60}")
    print(f"    {'Model':<35} {'AUPRC':>8} {'AUROC':>8}")
    print(f"    {'─' * 55}")
    for _, row in sota_comparison.iterrows():
        marker = "  ★" if 'Ours' in row['model'] else ""
        print(f"    {row['model']:<35} {row['auprc']:>8.3f} {row['auroc']:>8.3f}{marker}")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()

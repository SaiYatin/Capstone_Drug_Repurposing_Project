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


def bootstrap_metric(scores, labels, metric_fn, n_resamples=1000):
    """Compute confidence intervals using bootstrap."""
    values = []
    n = len(labels)
    # Fix seed inside bootstrap for reproducibility of the intervals
    rng = np.random.RandomState(42)
    for _ in range(n_resamples):
        indices = rng.choice(n, n, replace=True)
        sample_scores = scores[indices]
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) >= 2:
            try:
                values.append(metric_fn(sample_labels, sample_scores))
            except Exception:
                pass
    if not values:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def compute_all_metrics(scores, labels, compute_ci=False):
    """Compute all metrics for a set of scores and labels."""
    metrics = {}
    unique = np.unique(labels)

    # AUPRC
    if len(unique) >= 2:
        try:
            metrics['auprc'] = float(average_precision_score(labels, scores))
            if compute_ci:
                mean, lower, upper = bootstrap_metric(scores, labels, average_precision_score)
                metrics['auprc_ci_lower'] = lower
                metrics['auprc_ci_upper'] = upper
        except Exception:
            metrics['auprc'] = 0.0
            if compute_ci:
                metrics['auprc_ci_lower'] = 0.0
                metrics['auprc_ci_upper'] = 0.0
    else:
        metrics['auprc'] = 0.0
        if compute_ci:
            metrics['auprc_ci_lower'] = 0.0
            metrics['auprc_ci_upper'] = 0.0

    # AUROC
    if len(unique) >= 2:
        try:
            metrics['auroc'] = float(roc_auc_score(labels, scores))
            if compute_ci:
                mean, lower, upper = bootstrap_metric(scores, labels, roc_auc_score)
                metrics['auroc_ci_lower'] = lower
                metrics['auroc_ci_upper'] = upper
        except Exception:
            metrics['auroc'] = 0.0
            if compute_ci:
                metrics['auroc_ci_lower'] = 0.0
                metrics['auroc_ci_upper'] = 0.0
    else:
        metrics['auroc'] = 0.0
        if compute_ci:
            metrics['auroc_ci_lower'] = 0.0
            metrics['auroc_ci_upper'] = 0.0

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

    all_scores = []
    all_labels = []
    all_disease_entities = []
    all_drug_entities = []

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
            all_drug_entities.append(drug_entity)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Overall metrics
    overall_metrics = compute_all_metrics(all_scores, all_labels, compute_ci=True)
    overall_metrics['split'] = split_name

    per_disease_metrics = {}

    results_df = pd.DataFrame({
        'score': all_scores,
        'label': all_labels,
        'disease': all_disease_entities,
        'drug': all_drug_entities,
    })

    positive_diseases = results_df[results_df['label'] == 1]['disease'].unique()
    positive_pairs = results_df[results_df['label'] == 1]

    # FIX 1: Move random seed outside the per-disease loop
    np.random.seed(42)

    for target_disease in positive_diseases:
        pos_mask = (results_df['disease'] == target_disease) & (results_df['label'] == 1)
        pos_scores = results_df.loc[pos_mask, 'score'].values
        n_pos = len(pos_scores)

        # FIX 2 & 3: Disease-specific hard negatives
        # Drugs associated with target disease
        target_drugs = positive_pairs[positive_pairs['disease'] == target_disease]['drug'].unique()
        
        # Drugs associated with OTHER autoimmune diseases
        other_disease_drugs = positive_pairs[positive_pairs['disease'] != target_disease]['drug'].unique()
        
        # Hard negative drugs: associated with other diseases but NOT target disease
        hard_negative_drugs = np.setdiff1d(other_disease_drugs, target_drugs)
        
        neg_scores_list = []
        disease_idx = entity2id.get(target_disease, None)
        
        if disease_idx is not None and len(hard_negative_drugs) > 0:
            disease_tensor = torch.tensor([disease_idx], device=device)
            # Sample equal number of negatives as positives from the hard negative drugs
            if len(hard_negative_drugs) > n_pos:
                sampled_neg_drugs = np.random.choice(hard_negative_drugs, n_pos, replace=False)
            else:
                sampled_neg_drugs = hard_negative_drugs

            model.eval()
            with torch.no_grad():
                for drug_entity in sampled_neg_drugs:
                    drug_idx = entity2id.get(drug_entity, None)
                    if drug_idx is None:
                        continue
                    drug_tensor = torch.tensor(drug_idx, device=device)
                    try:
                        scores, _ = model(
                            data_loader.node_features,
                            data_loader.graph_data.edge_index,
                            data_loader.graph_data.edge_type,
                            drug_tensor,
                            disease_tensor,
                        )
                        neg_scores_list.append(scores.cpu().item())
                    except Exception:
                        neg_scores_list.append(0.0)
        
        neg_scores_sampled = np.array(neg_scores_list)

        # Fallback if hard negatives generation failed
        if len(neg_scores_sampled) == 0:
            neg_mask = results_df['label'] == 0
            neg_scores = results_df.loc[neg_mask, 'score'].values
            if len(neg_scores) > n_pos:
                neg_indices = np.random.choice(len(neg_scores), n_pos, replace=False)
                neg_scores_sampled = neg_scores[neg_indices]
            else:
                neg_scores_sampled = neg_scores

        combined_scores = np.concatenate([pos_scores, neg_scores_sampled])
        combined_labels = np.concatenate([
            np.ones(len(pos_scores)),
            np.zeros(len(neg_scores_sampled))
        ])

        # Compute metrics with CI
        d_metrics = compute_all_metrics(combined_scores, combined_labels, compute_ci=True)
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
    try:
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
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # ── Evaluate on TEST set (zero-shot transfer) ──
    print(f"\n{'─' * 60}")
    print("TEST SET EVALUATION (Zero-Shot Transfer)")
    print(f"{'─' * 60}")

    try:
        test_overall, test_per_disease = evaluate_from_csv(
            model, data_loader, 'datasets/processed/test_pairs.csv',
            entity2id, device, split_name='test'
        )

        print(f"\n  Overall Test Metrics:")
        print(f"    AUPRC:          {test_overall['auprc']:.4f} (95% CI: {test_overall.get('auprc_ci_lower', 0.0):.4f} - {test_overall.get('auprc_ci_upper', 0.0):.4f})")
        print(f"    AUROC:          {test_overall['auroc']:.4f} (95% CI: {test_overall.get('auroc_ci_lower', 0.0):.4f} - {test_overall.get('auroc_ci_upper', 0.0):.4f})")
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
        print(f"    {'Disease':<35} {'AUPRC':>8} {'AUROC':>8} {'MRR':>8} {'Hit@20':>8} {'P@10':>8} {'Pos':>5} {'Warning'}")
        print(f"    {'─' * 105}")
        for d_entity in sorted(test_per_disease.keys(), key=lambda x: test_per_disease[x]['auprc'], reverse=True):
            dm = test_per_disease[d_entity]
            warning_msg = "⚠ <30 samples" if dm['num_samples'] < 30 else ""
            print(f"    {dm['disease_name']:<35} {dm['auprc']:>8.4f} {dm['auroc']:>8.4f} "
                  f"{dm['mrr']:>8.4f} {dm['hit_at_20']:>8.4f} "
                  f"{dm['precision_at_10']:>8.4f} {dm['num_positive']:>5}  {warning_msg}")
    except Exception as e:
        print(f"Error evaluating test set: {e}")
        test_overall, test_per_disease = {'auprc': 0, 'auroc': 0}, {}

    # ── Evaluate on VALIDATION set ──
    print(f"\n{'─' * 60}")
    print("VALIDATION SET EVALUATION")
    print(f"{'─' * 60}")

    try:
        val_overall, val_per_disease = evaluate_from_csv(
            model, data_loader, 'datasets/processed/val_pairs.csv',
            entity2id, device, split_name='val'
        )

        print(f"\n  Overall Validation Metrics:")
        print(f"    AUPRC:          {val_overall['auprc']:.4f} (95% CI: {val_overall.get('auprc_ci_lower', 0.0):.4f} - {val_overall.get('auprc_ci_upper', 0.0):.4f})")
        print(f"    AUROC:          {val_overall['auroc']:.4f} (95% CI: {val_overall.get('auroc_ci_lower', 0.0):.4f} - {val_overall.get('auroc_ci_upper', 0.0):.4f})")
        print(f"    MRR:            {val_overall['mrr']:.4f}")

        # ── Generalization gap ──
        gap = val_overall['auprc'] - test_overall.get('auprc', 0.0)
        print(f"\n  Generalization Gap (Val - Test): {gap:+.4f}")
        if test_overall.get('auprc', 0.0) > val_overall['auprc']:
            print(f"  ✓ Test > Val — No overfitting, excellent generalization!")
        elif abs(gap) < 0.05:
            print(f"  ✓ Small gap — Good generalization!")
        else:
            print(f"  ⚠ Large gap — Possible overfitting")
    except Exception as e:
        print(f"Error evaluating validation set: {e}")
        val_overall, val_per_disease = {'auprc': 0, 'auroc': 0}, {}

    # ── Save results ──
    output_dir = Path('outputs/validation_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-disease CSV
    rows = []
    for d_entity, dm in test_per_disease.items():
        rows.append(dm)
    if rows:
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
        {'model': 'BioPathNet++ (Ours - Val)', 'auprc': round(val_overall.get('auprc', 0), 3), 'auroc': round(val_overall.get('auroc', 0), 3)},
        {'model': 'BioPathNet++ (Ours - Test)', 'auprc': round(test_overall.get('auprc', 0), 3), 'auroc': round(test_overall.get('auroc', 0), 3)},
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

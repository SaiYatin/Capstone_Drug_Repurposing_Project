"""
Few-Shot Fine-Tuning Experiments (v2 - Fixed)
===============================================

Compares zero-shot vs few-shot transfer for rare diseases.
Fixed: properly handles negatives by sampling from the full
negative pool rather than trying to match per-disease.

Usage:
    python transfer_learning/few_shot.py

Author: Yatin
Date: April 2026
"""

import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import copy
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from models.biopathnet import BioPathNetPlusPlus
from models.data_loader import GraphDataLoader
from models.disease_mappings import (
    get_disease_name, get_disease_short_name,
    load_entity_mappings, TEST_DISEASE_ENTITIES,
)


def build_disease_eval_set(pairs_file, disease_entity, entity2id):
    """
    Build a balanced evaluation set for a disease.
    Takes all positives for that disease + matched number of negatives.
    """
    df = pd.read_csv(pairs_file)

    positives = []
    negatives = []

    for _, row in df.iterrows():
        head, tail, label = row['head'], row['tail'], int(row['label'])

        if 'Compound' in str(head):
            drug_entity, dis_entity = head, tail
        else:
            drug_entity, dis_entity = tail, head

        drug_idx = entity2id.get(drug_entity)
        disease_idx = entity2id.get(dis_entity)

        if drug_idx is None or disease_idx is None:
            continue

        pair = {
            'drug_entity': drug_entity,
            'drug_idx': drug_idx,
            'disease_entity': dis_entity,
            'disease_idx': disease_idx,
            'label': label,
        }

        # Positives: must match target disease
        if label == 1 and dis_entity == disease_entity:
            positives.append(pair)
        # Negatives: any negative will do (they're random drug-disease pairs)
        elif label == 0:
            negatives.append(pair)

    # Balance: take same number of negatives as positives
    n_pos = len(positives)
    if len(negatives) > n_pos:
        np.random.seed(42)
        neg_indices = np.random.choice(len(negatives), n_pos, replace=False)
        negatives = [negatives[i] for i in neg_indices]

    all_pairs = positives + negatives
    np.random.seed(42)
    np.random.shuffle(all_pairs)

    return all_pairs, len(positives), len(negatives)


def evaluate_model(model, data_loader, pairs, device):
    """Evaluate model on a set of pairs."""
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for pair in pairs:
            drug_idx = torch.tensor(pair['drug_idx'], device=device)
            disease_idx = torch.tensor([pair['disease_idx']], device=device)

            out, _ = model(
                data_loader.node_features,
                data_loader.graph_data.edge_index,
                data_loader.graph_data.edge_type,
                drug_idx,
                disease_idx,
            )
            scores.append(out.cpu().item())
            labels.append(pair['label'])

    scores = np.array(scores)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        return 0.0

    try:
        return average_precision_score(labels, scores)
    except Exception:
        return 0.0


def few_shot_finetune(model, data_loader, support_pairs, device, epochs=20, lr=1e-5):
    """Fine-tune only prediction head on K support examples."""
    ft_model = copy.deepcopy(model)
    ft_model.to(device)

    # Freeze everything except predictor/fusion
    for name, param in ft_model.named_parameters():
        if 'predictor' not in name and 'fusion' not in name:
            param.requires_grad = False

    trainable_params = [p for p in ft_model.parameters() if p.requires_grad]
    if not trainable_params:
        print("  ⚠ No trainable parameters found!")
        return ft_model

    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    ft_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pair in support_pairs:
            drug_idx = torch.tensor(pair['drug_idx'], device=device)
            disease_idx = torch.tensor([pair['disease_idx']], device=device)
            label = torch.tensor([float(pair['label'])], device=device)

            optimizer.zero_grad()
            scores, _ = ft_model(
                data_loader.node_features,
                data_loader.graph_data.edge_index,
                data_loader.graph_data.edge_type,
                drug_idx,
                disease_idx,
            )
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return ft_model


def main():
    print("=" * 80)
    print("FEW-SHOT FINE-TUNING EXPERIMENTS (v2)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading data...")
    data_loader = GraphDataLoader(processed_dir='datasets/processed', batch_size=4)
    entity2id, id2entity = load_entity_mappings('datasets/processed')

    print("Loading model...")
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
    config = checkpoint['config']

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

    test_pairs_file = 'datasets/processed/test_pairs.csv'

    results = []
    k_shots = [5, 10]

    for disease_entity in TEST_DISEASE_ENTITIES:
        disease_name = get_disease_name(disease_entity)
        print(f"\n{'─' * 60}")
        print(f"Disease: {disease_name}")
        print(f"{'─' * 60}")

        # Build balanced eval set
        all_pairs, n_pos, n_neg = build_disease_eval_set(
            test_pairs_file, disease_entity, entity2id
        )
        print(f"  Eval set: {n_pos} pos + {n_neg} neg = {len(all_pairs)} pairs")

        if n_pos < 3:
            print(f"  ⚠ Too few positives ({n_pos}) — skipping")
            continue

        # Zero-shot evaluation
        zs_auprc = evaluate_model(model, data_loader, all_pairs, device)
        print(f"  Zero-shot AUPRC: {zs_auprc:.4f}")

        result_row = {
            'disease_entity': disease_entity,
            'disease_name': disease_name,
            'num_positives': n_pos,
            'num_negatives': n_neg,
            'total_pairs': len(all_pairs),
            'zero_shot_auprc': zs_auprc,
        }

        for k in k_shots:
            if n_pos < k + 2:
                print(f"  ⚠ Not enough positives for {k}-shot (have {n_pos})")
                result_row[f'{k}_shot_auprc'] = None
                result_row[f'{k}_shot_improvement'] = None
                continue

            # Split positives: first K as support, rest for query
            pos_pairs = [p for p in all_pairs if p['label'] == 1]
            neg_pairs = [p for p in all_pairs if p['label'] == 0]

            np.random.seed(42)
            perm = np.random.permutation(len(pos_pairs))
            support_pos = [pos_pairs[i] for i in perm[:k]]
            query_pos = [pos_pairs[i] for i in perm[k:]]

            # Support = K pos + K neg
            support_neg = neg_pairs[:k]
            support = support_pos + support_neg

            # Query = remaining pos + remaining neg
            query = query_pos + neg_pairs[k:]

            if len(query) < 4 or len([p for p in query if p['label'] == 1]) < 2:
                print(f"  ⚠ Not enough query samples for {k}-shot")
                result_row[f'{k}_shot_auprc'] = None
                result_row[f'{k}_shot_improvement'] = None
                continue

            # Fine-tune
            print(f"  Fine-tuning {k}-shot ({k} pos + {k} neg support)...")
            ft_model = few_shot_finetune(model, data_loader, support, device,
                                          epochs=20, lr=1e-5)

            # Evaluate on query set
            fs_auprc = evaluate_model(ft_model, data_loader, query, device)
            improvement = fs_auprc - zs_auprc

            print(f"  {k}-shot AUPRC: {fs_auprc:.4f} (Δ = {improvement:+.4f})")

            result_row[f'{k}_shot_auprc'] = fs_auprc
            result_row[f'{k}_shot_improvement'] = improvement

            del ft_model
            torch.cuda.empty_cache()

        results.append(result_row)

    # Save
    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'few_shot_comparison.csv', index=False)

    print(f"\n{'=' * 80}")
    print("FEW-SHOT SUMMARY")
    print(f"{'=' * 80}")
    for _, row in results_df.iterrows():
        zs = f"{row['zero_shot_auprc']:.4f}"
        fs5 = f"{row.get('5_shot_auprc', 0):.4f}" if pd.notna(row.get('5_shot_auprc')) else 'N/A'
        fs10 = f"{row.get('10_shot_auprc', 0):.4f}" if pd.notna(row.get('10_shot_auprc')) else 'N/A'
        print(f"  {row['disease_name']:<35} ZS: {zs}  5-shot: {fs5}  10-shot: {fs10}")

    print(f"\n✓ Results saved to {output_dir / 'few_shot_comparison.csv'}")


if __name__ == '__main__':
    main()

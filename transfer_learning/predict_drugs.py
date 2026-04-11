"""
Zero-Shot Drug Prediction — Disease-Centric Approach (v2)
==========================================================

Instead of scoring ALL 10,572 drugs (most of which have no path to the
disease), this version:
  1. Finds all drugs in the disease's K-hop neighborhood (K=3)
  2. Scores ONLY those drugs (they have actual paths to the disease)
  3. This gives meaningful score differences instead of constant ~3.3

Usage:
    python transfer_learning/predict_drugs.py

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
import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph

from models.biopathnet import BioPathNetPlusPlus
from models.data_loader import GraphDataLoader
from models.disease_mappings import (
    get_disease_name, get_disease_short_name, get_drug_name,
    get_drug_name_with_lookup, get_drug_id, get_gene_symbol,
    load_entity_mappings, _load_drug_cache, _save_drug_cache,
    TEST_DISEASE_ENTITIES, ALL_DISEASE_ENTITIES,
)


def load_model(checkpoint_path, data_loader, device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    model.eval()

    print(f"  Epoch: {checkpoint.get('epoch', '?')}, Val AUPRC: {checkpoint.get('best_val_auprc', '?'):.4f}")
    return model, config


def find_candidate_drugs(disease_entity, entity2id, id2entity, edge_index, num_nodes, num_hops=3):
    """
    Find all compound nodes within K hops of the disease in the KG.
    These are drugs with actual biological paths to the disease.
    """
    disease_idx = entity2id.get(disease_entity)
    if disease_idx is None:
        return [], []

    disease_tensor = torch.tensor([disease_idx])

    # Use PyG's k_hop_subgraph to find neighborhood
    node_subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=disease_tensor,
        num_hops=num_hops,
        edge_index=edge_index.cpu(),
        num_nodes=num_nodes,
        relabel_nodes=False,
    )

    # Filter for compounds
    candidate_drugs = []
    candidate_indices = []
    for node_idx in node_subset.tolist():
        entity = id2entity.get(node_idx, "")
        if entity.startswith("Compound::"):
            candidate_drugs.append(entity)
            candidate_indices.append(node_idx)

    return candidate_drugs, candidate_indices


def find_drug_paths(drug_entity, disease_entity, entity2id, id2entity, edge_index):
    """
    Find intermediate gene nodes between a drug and disease (1 or 2 hops).
    Returns list of gene symbols on the paths.
    """
    drug_idx = entity2id.get(drug_entity)
    disease_idx = entity2id.get(disease_entity)
    if drug_idx is None or disease_idx is None:
        return []

    # Find direct neighbors of drug
    drug_mask_h = edge_index[0] == drug_idx
    drug_mask_t = edge_index[1] == drug_idx
    drug_neighbors = set()
    drug_neighbors.update(edge_index[1][drug_mask_h].tolist())
    drug_neighbors.update(edge_index[0][drug_mask_t].tolist())

    # Find direct neighbors of disease
    dis_mask_h = edge_index[0] == disease_idx
    dis_mask_t = edge_index[1] == disease_idx
    dis_neighbors = set()
    dis_neighbors.update(edge_index[1][dis_mask_h].tolist())
    dis_neighbors.update(edge_index[0][dis_mask_t].tolist())

    # Find shared neighbors (genes on the path)
    shared = drug_neighbors & dis_neighbors
    path_genes = []
    for idx in shared:
        entity = id2entity.get(idx, "")
        if entity.startswith("Gene::"):
            symbol = get_gene_symbol(entity)
            path_genes.append(symbol)

    return path_genes


def predict_drugs_for_disease(
    model, data_loader, disease_entity, entity2id, id2entity,
    edge_index, device, top_k=50, num_hops=3, known_positive_drugs=None,
):
    """
    Disease-centric drug prediction:
    1. Find all drugs within num_hops of the disease
    2. Score each one
    3. Return ranked list with paths
    """
    if known_positive_drugs is None:
        known_positive_drugs = set()

    disease_name = get_disease_name(disease_entity)
    disease_idx = entity2id.get(disease_entity)

    if disease_idx is None:
        print(f"  ⚠ Disease '{disease_entity}' not found!")
        return pd.DataFrame()

    # Step 1: Find candidate drugs in disease neighborhood
    print(f"  Finding drugs within {num_hops} hops of {disease_name}...")
    candidates, candidate_indices = find_candidate_drugs(
        disease_entity, entity2id, id2entity,
        edge_index, len(entity2id), num_hops=num_hops,
    )
    print(f"  Found {len(candidates)} candidate drugs")

    if len(candidates) == 0:
        print(f"  ⚠ No drugs found within {num_hops} hops!")
        return pd.DataFrame()

    # Step 2: Score each candidate
    all_scores = []
    all_attentions = []
    disease_idx_tensor = torch.tensor([disease_idx], device=device)

    model.eval()
    with torch.no_grad():
        for drug_entity in tqdm(candidates, desc="  Scoring", leave=False):
            drug_idx = entity2id[drug_entity]
            drug_idx_tensor = torch.tensor(drug_idx, device=device)

            try:
                scores, attn_weights = model(
                    data_loader.node_features,
                    data_loader.graph_data.edge_index,
                    data_loader.graph_data.edge_type,
                    drug_idx_tensor,
                    disease_idx_tensor,
                )
                all_scores.append(scores.cpu().item())
                if attn_weights is not None:
                    all_attentions.append(attn_weights.cpu().numpy().tolist())
                else:
                    all_attentions.append([0.0] * 7)
            except Exception as e:
                all_scores.append(float('-inf'))
                all_attentions.append([0.0] * 7)

    # Step 3: Find paths for each drug
    edge_index_cpu = edge_index.cpu()
    path_info = []
    for drug_entity in candidates:
        path_genes = find_drug_paths(
            drug_entity, disease_entity,
            entity2id, id2entity, edge_index_cpu,
        )
        path_info.append(path_genes)

    # Build DataFrame
    df = pd.DataFrame({
        'drug_entity': candidates,
        'score': all_scores,
        'attention_weights': all_attentions,
        'path_genes': path_info,
    })

    # Sort by score descending
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    # Add metadata
    df['rank'] = range(1, len(df) + 1)
    df['drug_id'] = df['drug_entity'].apply(get_drug_id)
    df['drug_name'] = df['drug_entity'].apply(get_drug_name)
    df['disease_entity'] = disease_entity
    df['disease_name'] = disease_name
    df['known_drug'] = df['drug_entity'].isin(known_positive_drugs)
    df['path_genes_str'] = df['path_genes'].apply(lambda x: ', '.join(x) if x else 'indirect')

    # Reorder columns
    cols = ['rank', 'drug_entity', 'drug_id', 'drug_name', 'score',
            'known_drug', 'disease_entity', 'disease_name',
            'path_genes_str', 'attention_weights']
    df = df[cols]

    return df.head(top_k)


def get_known_drugs_per_disease(entity2id):
    """Get known positive drugs from train + val + test pairs."""
    known = {}
    for pairs_file in ['datasets/processed/train_pairs.csv',
                       'datasets/processed/val_pairs.csv',
                       'datasets/processed/test_pairs.csv']:
        if not Path(pairs_file).exists():
            continue
        df = pd.read_csv(pairs_file)
        pos = df[df['label'] == 1]
        for _, row in pos.iterrows():
            head, tail = row['head'], row['tail']
            if 'Compound' in str(head):
                drug, disease = head, tail
            else:
                drug, disease = tail, head
            known.setdefault(disease, set()).add(drug)
    return known


def main():
    print("=" * 80)
    print("PHASE 3: ZERO-SHOT DRUG PREDICTION (Disease-Centric)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading graph data...")
    data_loader = GraphDataLoader(processed_dir='datasets/processed', batch_size=4)
    entity2id, id2entity = load_entity_mappings('datasets/processed')

    edge_index = data_loader.graph_data.edge_index

    # Load model
    print("\nLoading model...")
    model, config = load_model('checkpoints/best_model.pt', data_loader, device)

    # Known drugs
    known_drugs = get_known_drugs_per_disease(entity2id)

    # Output dirs
    output_dir = Path('outputs/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load drug name cache
    _load_drug_cache()

    # Predict for each target disease
    all_results = []
    target_diseases = TEST_DISEASE_ENTITIES

    print(f"\n{'=' * 80}")
    print(f"Predicting drugs for {len(target_diseases)} target diseases")
    print(f"Using 3-hop neighborhood (disease-centric)")
    print(f"{'=' * 80}")

    for disease_entity in target_diseases:
        disease_name = get_disease_name(disease_entity)
        short_name = get_disease_short_name(disease_entity)

        print(f"\n{'─' * 60}")
        print(f"Disease: {disease_name} ({disease_entity})")
        print(f"{'─' * 60}")

        known = known_drugs.get(disease_entity, set())
        print(f"  Known positive drugs: {len(known)}")
        if known:
            for k in known:
                print(f"    - {get_drug_name(k)} ({get_drug_id(k)})")

        start_time = time.time()

        df = predict_drugs_for_disease(
            model=model,
            data_loader=data_loader,
            disease_entity=disease_entity,
            entity2id=entity2id,
            id2entity=id2entity,
            edge_index=edge_index,
            device=device,
            top_k=50,
            num_hops=3,
            known_positive_drugs=known,
        )

        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.1f}s")

        if len(df) == 0:
            print(f"  ⚠ No predictions generated!")
            continue

        # Try to look up unknown drug names via PubChem
        print(f"  Looking up drug names...")
        for idx, row in df.iterrows():
            if row['drug_name'] == row['drug_id'] and row['drug_id'].startswith('DB'):
                name = get_drug_name_with_lookup(row['drug_entity'])
                if name and name != row['drug_id']:
                    df.at[idx, 'drug_name'] = name
                time.sleep(0.1)  # Rate limit

        # Save per-disease CSV
        csv_path = output_dir / f"{short_name}_top50.csv"
        df_save = df.drop(columns=['attention_weights'])
        df_save.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path}")

        # Save attention data
        attn_path = output_dir / f"{short_name}_attention.json"
        attn_data = []
        for _, row in df.head(20).iterrows():
            attn_data.append({
                'rank': int(row['rank']),
                'drug': row['drug_name'],
                'drug_id': row['drug_id'],
                'score': float(row['score']),
                'attention': row['attention_weights'],
                'path_genes': row['path_genes_str'],
            })
        with open(attn_path, 'w') as f:
            json.dump(attn_data, f, indent=2)

        # Print top 20
        print(f"\n  Top 20 Drug Candidates for {disease_name}:")
        print(f"  {'Rank':<5} {'Drug Name':<30} {'Score':<10} {'Known?':<8} {'Path Genes'}")
        print(f"  {'─' * 90}")
        for _, row in df.head(20).iterrows():
            known_flag = "✓ YES" if row['known_drug'] else ""
            genes_str = row['path_genes_str'][:30] if row['path_genes_str'] else ""
            print(f"  {row['rank']:<5} {str(row['drug_name'])[:28]:<30} "
                  f"{row['score']:<10.4f} {known_flag:<8} {genes_str}")

        # Hit@K metrics
        for k in [10, 20, 50]:
            top_k_df = df.head(k)
            hits = top_k_df['known_drug'].sum()
            total_known = len(known)
            pct = (hits / total_known * 100) if total_known > 0 else 0
            print(f"  Hit@{k}: {hits}/{total_known} ({pct:.1f}%)")

        all_results.append(df)

    # Save drug name cache
    _save_drug_cache()

    # Save combined summary
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        summary_path = output_dir / "all_diseases_summary.csv"
        combined.drop(columns=['attention_weights']).to_csv(summary_path, index=False)
        print(f"\n✓ Combined summary saved: {summary_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("PREDICTION SUMMARY")
    print(f"{'=' * 80}")
    for disease_entity in target_diseases:
        disease_name = get_disease_name(disease_entity)
        known = known_drugs.get(disease_entity, set())
        csv_file = output_dir / f"{get_disease_short_name(disease_entity)}_top50.csv"
        if csv_file.exists():
            ddf = pd.read_csv(csv_file)
            print(f"  {disease_name}: {len(ddf)} candidates scored, "
                  f"{len(known)} known drugs, "
                  f"top score: {ddf['score'].max():.4f}")
        else:
            print(f"  {disease_name}: no predictions")

    print(f"\nAll predictions saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

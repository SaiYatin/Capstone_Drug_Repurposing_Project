"""
Predict Drugs using BioPathNet++ v2 (retrained model)
=====================================================

Loads the retrained v2 model and generates drug predictions
for all 6 zero-shot test diseases using hybrid scoring.

Author: Yatin
Date: October 2026
"""

import sys
import io
import math
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph

from models.biopathnet_v2 import BioPathNetPlusPlus  # v2 model
from models.data_loader import GraphDataLoader
from models.disease_mappings import (
    get_disease_name, get_disease_short_name, get_drug_name,
    get_drug_name_with_lookup, get_drug_id, get_gene_symbol,
    load_entity_mappings, _load_drug_cache, _save_drug_cache
)

TEST_DISEASES = [
    'Disease::MESH:D012859',  # Sjogren's
    'Disease::MESH:D012595',  # Systemic Sclerosis
    'Disease::MESH:D008105',  # Primary Biliary Cholangitis
    'Disease::MESH:D009157',  # Myasthenia Gravis
    'Disease::MESH:D019693',  # Autoimmune Hepatitis
    'Disease::MESH:D003424',  # Crohn's Disease
]

IMMUNE_GENES = {
    'JAK1', 'JAK2', 'JAK3', 'TYK2', 'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'TNF', 'TNFRSF1A', 'TNFRSF1B', 'TNFSF10', 'TRADD', 'TRAF2', 'RIPK1',
    'IL1A', 'IL1B', 'IL2', 'IL4', 'IL5', 'IL6', 'IL10', 'IL12A', 'IL12B', 'IL13', 'IL15', 'IL17A', 'IL17F', 'IL18', 'IL21', 'IL23A', 'IL33',
    'IFNG', 'IFNA1', 'IFNB1', 'CXCL8', 'CXCL10', 'CCL2', 'CCL5', 'CSF2',
    'NFKB1', 'NFKB2', 'RELA', 'IKBKB', 'CHUK',
    'CD4', 'CD8A', 'CD8B', 'CD3D', 'CD3E', 'CD28', 'CTLA4', 'PDCD1', 'FOXP3', 'TGFB1', 'TGFB3',
    'CD19', 'CD20', 'MS4A1', 'CD40', 'CD80', 'CD86', 'IGHG1', 'BLNK',
    'ICAM1', 'VCAM1', 'HGF', 'BAX', 'CDKN1A',
}


def load_v2_model(checkpoint_path, data_loader, device):
    """Load retrained v2 model."""
    print(f"Loading v2 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = BioPathNetPlusPlus(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        hidden_dim=256,
        num_nbf_layers=3,
        num_metapaths=7,
        num_attention_heads=4,
        dropout=0.1,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, Val AUPRC: {checkpoint.get('best_val_auprc', '?')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def find_candidate_drugs(disease_entity, entity2id, id2entity, edge_index, num_nodes, num_hops=3):
    """Find all compound nodes within K hops of the disease."""
    disease_idx = entity2id.get(disease_entity)
    if disease_idx is None:
        return [], []

    disease_tensor = torch.tensor([disease_idx])
    node_subset, _, _, _ = k_hop_subgraph(
        node_idx=disease_tensor, num_hops=num_hops,
        edge_index=edge_index.cpu(), num_nodes=num_nodes, relabel_nodes=False,
    )

    candidate_drugs, candidate_indices = [], []
    for node_idx in node_subset.tolist():
        entity = id2entity.get(node_idx, "")
        if entity.startswith("Compound::"):
            candidate_drugs.append(entity)
            candidate_indices.append(node_idx)
    return candidate_drugs, candidate_indices


def get_neighbors(node_idx, edge_index):
    mask_h = edge_index[0] == node_idx
    mask_t = edge_index[1] == node_idx
    neighbors = set()
    neighbors.update(edge_index[1][mask_h].tolist())
    neighbors.update(edge_index[0][mask_t].tolist())
    return neighbors


def compute_graph_structural_score(drug_entity, disease_entity, entity2id, id2entity, edge_index):
    drug_idx = entity2id.get(drug_entity)
    disease_idx = entity2id.get(disease_entity)
    
    if drug_idx is None or disease_idx is None:
        return {'path_count_1hop': 0, 'path_count_2hop': 0, 'shared_gene_score': 0.0, 'immune_gene_score': 0.0}
    
    drug_neighbors = get_neighbors(drug_idx, edge_index)
    dis_neighbors = get_neighbors(disease_idx, edge_index)
    
    path_count_1hop = 1 if disease_idx in drug_neighbors else 0
    shared_neighbors = drug_neighbors.intersection(dis_neighbors)
    path_count_2hop = len(shared_neighbors)
    
    # Gene overlap
    drug_genes = {id2entity.get(n, '') for n in drug_neighbors if id2entity.get(n, '').startswith('Gene::')}
    dis_genes = {id2entity.get(n, '') for n in dis_neighbors if id2entity.get(n, '').startswith('Gene::')}
    
    if drug_genes and dis_genes:
        shared_genes = drug_genes.intersection(dis_genes)
        shared_gene_score = len(shared_genes) / max(len(drug_genes.union(dis_genes)), 1)
    else:
        shared_gene_score = 0.0
    
    # Immune gene score
    drug_gene_symbols = {g.split('::')[-1] for g in drug_genes}
    immune_hits = drug_gene_symbols.intersection(IMMUNE_GENES)
    immune_gene_score = len(immune_hits) / max(len(drug_gene_symbols), 1) if drug_gene_symbols else 0.0
    
    return {
        'path_count_1hop': path_count_1hop,
        'path_count_2hop': path_count_2hop,
        'shared_gene_score': shared_gene_score,
        'immune_gene_score': immune_gene_score
    }


def get_known_drugs_per_disease(data_loader, entity2id):
    """Get drugs known to treat each test disease (from train data)."""
    known = {}
    train_file = Path(data_loader.processed_dir) / 'train_pairs.csv'
    if train_file.exists():
        df = pd.read_csv(train_file)
        for _, row in df.iterrows():
            h, t = str(row.get('head', '')), str(row.get('tail', ''))
            drug = h if 'Compound' in h else t
            disease = t if 'Disease' in t else h
            if disease not in known:
                known[disease] = set()
            known[disease].add(drug)
    return known


def predict_for_disease(model, data_loader, disease_entity, entity2id, id2entity, device, top_k=20):
    """Run v2 model predictions for a single disease."""
    disease_idx = entity2id.get(disease_entity)
    if disease_idx is None:
        print(f"  Disease {disease_entity} not found in graph!")
        return []
    
    # Find candidate drugs
    candidate_drugs, candidate_indices = find_candidate_drugs(
        disease_entity, entity2id, id2entity,
        data_loader.graph_data.edge_index, data_loader.num_nodes
    )
    print(f"  Found {len(candidate_drugs)} candidate drugs in 3-hop neighborhood")
    
    if not candidate_drugs:
        return []
    
    # Score each candidate
    node_features = data_loader.node_features.to(device)
    edge_index = data_loader.graph_data.edge_index.to(device)
    edge_type = data_loader.graph_data.edge_type.to(device)
    
    results = []
    with torch.no_grad():
        for drug_entity, drug_idx in tqdm(zip(candidate_drugs, candidate_indices), 
                                           total=len(candidate_drugs), desc="  Scoring"):
            drug_tensor = torch.tensor(drug_idx, device=device)
            disease_tensor = torch.tensor([disease_idx], device=device)
            
            try:
                scores, _ = model(node_features, edge_index, edge_type, drug_tensor, disease_tensor)
                neural_score = scores[0].item()
            except Exception as e:
                neural_score = 0.0
            
            # Graph structural features
            struct = compute_graph_structural_score(
                drug_entity, disease_entity, entity2id, id2entity, edge_index.cpu()
            )
            
            # Path diversity
            path_diversity = min(struct['path_count_2hop'] / 50.0, 1.0)
            
            # Hybrid score
            hybrid_score = (
                0.40 * neural_score +
                0.25 * path_diversity +
                0.20 * struct['immune_gene_score'] +
                0.15 * struct['shared_gene_score']
            )
            
            drug_name = get_drug_name_with_lookup(drug_entity)
            results.append({
                'drug_entity': drug_entity,
                'drug_name': drug_name,
                'neural_score': neural_score,
                'hybrid_score': hybrid_score,
                'path_diversity': path_diversity,
                'immune_gene_score': struct['immune_gene_score'],
                'shared_gene_score': struct['shared_gene_score'],
            })
    
    # Sort by hybrid score
    results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return results[:top_k]


def main():
    print("=" * 70)
    print("BioPathNet++ v2 — Drug Predictions (Retrained Model)")
    print("=" * 70)
    
    device = torch.device('cpu')  # CPU is fine for inference
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading graph data...")
    data_loader = GraphDataLoader(processed_dir='datasets/processed', batch_size=4)
    
    # Load entity mappings
    entity2id, id2entity = load_entity_mappings('datasets/processed')
    
    # Load v2 model
    model = load_v2_model('checkpoints/best_model_v2.pt', data_loader, device)
    
    # Get known drugs
    known_drugs = get_known_drugs_per_disease(data_loader, entity2id)
    
    # Output directory
    out_dir = Path('outputs/predictions_v2_retrained')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Predicting for {len(TEST_DISEASES)} diseases")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    for disease_entity in TEST_DISEASES:
        disease_name = get_disease_name(disease_entity)
        short_name = get_disease_short_name(disease_entity)
        print(f"\n{'='*60}")
        print(f"Disease: {disease_name}")
        print(f"{'='*60}")
        
        results = predict_for_disease(
            model, data_loader, disease_entity, entity2id, id2entity, device, top_k=20
        )
        
        if not results:
            print("  No candidates found!")
            continue
        
        # Mark known/novel
        disease_known = known_drugs.get(disease_entity, set())
        for r in results:
            r['status'] = 'Known' if r['drug_entity'] in disease_known else 'Novel'
        
        # Save CSV
        df = pd.DataFrame(results)
        safe_name = short_name.replace("'", "").replace(" ", "_").replace("ö", "o")
        df.to_csv(out_dir / f"{safe_name}_predictions.csv", index=False)
        
        # Print top 10
        print(f"\n  Top-10 predictions:")
        print(f"  {'Rank':<5} {'Drug':<25} {'Hybrid':<8} {'Neural':<8} {'Status':<8}")
        print(f"  {'-'*55}")
        for i, r in enumerate(results[:10]):
            name = r['drug_name'][:24]
            print(f"  {i+1:<5} {name:<25} {r['hybrid_score']:.4f}  {r['neural_score']:.4f}  {r['status']}")
        
        all_results[disease_name] = results
    
    # Save summary
    summary = []
    for disease_name, results in all_results.items():
        for r in results[:10]:
            summary.append({
                'disease': disease_name,
                'drug': r['drug_name'],
                'hybrid_score': r['hybrid_score'],
                'neural_score': r['neural_score'],
                'status': r['status']
            })
    
    pd.DataFrame(summary).to_csv(out_dir / 'all_predictions_summary.csv', index=False)
    print(f"\n\nAll results saved to {out_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()

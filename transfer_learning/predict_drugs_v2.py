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
import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm

from models.data_loader import GraphDataLoader
from models.disease_mappings import (
    get_disease_name, get_disease_short_name, get_drug_name,
    get_drug_name_with_lookup, get_drug_id, get_gene_symbol,
    load_entity_mappings, _load_drug_cache, _save_drug_cache
)
from transfer_learning.predict_drugs import (
    load_model, find_candidate_drugs, get_known_drugs_per_disease, find_drug_paths
)

TEST_DISEASES = [
    'Disease::MESH:D012859',  # Sjögren's
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

def get_neighbors(node_idx, edge_index):
    mask_h = edge_index[0] == node_idx
    mask_t = edge_index[1] == node_idx
    neighbors = set()
    neighbors.update(edge_index[1][mask_h].tolist())
    neighbors.update(edge_index[0][mask_t].tolist())
    return neighbors

def compute_graph_structural_score(drug_entity, disease_entity, entity2id, id2entity, edge_index, edge_type=None):
    drug_idx = entity2id.get(drug_entity)
    disease_idx = entity2id.get(disease_entity)
    
    if drug_idx is None or disease_idx is None:
        return {
            'path_count_1hop': 0, 'path_count_2hop': 0, 'path_count_3hop': 0,
            'shared_gene_score': 0.0, 'immune_gene_score': 0.0
        }
        
    drug_neighbors = get_neighbors(drug_idx, edge_index)
    dis_neighbors = get_neighbors(disease_idx, edge_index)
    
    path_count_1hop = 1 if disease_idx in drug_neighbors else 0
    
    shared_neighbors = drug_neighbors.intersection(dis_neighbors)
    path_count_2hop = len(shared_neighbors)
    
    path_count_3hop = 0
    for n1 in drug_neighbors:
        if n1 == disease_idx: continue
        n1_neighbors = get_neighbors(n1, edge_index)
        for n2 in n1_neighbors:
            if n2 == drug_idx or n2 == disease_idx: continue
            if n2 in dis_neighbors:
                path_count_3hop += 1
                
    def get_genes(neighbors):
        genes = set()
        for idx in neighbors:
            ent = id2entity.get(idx, "")
            if ent.startswith("Gene::"):
                genes.add(ent)
        return genes
        
    drug_genes = get_genes(drug_neighbors)
    dis_genes = get_genes(dis_neighbors)
    
    intersection = drug_genes.intersection(dis_genes)
    union = drug_genes.union(dis_genes)
    shared_gene_score = len(intersection) / len(union) if union else 0.0
    
    immune_genes_count = 0
    for g_ent in intersection:
        sym = get_gene_symbol(g_ent)
        if sym in IMMUNE_GENES:
            immune_genes_count += 1
            
    immune_gene_score = immune_genes_count / len(intersection) if intersection else 0.0
    
    return {
        'path_count_1hop': path_count_1hop,
        'path_count_2hop': path_count_2hop,
        'path_count_3hop': path_count_3hop,
        'shared_gene_score': shared_gene_score,
        'immune_gene_score': immune_gene_score
    }

def compute_hybrid_score(model_score, graph_features):
    path_diversity_score = math.log(1 + graph_features['path_count_2hop'] + graph_features['path_count_3hop'])
    immune_relevance_score = graph_features['immune_gene_score'] * 10
    gene_overlap_score = graph_features['shared_gene_score'] * 10
    
    hybrid_score = (model_score * 0.4) + (path_diversity_score * 0.25) + (immune_relevance_score * 0.2) + (gene_overlap_score * 0.15)
    return hybrid_score, path_diversity_score, immune_relevance_score, gene_overlap_score

def predict_drugs_v2_for_disease(
    model, data_loader, disease_entity, entity2id, id2entity,
    edge_index, device, top_k=50, num_hops=3, known_positive_drugs=None,
):
    if known_positive_drugs is None:
        known_positive_drugs = set()

    disease_name = get_disease_name(disease_entity)
    disease_idx = entity2id.get(disease_entity)

    if disease_idx is None:
        print(f"  ⚠ Disease '{disease_entity}' not found!")
        return pd.DataFrame()

    print(f"  Finding drugs within {num_hops} hops of {disease_name}...")
    candidates, candidate_indices = find_candidate_drugs(
        disease_entity, entity2id, id2entity,
        edge_index, len(entity2id), num_hops=num_hops,
    )
    print(f"  Found {len(candidates)} candidate drugs")

    if len(candidates) == 0:
        print(f"  ⚠ No drugs found within {num_hops} hops!")
        return pd.DataFrame()

    all_raw_scores = []
    all_hybrid_scores = []
    all_graph_features = []
    all_attentions = []
    disease_idx_tensor = torch.tensor([disease_idx], device=device)

    edge_index_cpu = edge_index.cpu()
    
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
                model_score = scores.cpu().item()
                if attn_weights is not None:
                    attn = attn_weights.cpu().numpy().tolist()
                else:
                    attn = [0.0] * 7
            except Exception as e:
                model_score = float('-inf')
                attn = [0.0] * 7
                
            all_raw_scores.append(model_score)
            all_attentions.append(attn)
            
            graph_features = compute_graph_structural_score(drug_entity, disease_entity, entity2id, id2entity, edge_index_cpu)
            hybrid_score, pd_score, ir_score, go_score = compute_hybrid_score(model_score, graph_features)
            
            all_graph_features.append({
                **graph_features,
                'path_diversity_score': pd_score,
                'immune_relevance_score': ir_score,
                'gene_overlap_score': go_score
            })
            all_hybrid_scores.append(hybrid_score)

    path_info = []
    for drug_entity in candidates:
        path_genes = find_drug_paths(
            drug_entity, disease_entity,
            entity2id, id2entity, edge_index_cpu,
        )
        path_info.append(path_genes)

    df = pd.DataFrame({
        'drug_entity': candidates,
        'model_score': all_raw_scores,
        'hybrid_score': all_hybrid_scores,
        'attention_weights': all_attentions,
        'path_genes': path_info,
    })
    
    for key in all_graph_features[0].keys():
        df[key] = [gf[key] for gf in all_graph_features]

    df = df.sort_values('hybrid_score', ascending=False).reset_index(drop=True)

    df['rank'] = range(1, len(df) + 1)
    df['drug_id'] = df['drug_entity'].apply(get_drug_id)
    df['drug_name'] = df['drug_entity'].apply(get_drug_name)
    df['disease_entity'] = disease_entity
    df['disease_name'] = disease_name
    df['known_drug'] = df['drug_entity'].isin(known_positive_drugs)
    df['path_genes_str'] = df['path_genes'].apply(lambda x: ', '.join(x) if x else 'indirect')

    cols = ['rank', 'drug_entity', 'drug_id', 'drug_name', 'hybrid_score', 'model_score',
            'known_drug', 'disease_entity', 'disease_name', 'path_genes_str', 
            'path_count_1hop', 'path_count_2hop', 'path_count_3hop',
            'shared_gene_score', 'immune_gene_score', 'path_diversity_score',
            'immune_relevance_score', 'gene_overlap_score', 'attention_weights']
    df = df[cols]

    return df.head(top_k)

def main():
    print("=" * 80)
    print("PHASE 3: ZERO-SHOT DRUG PREDICTION (V2 - HYBRID SCORING)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading graph data...")
    data_loader = GraphDataLoader(processed_dir='datasets/processed', batch_size=4)
    entity2id, id2entity = load_entity_mappings('datasets/processed')

    edge_index = data_loader.graph_data.edge_index

    print("\nLoading model...")
    model, config = load_model('checkpoints/best_model.pt', data_loader, device)

    known_drugs = get_known_drugs_per_disease(entity2id)

    output_dir = Path('outputs/predictions_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    _load_drug_cache()

    all_results = []

    print(f"\n{'=' * 80}")
    print(f"Predicting drugs for {len(TEST_DISEASES)} target diseases")
    print(f"Using Hybrid Scoring Approach")
    print(f"{'=' * 80}")

    for disease_entity in TEST_DISEASES:
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

        df = predict_drugs_v2_for_disease(
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

        print(f"  Looking up drug names...")
        for idx, row in df.iterrows():
            if row['drug_name'] == row['drug_id'] and row['drug_id'].startswith('DB'):
                name = get_drug_name_with_lookup(row['drug_entity'])
                if name and name != row['drug_id']:
                    df.at[idx, 'drug_name'] = name
                time.sleep(0.1)

        csv_path = output_dir / f"{short_name}_top50_hybrid.csv"
        df_save = df.drop(columns=['attention_weights'])
        df_save.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path}")

        attn_path = output_dir / f"{short_name}_attention.json"
        attn_data = []
        for _, row in df.head(20).iterrows():
            attn_data.append({
                'rank': int(row['rank']),
                'drug': row['drug_name'],
                'drug_id': row['drug_id'],
                'hybrid_score': float(row['hybrid_score']),
                'model_score': float(row['model_score']),
                'attention': row['attention_weights'],
                'path_genes': row['path_genes_str'],
            })
        with open(attn_path, 'w') as f:
            json.dump(attn_data, f, indent=2)

        print(f"\n  Top 20 Drug Candidates for {disease_name}:")
        print(f"  {'Rank':<5} {'Drug Name':<30} {'Hybrid':<10} {'Model':<10} {'Known?':<8} {'Path Genes'}")
        print(f"  {'─' * 95}")
        for _, row in df.head(20).iterrows():
            known_flag = "✓ YES" if row['known_drug'] else ""
            genes_str = row['path_genes_str'][:30] if row['path_genes_str'] else ""
            print(f"  {row['rank']:<5} {str(row['drug_name'])[:28]:<30} "
                  f"{row['hybrid_score']:<10.4f} {row['model_score']:<10.4f} {known_flag:<8} {genes_str}")

        for k in [10, 20, 50]:
            top_k_df = df.head(k)
            hits = top_k_df['known_drug'].sum()
            total_known = len(known)
            pct = (hits / total_known * 100) if total_known > 0 else 0
            print(f"  Hit@{k}: {hits}/{total_known} ({pct:.1f}%)")

        all_results.append(df)

    _save_drug_cache()

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        summary_path = output_dir / "all_diseases_summary_hybrid.csv"
        combined.drop(columns=['attention_weights']).to_csv(summary_path, index=False)
        print(f"\n✓ Combined summary saved: {summary_path}")

    print(f"\n{'=' * 80}")
    print("PREDICTION SUMMARY")
    print(f"{'=' * 80}")
    for disease_entity in TEST_DISEASES:
        disease_name = get_disease_name(disease_entity)
        known = known_drugs.get(disease_entity, set())
        csv_file = output_dir / f"{get_disease_short_name(disease_entity)}_top50_hybrid.csv"
        if csv_file.exists():
            ddf = pd.read_csv(csv_file)
            print(f"  {disease_name}: {len(ddf)} candidates scored, "
                  f"{len(known)} known drugs, "
                  f"top score: {ddf['hybrid_score'].max():.4f}")
        else:
            print(f"  {disease_name}: no predictions")

    print(f"\nAll predictions saved to: {output_dir}")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    main()

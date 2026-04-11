"""
Immune Pathway Analysis (v2 - Fixed)
======================================

Uses Entrez Gene IDs (matching KG format) to check if top drug predictions
involve known autoimmune immune mechanisms.

Usage:
    python validation/pathway_analysis.py

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
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from models.disease_mappings import (
    IMMUNE_PATHWAY_GENES, ALL_IMMUNE_GENE_IDS,
    get_disease_name, get_disease_short_name, get_drug_name, get_drug_id,
    get_gene_symbol, load_entity_mappings, TEST_DISEASE_ENTITIES,
    check_immune_pathway, ENTREZ_TO_SYMBOL,
)


def find_drug_gene_neighbors(drug_entity, edge_index, entity2id, id2entity):
    """
    Find all genes directly connected to a drug in the KG.
    Returns list of gene entity strings.
    """
    drug_idx = entity2id.get(drug_entity)
    if drug_idx is None:
        return []

    neighbors = set()

    # Drug as head → tail is neighbor
    head_mask = edge_index[0] == drug_idx
    tail_indices = edge_index[1][head_mask].cpu().numpy()
    for idx in tail_indices:
        entity = id2entity.get(int(idx), "")
        if entity.startswith("Gene::"):
            neighbors.add(entity)

    # Drug as tail → head is neighbor
    tail_mask = edge_index[1] == drug_idx
    head_indices = edge_index[0][tail_mask].cpu().numpy()
    for idx in head_indices:
        entity = id2entity.get(int(idx), "")
        if entity.startswith("Gene::"):
            neighbors.add(entity)

    return list(neighbors)


def analyze_drug_pathways(gene_neighbors):
    """
    For a drug's gene neighbors, check which immune pathways they match.
    Uses Entrez Gene IDs for comparison.
    """
    pathway_matches = {}
    immune_gene_count = 0

    for gene_entity in gene_neighbors:
        # Extract Entrez ID (e.g., 'Gene::3569' → '3569')
        entrez_id = gene_entity.split("::", 1)[1] if "::" in gene_entity else gene_entity

        matched_pathways = check_immune_pathway(gene_entity)
        if matched_pathways:
            immune_gene_count += 1
            symbol = get_gene_symbol(gene_entity)
            for pathway in matched_pathways:
                pathway_matches.setdefault(pathway, []).append(symbol)

    return pathway_matches, immune_gene_count


def main():
    print("=" * 80)
    print("IMMUNE PATHWAY ANALYSIS (v2 - Entrez IDs)")
    print("=" * 80)

    entity2id, id2entity = load_entity_mappings('datasets/processed')

    print("Loading graph...")
    graph_data = torch.load('datasets/processed/pyg_graph.pt', weights_only=False)
    edge_index = graph_data.edge_index

    pred_dir = Path('outputs/predictions')
    if not pred_dir.exists():
        print("⚠ Predictions not found! Run predict_drugs.py first.")
        return

    all_results = []

    for disease_entity in TEST_DISEASE_ENTITIES:
        disease_name = get_disease_name(disease_entity)
        short_name = get_disease_short_name(disease_entity)
        csv_path = pred_dir / f"{short_name}_top50.csv"

        if not csv_path.exists():
            print(f"  ⚠ Predictions not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        top_20 = df.head(20)

        print(f"\n{'─' * 60}")
        print(f"Analyzing: {disease_name} ({len(top_20)} predictions)")
        print(f"{'─' * 60}")

        disease_immune_count = 0

        for _, row in top_20.iterrows():
            drug_entity = row['drug_entity']
            drug_name = row.get('drug_name', get_drug_name(drug_entity))

            # Find gene neighbors
            gene_neighbors = find_drug_gene_neighbors(
                drug_entity, edge_index, entity2id, id2entity
            )

            # Analyze pathways
            pathway_matches, immune_count = analyze_drug_pathways(gene_neighbors)

            has_immune = len(pathway_matches) > 0
            if has_immune:
                disease_immune_count += 1

            result = {
                'disease_entity': disease_entity,
                'disease_name': disease_name,
                'drug_entity': drug_entity,
                'drug_id': row['drug_id'],
                'drug_name': drug_name,
                'rank': row['rank'],
                'score': row['score'],
                'num_gene_neighbors': len(gene_neighbors),
                'num_immune_genes': immune_count,
                'has_immune_pathway': has_immune,
                'matched_pathways': ', '.join(pathway_matches.keys()) if pathway_matches else 'None',
                'matched_genes': json.dumps({p: g for p, g in pathway_matches.items()}) if pathway_matches else '{}',
            }
            all_results.append(result)

            if has_immune:
                gene_list = []
                for genes in pathway_matches.values():
                    gene_list.extend(genes)
                pathways_str = ', '.join(pathway_matches.keys())
                print(f"  ✓ Rank {row['rank']}: {drug_name} → {pathways_str} "
                      f"(genes: {', '.join(gene_list[:5])})")
            else:
                print(f"  ✗ Rank {row['rank']}: {drug_name} → No immune match "
                      f"({len(gene_neighbors)} gene neighbors)")

        match_rate = (disease_immune_count / len(top_20) * 100) if len(top_20) > 0 else 0
        print(f"  → Match rate: {disease_immune_count}/{len(top_20)} ({match_rate:.1f}%)")

    results_df = pd.DataFrame(all_results)

    # Summary
    if len(results_df) > 0:
        total = len(results_df)
        immune_matches = results_df['has_immune_pathway'].sum()
        overall_rate = immune_matches / total * 100

        print(f"\n{'=' * 80}")
        print("PATHWAY ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(f"  Total predictions analyzed: {total}")
        print(f"  Immune pathway matches: {int(immune_matches)} ({overall_rate:.1f}%)")
        print(f"  (Target: ≥70%)")

        all_pathways = {}
        for _, row in results_df[results_df['has_immune_pathway']].iterrows():
            for p in row['matched_pathways'].split(', '):
                if p and p != 'None':
                    all_pathways[p] = all_pathways.get(p, 0) + 1

        if all_pathways:
            print(f"\n  Pathway frequency:")
            for pathway, count in sorted(all_pathways.items(), key=lambda x: -x[1]):
                print(f"    {pathway}: {count} predictions")

        summary = {
            'overall_match_rate': overall_rate,
            'total_analyzed': total,
            'immune_matches': int(immune_matches),
            'pathway_frequency': all_pathways,
            'per_disease': {},
        }
        for disease in results_df['disease_name'].unique():
            d_df = results_df[results_df['disease_name'] == disease]
            d_matches = d_df['has_immune_pathway'].sum()
            summary['per_disease'][disease] = {
                'analyzed': len(d_df),
                'matches': int(d_matches),
                'rate': d_matches / len(d_df) * 100 if len(d_df) > 0 else 0,
            }
    else:
        summary = {'overall_match_rate': 0, 'total_analyzed': 0}

    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'pathway_analysis.csv', index=False)
    with open(output_dir / 'pathway_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()

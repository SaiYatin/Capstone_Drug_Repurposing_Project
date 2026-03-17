"""
PyTorch Geometric Data Converter
=================================

Converts enriched DRKG to PyTorch Geometric format for GNN training.
NOW INCLUDES negative sampling for proper AUPRC evaluation.

Author: Yatin
Date: February 2026
"""

import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


class PyGDataConverter:
    """Convert to PyTorch Geometric format with proper negatives"""

    def __init__(self, processed_dir='datasets/processed'):
        self.processed_dir = Path(processed_dir)

        print("=" * 80)
        print("PYTORCH GEOMETRIC DATA CONVERTER")
        print("=" * 80)
        print()

    def load_all_data(self):
        """Load graph, embeddings, and mappings"""
        print("Loading data...")

        graph_file = self.processed_dir / 'enriched_drkg.tsv'
        self.graph = pd.read_csv(
            graph_file, sep='\t', header=None,
            names=['head', 'relation', 'tail']
        )
        print(f"  ✓ Graph: {len(self.graph):,} triples")

        with open(self.processed_dir / 'enriched_entity2id.json', 'r') as f:
            self.entity2id = json.load(f)
        print(f"  ✓ Entities: {len(self.entity2id):,}")

        with open(self.processed_dir / 'enriched_relation2id.json', 'r') as f:
            self.relation2id = json.load(f)
        print(f"  ✓ Relations: {len(self.relation2id):,}")

        self.embeddings = np.load(self.processed_dir / 'entity_embeddings.npy')
        print(f"  ✓ Embeddings: {self.embeddings.shape}")
        print()

    def create_edge_index(self):
        """Convert triples to edge index using fast vectorized mapping"""
        print("Creating edge index (vectorized)...")

        heads = self.graph['head'].map(self.entity2id)
        tails = self.graph['tail'].map(self.entity2id)
        rels  = self.graph['relation'].map(self.relation2id)

        # Drop rows where mapping failed
        mask = heads.notna() & tails.notna() & rels.notna()
        heads = heads[mask].astype(int).values
        tails = tails[mask].astype(int).values
        rels  = rels[mask].astype(int).values

        edge_index = torch.tensor(np.stack([heads, tails], axis=0), dtype=torch.long)
        edge_type  = torch.tensor(rels, dtype=torch.long)

        print(f"  ✓ Edge index: {edge_index.shape}")
        print(f"  ✓ Edge types: {edge_type.shape}")
        print()
        return edge_index, edge_type

    def create_train_test_split(self):
        """Create disease-based train/val/test splits"""
        print("Creating train/val/test splits...")

        drug_disease_mask = (
            (self.graph['head'].str.startswith('Compound::', na=False) &
             self.graph['tail'].str.startswith('Disease::', na=False)) |
            (self.graph['tail'].str.startswith('Compound::', na=False) &
             self.graph['head'].str.startswith('Disease::', na=False))
        )
        drug_disease_triples = self.graph[drug_disease_mask].copy()
        print(f"  Total drug-disease pairs: {len(drug_disease_triples):,}")

        training_diseases = [
            'Disease::MESH:D001172',   # RA
            'Disease::MESH:D008180',   # SLE
            'Disease::MESH:D009103',   # MS
        ]
        target_diseases = [
            'Disease::MESH:D003424',   # Crohn's
            'Disease::MESH:D012859',   # Sjogren's
            'Disease::MESH:D012595',   # Systemic Sclerosis
            'Disease::MESH:D008105',   # PBC
            'Disease::MESH:D009157',   # Myasthenia Gravis
            'Disease::MESH:D019693',   # Autoimmune Hepatitis
        ]

        def in_disease_list(row, disease_list):
            return any(d in str(row['head']) or d in str(row['tail'])
                       for d in disease_list)

        train_mask = drug_disease_triples.apply(
            lambda r: in_disease_list(r, training_diseases), axis=1)
        test_mask = drug_disease_triples.apply(
            lambda r: in_disease_list(r, target_diseases), axis=1)

        train_triples = drug_disease_triples[train_mask]
        test_triples  = drug_disease_triples[test_mask]

        if len(train_triples) > 1:
            train_data, val_data = train_test_split(
                train_triples, test_size=0.2, random_state=42)
        else:
            train_data = train_triples
            val_data   = pd.DataFrame(columns=train_triples.columns)

        print(f"\n  Split summary:")
        print(f"    Training set:   {len(train_data):,} positive pairs")
        print(f"    Validation set: {len(val_data):,}  positive pairs")
        print(f"    Test set:       {len(test_triples):,} positive pairs (zero-shot)")
        print()
        return train_data, val_data, test_triples

    def generate_negatives(self, positive_df, all_drugs, all_diseases,
                           positive_set, ratio=1, seed=42):
        """
        Generate random negative drug-disease pairs.
        Negatives = drug-disease combos NOT in the known positive set.

        Args:
            positive_df:   DataFrame of positive pairs
            all_drugs:     List of all drug entity strings
            all_diseases:  List of all disease entity strings
            positive_set:  Set of (drug, disease) tuples already known positive
            ratio:         Negatives per positive (1 = balanced)
            seed:          Random seed for reproducibility
        """
        random.seed(seed)
        needed   = len(positive_df) * ratio
        negatives = []
        attempts  = 0
        max_attempts = needed * 50  # safety cap

        while len(negatives) < needed and attempts < max_attempts:
            drug    = random.choice(all_drugs)
            disease = random.choice(all_diseases)
            if (drug, disease) not in positive_set and \
               (disease, drug) not in positive_set:
                negatives.append({
                    'head':     drug,
                    'relation': 'negative',
                    'tail':     disease,
                    'label':    0
                })
            attempts += 1

        if len(negatives) < needed:
            print(f"  ⚠ Only generated {len(negatives)}/{needed} negatives "
                  f"(positive set too dense)")

        return pd.DataFrame(negatives)

    def create_pyg_data(self, edge_index, edge_type):
        """Create PyTorch Geometric Data object"""
        print("Creating PyG Data object...")

        x    = torch.tensor(self.embeddings, dtype=torch.float)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(self.entity2id)
        )

        print(f"  ✓ Nodes: {data.num_nodes:,}")
        print(f"  ✓ Edges: {data.edge_index.size(1):,}")
        print(f"  ✓ Node feature dim: {data.x.size(1)}")
        print()
        return data

    def save_data(self, data, train_data, val_data, test_data):
        """Add labels + negatives, then save everything"""
        print("Generating negative samples...")

        # Collect all drugs and diseases from the full graph
        all_drugs = list(set(
            list(self.graph[
                self.graph['head'].str.startswith('Compound::', na=False)
            ]['head']) +
            list(self.graph[
                self.graph['tail'].str.startswith('Compound::', na=False)
            ]['tail'])
        ))
        all_diseases = list(set(
            list(self.graph[
                self.graph['head'].str.startswith('Disease::', na=False)
            ]['head']) +
            list(self.graph[
                self.graph['tail'].str.startswith('Disease::', na=False)
            ]['tail'])
        ))

        print(f"  Total drugs available:    {len(all_drugs):,}")
        print(f"  Total diseases available: {len(all_diseases):,}")

        # Build global positive set to avoid false negatives
        all_positives = pd.concat([train_data, val_data, test_data])
        positive_set  = set()
        for _, row in all_positives.iterrows():
            h, t = str(row['head']), str(row['tail'])
            if h.startswith('Compound::'):
                positive_set.add((h, t))
            else:
                positive_set.add((t, h))

        print(f"  Total known positives:    {len(positive_set):,}")

        # Add label=1 to positives
        train_data = train_data.copy(); train_data['label'] = 1
        val_data   = val_data.copy();   val_data['label']   = 1
        test_data  = test_data.copy();  test_data['label']  = 1

        # Generate negatives for each split
        print("\n  Generating train negatives...")
        train_neg = self.generate_negatives(
            train_data, all_drugs, all_diseases, positive_set, ratio=1, seed=42)

        print("  Generating val negatives...")
        val_neg = self.generate_negatives(
            val_data, all_drugs, all_diseases, positive_set, ratio=1, seed=43)

        print("  Generating test negatives...")
        test_neg = self.generate_negatives(
            test_data, all_drugs, all_diseases, positive_set, ratio=1, seed=44)

        # Combine positives + negatives and shuffle
        train_final = pd.concat([train_data, train_neg])\
            .sample(frac=1, random_state=42).reset_index(drop=True)
        val_final   = pd.concat([val_data,   val_neg])\
            .sample(frac=1, random_state=42).reset_index(drop=True)
        test_final  = pd.concat([test_data,  test_neg])\
            .sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n  Final split sizes (pos + neg):")
        print(f"    Train: {len(train_final):,}  "
              f"({train_final['label'].sum():,} pos / "
              f"{(train_final['label']==0).sum():,} neg)")
        print(f"    Val:   {len(val_final):,}  "
              f"({val_final['label'].sum():,} pos / "
              f"{(val_final['label']==0).sum():,} neg)")
        print(f"    Test:  {len(test_final):,}  "
              f"({test_final['label'].sum():,} pos / "
              f"{(test_final['label']==0).sum():,} neg)")

        # Save PyG graph
        graph_file = self.processed_dir / 'pyg_graph.pt'
        torch.save(data, graph_file)
        print(f"\n✓ Saved graph: {graph_file}")

        # Save splits
        train_final.to_csv(self.processed_dir / 'train_pairs.csv', index=False)
        val_final.to_csv(self.processed_dir   / 'val_pairs.csv',   index=False)
        test_final.to_csv(self.processed_dir  / 'test_pairs.csv',  index=False)
        print(f"✓ Saved train/val/test CSVs with labels")

        # Save metadata
        metadata = {
            'num_nodes':        data.num_nodes,
            'num_edges':        data.edge_index.size(1),
            'num_edge_types':   data.edge_type.max().item() + 1,
            'node_feature_dim': data.x.size(1),
            'train_pairs':      len(train_final),
            'val_pairs':        len(val_final),
            'test_pairs':       len(test_final),
            'negative_ratio':   1,
            'creation_date':    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.processed_dir / 'pyg_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata")
        print()

    def run(self):
        self.load_all_data()
        edge_index, edge_type = self.create_edge_index()
        train_data, val_data, test_data = self.create_train_test_split()
        data = self.create_pyg_data(edge_index, edge_type)
        self.save_data(data, train_data, val_data, test_data)

        print("=" * 80)
        print("PYG DATA CONVERSION COMPLETE")
        print("=" * 80)
        print("\n✓✓✓ READY FOR TRAINING WITH PROPER NEGATIVES ✓✓✓")
        print("=" * 80)


def main():
    converter = PyGDataConverter()
    converter.run()


if __name__ == '__main__':
    main()

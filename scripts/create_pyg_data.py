"""
PyTorch Geometric Data Converter
=================================

Converts enriched DRKG to PyTorch Geometric format for GNN training:
1. Loads enriched graph and embeddings
2. Creates edge index and edge types
3. Splits into train/validation/test sets
4. Saves PyG Data objects

Author: Yatin
Date: February 9, 2026
"""

import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

class PyGDataConverter:
    """Convert to PyTorch Geometric format"""
    
    def __init__(self, processed_dir='datasets/processed'):
        self.processed_dir = Path(processed_dir)
        
        print("="*80)
        print("PYTORCH GEOMETRIC DATA CONVERTER")
        print("="*80)
        print()
    
    def load_all_data(self):
        """Load graph, embeddings, and mappings"""
        print("Loading data...")
        
        # 1. Load enriched graph
        graph_file = self.processed_dir / 'enriched_drkg.tsv'
        self.graph = pd.read_csv(
            graph_file,
            sep='\t',
            header=None,
            names=['head', 'relation', 'tail']
        )
        print(f"  ✓ Graph: {len(self.graph):,} triples")
        
        # 2. Load entity mappings
        with open(self.processed_dir / 'enriched_entity2id.json', 'r') as f:
            self.entity2id = json.load(f)
        print(f"  ✓ Entities: {len(self.entity2id):,}")
        
        # 3. Load relation mappings
        with open(self.processed_dir / 'enriched_relation2id.json', 'r') as f:
            self.relation2id = json.load(f)
        print(f"  ✓ Relations: {len(self.relation2id):,}")
        
        # 4. Load embeddings
        self.embeddings = np.load(self.processed_dir / 'entity_embeddings.npy')
        print(f"  ✓ Embeddings: {self.embeddings.shape}")
        
        print()
    
    def create_edge_index(self):
        """Convert triples to edge index format"""
        print("Creating edge index...")
        
        edge_list = []
        edge_types = []
        
        print(f"  Processing {len(self.graph):,} triples...")
        
        for _, row in self.graph.iterrows():
            head_id = self.entity2id[row['head']]
            tail_id = self.entity2id[row['tail']]
            rel_id = self.relation2id[row['relation']]
            
            edge_list.append([head_id, tail_id])
            edge_types.append(rel_id)
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # [2, num_edges]
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        print(f"  ✓ Edge index: {edge_index.shape}")
        print(f"  ✓ Edge types: {edge_type.shape}")
        print()
        
        return edge_index, edge_type
    
    def create_train_test_split(self):
        """Create train/val/test splits for drug-disease pairs"""
        print("Creating train/val/test splits...")
        
        # Extract drug-disease pairs
        drug_disease_mask = (
            (self.graph['head'].str.startswith('Compound::', na=False) & 
             self.graph['tail'].str.startswith('Disease::', na=False)) |
            (self.graph['tail'].str.startswith('Compound::', na=False) & 
             self.graph['head'].str.startswith('Disease::', na=False))
        )
        
        drug_disease_triples = self.graph[drug_disease_mask].copy()
        
        print(f"  Total drug-disease pairs: {len(drug_disease_triples):,}")
        
        # Identify training vs target diseases
        training_diseases = [
            'Disease::MESH:D001172',  # RA
            'Disease::MESH:D008180',  # SLE
            'Disease::MESH:D009103',  # MS
        ]
        
        target_diseases = [
            'Disease::MESH:D003424',  # Crohn's
            'Disease::MESH:D012859',  # Sjögren's
            'Disease::MESH:D012595',  # Systemic Sclerosis
            'Disease::MESH:D008105',  # PBC
            'Disease::MESH:D009157',  # Myasthenia Gravis
            'Disease::MESH:D019693',  # Autoimmune Hepatitis
        ]
        
        # Split based on disease type
        train_mask = drug_disease_triples.apply(
            lambda row: any(d in row['head'] or d in row['tail'] for d in training_diseases),
            axis=1
        )
        
        test_mask = drug_disease_triples.apply(
            lambda row: any(d in row['head'] or d in row['tail'] for d in target_diseases),
            axis=1
        )
        
        train_triples = drug_disease_triples[train_mask]
        test_triples = drug_disease_triples[test_mask]
        
        # Further split train into train/val (80/20)
        if len(train_triples) > 0:
            train_data, val_data = train_test_split(
                train_triples,
                test_size=0.2,
                random_state=42
            )
        else:
            train_data = train_triples
            val_data = pd.DataFrame(columns=train_triples.columns)
        
        print(f"\n  Split summary:")
        print(f"    Training set: {len(train_data):,} pairs (data-rich diseases)")
        print(f"    Validation set: {len(val_data):,} pairs (held-out from training)")
        print(f"    Test set: {len(test_triples):,} pairs (zero-shot transfer)")
        
        print()
        
        return train_data, val_data, test_triples
    
    def create_pyg_data(self, edge_index, edge_type):
        """Create PyTorch Geometric Data object"""
        print("Creating PyG Data object...")
        
        # Node features (embeddings)
        x = torch.tensor(self.embeddings, dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(self.entity2id)
        )
        
        print(f"  ✓ PyG Data created:")
        print(f"    Nodes: {data.num_nodes:,}")
        print(f"    Edges: {data.edge_index.size(1):,}")
        print(f"    Node features: {data.x.size()}")
        print(f"    Edge types: {edge_type.max().item() + 1}")
        
        print()
        
        return data
    
    def save_data(self, data, train_data, val_data, test_data):
        """Save PyG data and splits"""
        print("Saving PyG data...")
        
        # Save main graph
        graph_file = self.processed_dir / 'pyg_graph.pt'
        torch.save(data, graph_file)
        print(f"✓ Saved graph: {graph_file}")
        print(f"  Size: {graph_file.stat().st_size / (1024**2):.1f} MB")
        
        # Save splits
        train_data.to_csv(self.processed_dir / 'train_pairs.csv', index=False)
        val_data.to_csv(self.processed_dir / 'val_pairs.csv', index=False)
        test_data.to_csv(self.processed_dir / 'test_pairs.csv', index=False)
        
        print(f"✓ Saved train pairs: {len(train_data):,}")
        print(f"✓ Saved val pairs: {len(val_data):,}")
        print(f"✓ Saved test pairs: {len(test_data):,}")
        
        # Save metadata
        metadata = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1),
            'num_edge_types': data.edge_type.max().item() + 1,
            'node_feature_dim': data.x.size(1),
            'train_pairs': len(train_data),
            'val_pairs': len(val_data),
            'test_pairs': len(test_data),
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.processed_dir / 'pyg_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata")
        print()
    
    def run(self):
        """Run complete conversion"""
        # Load data
        self.load_all_data()
        
        # Create edge index
        edge_index, edge_type = self.create_edge_index()
        
        # Create train/val/test splits
        train_data, val_data, test_data = self.create_train_test_split()
        
        # Create PyG Data object
        data = self.create_pyg_data(edge_index, edge_type)
        
        # Save
        self.save_data(data, train_data, val_data, test_data)
        
        # Final summary
        print("="*80)
        print("PYG DATA CONVERSION COMPLETE")
        print("="*80)
        print(f"\nGraph:")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.edge_index.size(1):,}")
        print(f"  Node features: 768-dim (BioBERT)")
        print(f"\nSplits:")
        print(f"  Training: {len(train_data):,} drug-disease pairs")
        print(f"  Validation: {len(val_data):,} pairs")
        print(f"  Test (zero-shot): {len(test_data):,} pairs")
        print(f"\nOutput: {self.processed_dir / 'pyg_graph.pt'}")
        print("\n✓✓✓ PHASE 1 COMPLETE - READY FOR MODEL TRAINING! ✓✓✓")
        print("="*80)

def main():
    converter = PyGDataConverter()
    converter.run()

if __name__ == '__main__':
    main()

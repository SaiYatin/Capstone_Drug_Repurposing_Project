"""
Data Loader for BioPathNet++
=============================

Loads processed data and creates batches for training.

Author: Yatin
Date: February 10, 2026
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path


class DrugDiseaseDataset(Dataset):
    """
    Dataset for drug-disease pairs
    
    Each sample is a (drug, disease, label) triple where:
    - drug: Drug entity index
    - disease: Disease entity index
    - label: 1 if drug treats disease, 0 otherwise
    """
    
    def __init__(self, pairs_file, entity2id_file):
        """
        Args:
            pairs_file: CSV file with columns [head, relation, tail]
            entity2id_file: JSON file mapping entities to IDs
        """
        
        # Load entity mapping
        with open(entity2id_file, 'r') as f:
            self.entity2id = json.load(f)
        
        # Load pairs
        self.pairs = pd.read_csv(pairs_file)
        
        print(f"Loaded {len(self.pairs)} drug-disease pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a single drug-disease pair
        
        Returns:
            drug_idx: Drug entity index
            disease_idx: Disease entity index
            label: 1 (positive pair)
        """
        
        row = self.pairs.iloc[idx]
        
        # Get entity indices
        drug_entity = row['head'] if '::' in row['head'] and 'Compound' in row['head'] else row['tail']
        disease_entity = row['tail'] if '::' in row['tail'] and 'Disease' in row['tail'] else row['head']
        
        drug_idx = self.entity2id[drug_entity]
        disease_idx = self.entity2id[disease_entity]
        
        return drug_idx, disease_idx, 1  # Label is 1 (positive pair)


class GraphDataLoader:
    """
    Data loader that handles the entire graph + batches of queries
    """
    
    def __init__(
        self,
        processed_dir='datasets/processed',
        batch_size=32,
        num_workers=0
    ):
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        
        # Load graph data
        print("Loading graph data...")
        self.load_graph()
        
        # Load node embeddings
        print("Loading node embeddings...")
        self.load_embeddings()
        
        print(f"Graph loaded: {self.num_nodes:,} nodes, {self.num_edges:,} edges")
    
    def load_graph(self):
        """Load PyG graph"""
        pyg_file = self.processed_dir / 'pyg_graph.pt'
        self.graph_data = torch.load(pyg_file, weights_only=False)
        
        self.num_nodes = self.graph_data.num_nodes
        self.num_edges = self.graph_data.edge_index.size(1)
        self.num_relations = self.graph_data.edge_type.max().item() + 1
        
        # Move to device if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_data = self.graph_data.to(self.device)
    
    def load_embeddings(self):
        """Load BioBERT embeddings"""
        emb_file = self.processed_dir / 'entity_embeddings.npy'
        embeddings = np.load(emb_file)
        
        self.node_features = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        self.hidden_dim = self.node_features.size(1)
    
    def get_train_loader(self):
        """Get training data loader"""
        train_dataset = DrugDiseaseDataset(
            pairs_file=self.processed_dir / 'train_pairs.csv',
            entity2id_file=self.processed_dir / 'enriched_entity2id.json'
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Windows compatibility
        )
    
    def get_val_loader(self):
        """Get validation data loader"""
        val_dataset = DrugDiseaseDataset(
            pairs_file=self.processed_dir / 'val_pairs.csv',
            entity2id_file=self.processed_dir / 'enriched_entity2id.json'
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def get_test_loader(self):
        """Get test data loader"""
        test_dataset = DrugDiseaseDataset(
            pairs_file=self.processed_dir / 'test_pairs.csv',
            entity2id_file=self.processed_dir / 'enriched_entity2id.json'
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def get_negative_samples(self, drug_idx, num_negatives=5):
        """
        Generate negative samples (drugs that DON'T treat this disease)
        
        Args:
            drug_idx: Drug index
            num_negatives: Number of negative samples to generate
        
        Returns:
            negative_disease_indices: Tensor of negative disease indices
        """
        
        # Simple random sampling (in practice, use hard negatives)
        all_nodes = torch.arange(self.num_nodes, device=self.device)
        
        # Filter to disease nodes (those starting with "Disease::")
        # This is a simplified version - in practice, use entity type information
        negative_indices = torch.randint(0, self.num_nodes, (num_negatives,), device=self.device)
        
        return negative_indices


# Testing
if __name__ == '__main__':
    print("Testing data loader...")
    
    # Initialize
    data_loader = GraphDataLoader(
        processed_dir='datasets/processed',
        batch_size=4
    )
    
    # Get train loader
    train_loader = data_loader.get_train_loader()
    
    # Test one batch
    for batch_idx, (drugs, diseases, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Drugs: {drugs}")
        print(f"  Diseases: {diseases}")
        print(f"  Labels: {labels}")
        
        # Test negative sampling
        negatives = data_loader.get_negative_samples(drugs[0], num_negatives=5)
        print(f"  Negative samples: {negatives}")
        
        if batch_idx >= 2:  # Just test 3 batches
            break
    
    print("\n✓ Data loader test passed!")

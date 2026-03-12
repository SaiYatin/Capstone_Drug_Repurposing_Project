"""
BioPathNet++ Model
==================

Complete model for autoimmune drug repurposing with transfer learning.

Architecture:
1. NBFNet encoder - Path-based reasoning
2. Metapath extraction - 7 predefined biomedical paths
3. Attention mechanism - Weight path importance
4. BRG integration - Regulatory context
5. Prediction head - Drug-disease scoring

Author: Yatin
Date: February 10, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nbfnet import NBFNet


class MetapathExtractor(nn.Module):
    """
    Extracts and encodes predefined metapaths
    
    Metapaths for drug repurposing:
    1. Drug → Gene → Disease
    2. Drug → Gene → Gene → Disease
    3. Drug → Pathway → Disease
    4. Drug → Protein → Disease
    5. Drug → Phenotype → Disease
    6. Drug → Disease → Disease
    7. Drug → Gene → Pathway → Disease
    """
    
    def __init__(self, hidden_dim, num_metapaths=7):
        super(MetapathExtractor, self).__init__()
        
        self.num_metapaths = num_metapaths
        self.hidden_dim = hidden_dim
        
        # Metapath encoders (GRU to encode variable-length paths)
        self.path_encoders = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_metapaths)
        ])
        
        # Metapath type embeddings
        self.metapath_embeddings = nn.Embedding(num_metapaths, hidden_dim)
    
    def forward(self, node_features, paths_per_metapath):
        """
        Encode metapaths
        
        Args:
            node_features: All node representations (num_nodes, hidden_dim)
            paths_per_metapath: List of paths for each metapath type
                                Each element: tensor of shape (num_paths, path_length)
        
        Returns:
            Metapath representations (num_metapaths, hidden_dim)
        """
        
        metapath_reprs = []
        
        for metapath_idx, paths in enumerate(paths_per_metapath):
            if len(paths) == 0:
                # No paths of this type found
                metapath_reprs.append(torch.zeros(self.hidden_dim, device=node_features.device))
                continue
            
            # Get node features along paths
            path_features = []
            for path in paths:
                # path: (path_length,) tensor of node indices
                path_node_features = node_features[path]  # (path_length, hidden_dim)
                path_features.append(path_node_features)
            
            # Stack: (num_paths, path_length, hidden_dim)
            path_features = torch.stack(path_features)
            
            # Encode with GRU
            _, h_n = self.path_encoders[metapath_idx](path_features)
            # h_n: (1, num_paths, hidden_dim)
            
            # Average over paths
            metapath_repr = h_n.squeeze(0).mean(dim=0)  # (hidden_dim,)
            
            # Add metapath type embedding
            metapath_type_emb = self.metapath_embeddings(
                torch.tensor([metapath_idx], device=node_features.device)
            )
            metapath_repr = metapath_repr + metapath_type_emb.squeeze(0)
            
            metapath_reprs.append(metapath_repr)
        
        # Stack: (num_metapaths, hidden_dim)
        metapath_reprs = torch.stack(metapath_reprs)
        
        return metapath_reprs


class MetapathAttention(nn.Module):
    """
    Attention over metapaths
    
    Learns which metapaths are important for each drug-disease pair.
    """
    
    def __init__(self, hidden_dim, num_heads=4):
        super(MetapathAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, metapath_reprs, query_repr):
        """
        Compute attention-weighted metapath representation
        
        Args:
            metapath_reprs: Metapath representations (num_metapaths, hidden_dim)
            query_repr: Query representation (hidden_dim,) - drug embedding
        
        Returns:
            Weighted representation (hidden_dim,)
            Attention weights (num_metapaths,)
        """
        
        # Add batch and sequence dimensions
        metapath_reprs = metapath_reprs.unsqueeze(0)  # (1, num_metapaths, hidden_dim)
        query_repr = query_repr.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        
        # Attention: query attends to metapaths
        attn_output, attn_weights = self.attention(
            query_repr,  # Query
            metapath_reprs,  # Key
            metapath_reprs   # Value
        )
        # attn_output: (1, 1, hidden_dim)
        # attn_weights: (1, 1, num_metapaths)
        
        attn_output = attn_output.squeeze(0).squeeze(0)  # (hidden_dim,)
        attn_weights = attn_weights.squeeze(0).squeeze(0)  # (num_metapaths,)
        
        # Residual + LayerNorm
        output = self.layer_norm(attn_output + query_repr.squeeze(0).squeeze(0))
        
        return output, attn_weights


class BRGEncoder(nn.Module):
    """
    Background Regulatory Graph Encoder
    
    Encodes immune-specific regulatory context using the BRG.
    """
    
    def __init__(self, hidden_dim, num_layers=2):
        super(BRGEncoder, self).__init__()
        
        # GRU to encode regulatory cascades
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, regulatory_path_features):
        """
        Encode regulatory paths from BRG
        
        Args:
            regulatory_path_features: Features along regulatory paths
                                     (num_paths, path_length, hidden_dim)
        
        Returns:
            Regulatory context (hidden_dim,)
        """
        
        if len(regulatory_path_features) == 0:
            # No regulatory paths found
            return torch.zeros(self.gru.hidden_size, device=regulatory_path_features.device)
        
        # Encode with GRU
        _, h_n = self.gru(regulatory_path_features)
        # h_n: (num_layers, num_paths, hidden_dim)
        
        # Take last layer, average over paths
        regulatory_context = h_n[-1].mean(dim=0)  # (hidden_dim,)
        
        # Layer norm
        regulatory_context = self.layer_norm(regulatory_context)
        
        return regulatory_context


class BioPathNetPlusPlus(nn.Module):
    """
    Complete BioPathNet++ Model
    
    Combines:
    - NBFNet for path-based reasoning
    - Metapath extraction and attention
    - BRG integration for regulatory context
    - Transfer learning capability
    """
    
    def __init__(
        self,
        num_nodes,
        num_relations,
        hidden_dim=256,
        num_nbf_layers=3,
        num_metapaths=7,
        num_attention_heads=4,
        dropout=0.1
    ):
        super(BioPathNetPlusPlus, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # 1. NBFNet Encoder (path-based GNN)
        self.nbfnet = NBFNet(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_nbf_layers,
            dropout=dropout
        )
        
        # 2. Metapath System
        self.metapath_extractor = MetapathExtractor(
            hidden_dim=hidden_dim,
            num_metapaths=num_metapaths
        )
        
        self.metapath_attention = MetapathAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads
        )
        
        # 3. BRG Encoder
        self.brg_encoder = BRGEncoder(
            hidden_dim=hidden_dim,
            num_layers=2
        )
        
        # 4. Fusion Layer (combine NBFNet + Metapaths + BRG)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 5. Prediction Head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        node_features,
        edge_index,
        edge_type,
        query_drug,
        candidate_diseases,
        metapaths=None,
        brg_paths=None
    ):
        """
        Forward pass
        
        Args:
            node_features: Node feature matrix (num_nodes, hidden_dim)
            edge_index: Graph structure (2, num_edges)
            edge_type: Edge types (num_edges,)
            query_drug: Query drug node index (int or tensor)
            candidate_diseases: Candidate disease indices (num_candidates,)
            metapaths: Optional precomputed metapaths
            brg_paths: Optional precomputed BRG paths
        
        Returns:
            scores: Prediction scores (num_candidates,)
            attention_weights: Metapath attention weights (num_metapaths,)
        """
        
        # 1. NBFNet encoding (path-based reasoning)
        nbfnet_features = self.nbfnet(
            node_features,
            edge_index,
            edge_type,
            query_drug
        )  # (num_nodes, hidden_dim)
        
        # Get drug and disease representations from NBFNet
        drug_repr = nbfnet_features[query_drug]  # (hidden_dim,)
        disease_reprs = nbfnet_features[candidate_diseases]  # (num_candidates, hidden_dim)
        
        # 2. Metapath extraction and attention
        if metapaths is None:
            # Use simple placeholder (in practice, extract from graph)
            metapath_reprs = self.extract_simple_metapaths(
                nbfnet_features,
                query_drug,
                candidate_diseases
            )
        else:
            metapath_reprs = self.metapath_extractor(nbfnet_features, metapaths)
        
        # Attention over metapaths
        weighted_metapath_repr, attn_weights = self.metapath_attention(
            metapath_reprs,
            drug_repr
        )  # (hidden_dim,), (num_metapaths,)
        
        # 3. BRG context
        if brg_paths is None:
            # Placeholder
            brg_context = torch.zeros_like(drug_repr)
        else:
            brg_context = self.brg_encoder(brg_paths)
        
        # 4. Fusion (combine all information sources)
        combined_drug_repr = torch.cat([
            drug_repr,
            weighted_metapath_repr,
            brg_context
        ], dim=0)  # (hidden_dim * 3,)
        
        fused_drug_repr = self.fusion(combined_drug_repr)  # (hidden_dim,)
        
        # 5. Prediction for each candidate disease
        scores = []
        for disease_repr in disease_reprs:
            # Concatenate drug and disease representations
            pair_repr = torch.cat([fused_drug_repr, disease_repr], dim=0)  # (hidden_dim * 2,)
            
            # Predict score
            score = self.predictor(pair_repr)  # (1,)
            scores.append(score)
        
        scores = torch.cat(scores)  # (num_candidates,)
        
        return scores, attn_weights
    
    def extract_simple_metapaths(self, node_features, drug_idx, disease_indices):
        """
        Simple metapath extraction (placeholder)
        
        In practice, this should extract actual paths from the graph.
        For now, we use a simplified version based on node features.
        """
        
        # Just use drug and disease features as simple "metapaths"
        drug_feat = node_features[drug_idx]
        disease_feats = node_features[disease_indices]
        
        # Create 7 "metapath" representations by linear combinations
        metapath_reprs = []
        for i in range(7):
            # Different weighted combinations for different metapaths
            weight = (i + 1) / 7.0
            metapath_repr = drug_feat * weight + disease_feats.mean(dim=0) * (1 - weight)
            metapath_reprs.append(metapath_repr)
        
        metapath_reprs = torch.stack(metapath_reprs)  # (7, hidden_dim)
        
        return metapath_reprs
    
    def predict_all_diseases(self, node_features, edge_index, edge_type, query_drug):
        """
        Predict scores for ALL diseases (for ranking)
        
        Args:
            node_features, edge_index, edge_type: Graph data
            query_drug: Query drug index
        
        Returns:
            scores: Scores for all nodes (num_nodes,)
            disease_mask: Boolean mask indicating disease nodes
        """
        
        # Get all disease nodes (assuming they have a specific prefix or type)
        # This is a placeholder - in practice, filter by node type
        all_diseases = torch.arange(self.num_nodes, device=node_features.device)
        
        scores, attn_weights = self.forward(
            node_features,
            edge_index,
            edge_type,
            query_drug,
            all_diseases
        )
        
        return scores, all_diseases, attn_weights


# Testing
if __name__ == '__main__':
    print("Testing BioPathNet++ implementation...")
    
    # Toy parameters
    num_nodes = 1000
    num_relations = 100
    hidden_dim = 128
    num_edges = 5000
    
    # Random graph
    node_features = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    # Query
    query_drug = torch.tensor(0)
    candidate_diseases = torch.tensor([10, 20, 30, 40, 50])
    
    # Model
    model = BioPathNetPlusPlus(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        num_nbf_layers=3,
        num_metapaths=7,
        num_attention_heads=4,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    scores, attn_weights = model(
        node_features,
        edge_index,
        edge_type,
        query_drug,
        candidate_diseases
    )
    
    print(f"Scores shape: {scores.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Sample scores: {scores}")
    print(f"Sample attention weights: {attn_weights}")
    print("\n✓ BioPathNet++ test passed!")

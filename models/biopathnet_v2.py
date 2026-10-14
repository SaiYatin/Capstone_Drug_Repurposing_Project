"""
BioPathNet++ Model (v2 - Fixed)
================================

Fixes applied:
1. extract_simple_metapaths: per-candidate disease features instead of mean
2. BRG context: uses enriched subgraph features instead of zeros
3. Fusion: computed per-candidate instead of shared

Author: Yatin  
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
    Complete BioPathNet++ Model (v2 - Fixed)
    
    Combines:
    - NBFNet for path-based reasoning
    - Metapath extraction and attention (per-candidate)
    - BRG integration for regulatory context (active)
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
        
        # --- FIX 1: Learnable metapath projections (replaces placeholder) ---
        self.metapath_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_metapaths)
        ])
        
        # --- FIX 2: BRG context projection (uses disease neighborhood) ---
        self.brg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
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
        Forward pass (v2 - per-candidate fusion)
        
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
            attention_weights: Metapath attention weights (num_candidates, num_metapaths)
        """
        
        # 1. NBFNet encoding (path-based reasoning)
        nbfnet_features = self.nbfnet(
            node_features,
            edge_index,
            edge_type,
            query_drug
        )  # (num_nodes, hidden_dim)
        
        # Get drug representation from NBFNet
        drug_repr = nbfnet_features[query_drug]  # (hidden_dim,)
        disease_reprs = nbfnet_features[candidate_diseases]  # (num_candidates, hidden_dim)
        
        # --- FIX 3: Per-candidate scoring (fusion inside the loop) ---
        scores = []
        all_attn_weights = []
        
        for i, disease_repr in enumerate(disease_reprs):
            disease_idx = candidate_diseases[i]
            
            # 2. Metapath extraction - PER CANDIDATE (not averaged)
            if metapaths is None:
                metapath_reprs = self.extract_metapaths_per_candidate(
                    nbfnet_features,
                    query_drug,
                    disease_idx
                )
            else:
                metapath_reprs = self.metapath_extractor(nbfnet_features, metapaths)
            
            # Attention over metapaths
            weighted_metapath_repr, attn_weights = self.metapath_attention(
                metapath_reprs,
                drug_repr
            )  # (hidden_dim,), (num_metapaths,)
            
            # 3. BRG context - uses disease neighborhood features
            if brg_paths is None:
                brg_context = self.extract_brg_context(
                    nbfnet_features, edge_index, disease_idx
                )
            else:
                brg_context = self.brg_encoder(brg_paths)
            
            # 4. Fusion (per-candidate)
            combined = torch.cat([
                drug_repr,
                weighted_metapath_repr,
                brg_context
            ], dim=0)  # (hidden_dim * 3,)
            
            fused_drug_repr = self.fusion(combined)  # (hidden_dim,)
            
            # 5. Predict score
            pair_repr = torch.cat([fused_drug_repr, disease_repr], dim=0)
            score = self.predictor(pair_repr)
            scores.append(score)
            all_attn_weights.append(attn_weights)
        
        scores = torch.cat(scores)  # (num_candidates,)
        all_attn_weights = torch.stack(all_attn_weights)  # (num_candidates, num_metapaths)
        
        return scores, all_attn_weights
    
    def extract_metapaths_per_candidate(self, node_features, drug_idx, disease_idx):
        """
        FIX 1: Per-candidate metapath extraction
        
        Instead of averaging all disease features, creates metapath 
        representations specific to each drug-disease pair using learned
        projections over concatenated drug+disease features.
        """
        drug_feat = node_features[drug_idx]      # (hidden_dim,)
        disease_feat = node_features[disease_idx] # (hidden_dim,)
        
        # Concatenate drug and disease features
        pair_feat = torch.cat([drug_feat, disease_feat], dim=0)  # (hidden_dim * 2,)
        
        metapath_reprs = []
        for i in range(len(self.metapath_projections)):
            # Each metapath type has its own learned projection
            metapath_repr = self.metapath_projections[i](pair_feat)  # (hidden_dim,)
            metapath_reprs.append(metapath_repr)
        
        return torch.stack(metapath_reprs)  # (7, hidden_dim)
    
    def extract_brg_context(self, node_features, edge_index, disease_idx):
        """
        FIX 2: Active BRG context extraction
        
        Instead of zeros, extracts 1-hop neighborhood features around
        the disease node as regulatory context.
        """
        # Find 1-hop neighbors of disease node
        mask = (edge_index[1] == disease_idx) | (edge_index[0] == disease_idx)
        neighbor_indices = torch.cat([
            edge_index[0][mask],
            edge_index[1][mask]
        ]).unique()
        
        if len(neighbor_indices) == 0:
            return torch.zeros(self.hidden_dim, device=node_features.device)
        
        # Average neighbor features as regulatory context
        neighbor_feats = node_features[neighbor_indices].mean(dim=0)  # (hidden_dim,)
        
        # Project through BRG-specific layer
        brg_context = self.brg_projection(neighbor_feats)  # (hidden_dim,)
        
        return brg_context
    
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
        
        self.eval()
        with torch.no_grad():
            # Run NBFNet for all nodes
            nbfnet_features = self.nbfnet(
                node_features,
                edge_index,
                edge_type,
                query_drug
            )
            
            # Simple scoring for all nodes (for ranking)
            drug_repr = nbfnet_features[query_drug]
            
            # Score all nodes
            scores = []
            for i in range(nbfnet_features.size(0)):
                pair_repr = torch.cat([drug_repr, nbfnet_features[i]], dim=0)
                score = self.predictor(pair_repr)
                scores.append(score)
            
            scores = torch.cat(scores)
        
        return scores

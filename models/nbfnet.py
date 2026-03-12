"""
NBFNet Layer - Neural Bellman-Ford Network
===========================================

Implements message passing based on Bellman-Ford shortest path algorithm.
This is the core component of BioPathNet++ that learns path-based reasoning.

Key Idea:
- Traditional GNNs aggregate from neighbors
- NBFNet explicitly models paths from source to target
- Uses Bellman-Ford iterations to propagate information along paths

Reference: 
Zhu et al., "Neural Bellman-Ford Networks" (ICLR 2021)

Author: Yatin
Date: February 10, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class NBFNetConv(MessagePassing):
    """
    NBFNet Convolutional Layer
    
    Performs one iteration of Bellman-Ford message passing:
    1. For each edge (u, v) with relation r:
       - Compute message from u to v
       - Weight by relation type
    2. Aggregate messages at each node
    3. Update node representations
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        num_relations (int): Number of relation types in the graph
        message_func (str): How to compute messages ('rotate', 'distmult', 'complex')
        aggregate_func (str): How to aggregate ('sum', 'mean', 'max', 'pna')
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        num_relations,
        message_func='rotate',
        aggregate_func='pna',
        num_layers=1,
        dropout=0.0
    ):
        super(NBFNetConv, self).__init__(aggr=None)  # We'll handle aggregation manually
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Relation embeddings (one per relation type)
        self.relation_embedding = nn.Embedding(num_relations, in_channels)
        
        # Message transformation (different for each relation)
        if message_func == 'rotate':
            # RotatE-style: rotation in complex space
            self.W_msg = nn.Parameter(torch.Tensor(num_relations, in_channels))
        elif message_func == 'distmult':
            # DistMult-style: element-wise product
            self.W_msg = nn.Parameter(torch.Tensor(num_relations, in_channels))
        else:
            # Default: learned linear transformation per relation
            self.W_msg = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        
        # Boundary (self-loop) transformation
        self.W_bound = nn.Linear(in_channels, out_channels)
        
        # Update function (combines message and boundary)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # For PNA aggregation (multiple aggregators)
        if aggregate_func == 'pna':
            self.aggregators = ['sum', 'mean', 'max', 'min']
            self.scalers = ['identity', 'amplification', 'attenuation']
            self.pna_linear = nn.Linear(
                out_channels * len(self.aggregators) * len(self.scalers),
                out_channels
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.W_msg)
        self.W_bound.reset_parameters()
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    
    def forward(self, x, edge_index, edge_type, boundary):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_type: Edge relation types (num_edges,)
            boundary: Boundary condition - query node representation (in_channels,)
        
        Returns:
            Updated node features (num_nodes, out_channels)
        """
        
        # Propagate messages
        messages = self.propagate(
            edge_index,
            x=x,
            edge_type=edge_type
        )  # (num_nodes, out_channels)
        
        # Boundary (self-loop from query)
        boundary_repr = self.W_bound(boundary)  # (out_channels,)
        boundary_repr = boundary_repr.unsqueeze(0).expand(x.size(0), -1)  # (num_nodes, out_channels)
        
        # Combine messages and boundary
        combined = torch.cat([messages, boundary_repr], dim=-1)  # (num_nodes, out_channels * 2)
        
        # Update
        updated = self.mlp(combined)  # (num_nodes, out_channels)
        
        # Residual connection
        if x.size(-1) == updated.size(-1):
            updated = updated + x
        
        # Layer normalization
        updated = self.layer_norm(updated)
        
        return updated
    
    def message(self, x_j, edge_type):
        """
        Compute messages from source nodes to target nodes
        
        Args:
            x_j: Source node features (num_edges, in_channels)
            edge_type: Relation types (num_edges,)
        
        Returns:
            Messages (num_edges, out_channels)
        """
        
        if self.message_func == 'rotate':
            # RotatE: h_j * r (complex rotation)
            rel_emb = self.W_msg[edge_type]  # (num_edges, in_channels)
            
            # Treat as complex numbers: alternate dims are real/imaginary
            # Split into real and imaginary parts
            h_real, h_imag = x_j[..., ::2], x_j[..., 1::2]
            r_real, r_imag = rel_emb[..., ::2], rel_emb[..., 1::2]
            
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            msg_real = h_real * r_real - h_imag * r_imag
            msg_imag = h_real * r_imag + h_imag * r_real
            
            # Interleave back
            msg = torch.stack([msg_real, msg_imag], dim=-1).flatten(-2)
            
        elif self.message_func == 'distmult':
            # DistMult: h_j ⊙ r (element-wise product)
            rel_emb = self.W_msg[edge_type]  # (num_edges, in_channels)
            msg = x_j * rel_emb
            
        else:
            # Default: linear transformation per relation
            # msg = h_j @ W_r
            rel_weight = self.W_msg[edge_type]  # (num_edges, in_channels, out_channels)
            msg = torch.einsum('ei,eio->eo', x_j, rel_weight)
        
        return msg
    
    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregate messages at each node
        
        Args:
            inputs: Messages (num_edges, out_channels)
            index: Target node indices (num_edges,)
            dim_size: Number of nodes
        
        Returns:
            Aggregated messages (num_nodes, out_channels)
        """
        
        if self.aggregate_func == 'sum':
            return torch.zeros(dim_size, inputs.size(-1), device=inputs.device).scatter_add_(
                0, index.unsqueeze(-1).expand_as(inputs), inputs
            )
        
        elif self.aggregate_func == 'mean':
            sum_msg = torch.zeros(dim_size, inputs.size(-1), device=inputs.device).scatter_add_(
                0, index.unsqueeze(-1).expand_as(inputs), inputs
            )
            count = torch.zeros(dim_size, device=inputs.device).scatter_add_(
                0, index, torch.ones_like(index, dtype=torch.float)
            )
            return sum_msg / count.clamp(min=1).unsqueeze(-1)
        
        elif self.aggregate_func == 'max':
            max_msg = torch.full(
                (dim_size, inputs.size(-1)),
                float('-inf'),
                device=inputs.device
            )
            return max_msg.scatter_reduce_(
                0, index.unsqueeze(-1).expand_as(inputs), inputs, reduce='amax'
            )
        
        elif self.aggregate_func == 'pna':
            # Principal Neighbourhood Aggregation
            # Combines multiple aggregators and scalers
            
            aggregated = []
            
            for aggregator in self.aggregators:
                if aggregator == 'sum':
                    agg = torch.zeros(dim_size, inputs.size(-1), device=inputs.device).scatter_add_(
                        0, index.unsqueeze(-1).expand_as(inputs), inputs
                    )
                elif aggregator == 'mean':
                    sum_msg = torch.zeros(dim_size, inputs.size(-1), device=inputs.device).scatter_add_(
                        0, index.unsqueeze(-1).expand_as(inputs), inputs
                    )
                    count = torch.zeros(dim_size, device=inputs.device).scatter_add_(
                        0, index, torch.ones_like(index, dtype=torch.float)
                    )
                    agg = sum_msg / count.clamp(min=1).unsqueeze(-1)
                elif aggregator == 'max':
                    agg = torch.full(
                        (dim_size, inputs.size(-1)),
                        float('-inf'),
                        device=inputs.device
                    )
                    agg = agg.scatter_reduce_(
                        0, index.unsqueeze(-1).expand_as(inputs), inputs, reduce='amax'
                    )
                elif aggregator == 'min':
                    agg = torch.full(
                        (dim_size, inputs.size(-1)),
                        float('inf'),
                        device=inputs.device
                    )
                    agg = agg.scatter_reduce_(
                        0, index.unsqueeze(-1).expand_as(inputs), inputs, reduce='amin'
                    )
                
                # Apply scalers
                for scaler in self.scalers:
                    if scaler == 'identity':
                        aggregated.append(agg)
                    elif scaler == 'amplification':
                        aggregated.append(agg * 2)
                    elif scaler == 'attenuation':
                        aggregated.append(agg * 0.5)
            
            # Concatenate and project
            aggregated = torch.cat(aggregated, dim=-1)
            return self.pna_linear(aggregated)
        
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregate_func}")


class NBFNet(nn.Module):
    """
    Complete NBFNet model with multiple layers
    
    Stacks multiple NBFNetConv layers to enable multi-hop reasoning.
    Each layer extends the paths by one hop.
    
    Args:
        num_nodes (int): Number of nodes in the graph
        num_relations (int): Number of relation types
        hidden_dim (int): Hidden dimension
        num_layers (int): Number of NBFNet layers (path length)
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        num_nodes,
        num_relations,
        hidden_dim=256,
        num_layers=3,
        dropout=0.1,
        message_func='rotate',
        aggregate_func='pna'
    ):
        super(NBFNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection (if needed)
        # Assuming node features are already embedded
        
        # NBFNet layers
        self.layers = nn.ModuleList([
            NBFNetConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_relations=num_relations,
                message_func=message_func,
                aggregate_func=aggregate_func,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_type, query_node):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Graph structure (2, num_edges)
            edge_type: Edge types (num_edges,)
            query_node: Query node index (1,) or (batch_size,)
        
        Returns:
            Node representations after K-hop reasoning (num_nodes, hidden_dim)
        """
        
        # Get boundary (query node representation)
        if query_node.dim() == 0:
            query_node = query_node.unsqueeze(0)
        
        boundary = x[query_node[0]]  # (hidden_dim,)
        
        # Apply NBFNet layers (K iterations of Bellman-Ford)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_type, boundary)
            h = self.dropout(h)
        
        return h
    
    def predict(self, h_source, h_targets):
        """
        Predict scores for source-target pairs
        
        Args:
            h_source: Source node representation (hidden_dim,) or (batch, hidden_dim)
            h_targets: Target node representations (num_targets, hidden_dim)
        
        Returns:
            Scores (num_targets,) or (batch, num_targets)
        """
        
        # Simple dot product scoring
        if h_source.dim() == 1:
            scores = torch.matmul(h_targets, h_source)  # (num_targets,)
        else:
            scores = torch.matmul(h_targets, h_source.t())  # (num_targets, batch)
            scores = scores.t()  # (batch, num_targets)
        
        return scores


# Testing code
if __name__ == '__main__':
    """
    Test NBFNet on a toy graph
    """
    print("Testing NBFNet implementation...")
    
    # Toy graph
    num_nodes = 10
    num_relations = 5
    num_edges = 20
    hidden_dim = 64
    
    # Random features
    x = torch.randn(num_nodes, hidden_dim)
    
    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    # Query node
    query_node = torch.tensor([0])
    
    # Model
    model = NBFNet(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    h = model(x, edge_index, edge_type, query_node)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {h.shape}")
    
    # Prediction
    target_nodes = torch.tensor([5, 7, 9])
    scores = model.predict(h[query_node[0]], h[target_nodes])
    
    print(f"Prediction scores: {scores}")
    print("\n✓ NBFNet test passed!")

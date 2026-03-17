"""
NBFNet Layer - Neural Bellman-Ford Network
===========================================

Memory-efficient version using k-hop subgraph sampling.
Instead of propagating over all 1.9M edges at once,
we extract a small subgraph around each query drug node.

84,712 nodes x 1,906,058 edges full-graph = OOM on any GPU.
2-hop subgraph per drug = ~500-2000 edges = fits easily.

Author: Yatin
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph


class NBFNetConv(MessagePassing):
    """
    NBFNet Convolutional Layer.
    Operates on a small subgraph, not the full graph.
    Uses distmult (lighter than rotate, same quality for this task).
    """

    def __init__(
        self,
        hidden_dim,
        num_relations,
        message_func='distmult',
        aggregate_func='sum',
        dropout=0.1
    ):
        super(NBFNetConv, self).__init__(aggr=None)

        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.message_func = message_func
        self.aggregate_func = aggregate_func

        # Relation embeddings — same dim as node features
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        # Boundary transformation
        self.W_bound = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Update MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.W_bound.reset_parameters()
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_type, boundary):
        """
        Args:
            x:          Node features      (num_subgraph_nodes, hidden_dim)
            edge_index: Subgraph edges     (2, num_subgraph_edges)
            edge_type:  Edge types         (num_subgraph_edges,)
            boundary:   Query node repr    (hidden_dim,)
        Returns:
            Updated node features (num_subgraph_nodes, hidden_dim)
        """
        messages = self.propagate(
            edge_index,
            x=x,
            edge_type=edge_type,
            size=(x.size(0), x.size(0))
        )

        boundary_repr = self.W_bound(boundary).unsqueeze(0).expand_as(messages)
        combined = torch.cat([messages, boundary_repr], dim=-1)
        updated = self.mlp(combined)
        updated = self.layer_norm(updated + x)
        return updated

    def message(self, x_j, edge_type):
        """
        Args:
            x_j:       Source node features (num_edges, hidden_dim)
            edge_type: Relation types       (num_edges,)
        Returns:
            Messages (num_edges, hidden_dim)
        """
        rel_emb = self.relation_embedding(edge_type)  # (num_edges, hidden_dim)

        if self.message_func == 'distmult':
            return x_j * rel_emb

        elif self.message_func == 'rotate':
            half = self.hidden_dim // 2
            h_real, h_imag = x_j[:, :half], x_j[:, half:]
            r_real, r_imag = rel_emb[:, :half], rel_emb[:, half:]
            msg_real = h_real * r_real - h_imag * r_imag
            msg_imag = h_real * r_imag + h_imag * r_real
            return torch.cat([msg_real, msg_imag], dim=-1)

        else:
            return x_j + rel_emb

    def aggregate(self, inputs, index, dim_size=None):
        if dim_size is None:
            dim_size = index.max().item() + 1

        out = torch.zeros(dim_size, inputs.size(-1), device=inputs.device)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(inputs), inputs)
        return out


class NBFNet(nn.Module):
    """
    Memory-efficient NBFNet using k-hop subgraph sampling.

    Key idea: for each query drug, extract its 2-hop neighborhood
    (~500-2000 edges) instead of using all 1.9M edges.
    This reduces memory from ~83GB to ~0.1GB per forward pass.

    Args:
        num_nodes (int):     Total nodes in full graph
        num_relations (int): Number of relation types
        hidden_dim (int):    Hidden dimension (uniform everywhere)
        num_layers (int):    NBF layers = max path length explored
        dropout (float):     Dropout rate
        num_hops (int):      Subgraph neighborhood size (2 is optimal)
    """

    def __init__(
        self,
        num_nodes,
        num_relations,
        hidden_dim=512,
        num_layers=4,
        dropout=0.1,
        num_hops=2
    ):
        super(NBFNet, self).__init__()

        assert hidden_dim % 2 == 0, "hidden_dim must be even"

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_hops = num_hops

        # Project BioBERT 768-dim embeddings to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # NBFNet layers
        self.layers = nn.ModuleList([
            NBFNetConv(
                hidden_dim=hidden_dim,
                num_relations=num_relations,
                message_func='distmult',
                aggregate_func='sum',
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def extract_subgraph(self, query_node_idx, full_edge_index, full_edge_type, num_nodes):
        """
        Extract k-hop subgraph around the query drug node.

        Returns:
            sub_edge_index: Edges within subgraph, remapped to local indices
            sub_edge_type:  Relation types for those edges
            node_subset:    Global node indices included in subgraph
            local_query:    Local index of the query node within subgraph
        """
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=query_node_idx.item(),
            num_hops=self.num_hops,
            edge_index=full_edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

        sub_edge_type = full_edge_type[edge_mask]
        local_query = mapping[0] if mapping.numel() > 0 else torch.tensor(0)

        return sub_edge_index, sub_edge_type, subset, local_query

    def forward(self, x, edge_index, edge_type, query_node):
        """
        Args:
            x:           Full node features  (num_nodes, 768) from BioBERT
            edge_index:  Full graph          (2, num_edges)
            edge_type:   Full edge types     (num_edges,)
            query_node:  Query drug index    scalar or (1,)

        Returns:
            full_h: Node representations mapped back to full graph size
                    (num_nodes, hidden_dim) — zeros for non-subgraph nodes
        """
        if query_node.dim() == 0:
            query_node = query_node.unsqueeze(0)

        device = x.device

        # Step 1: Extract k-hop subgraph around query drug
        sub_edge_index, sub_edge_type, node_subset, local_query = \
            self.extract_subgraph(query_node, edge_index, edge_type, x.size(0))

        # Step 2: Project subgraph node features from 768 to hidden_dim
        sub_x = self.input_proj(x[node_subset])  # (sub_nodes, hidden_dim)

        # Step 3: Boundary = projected query node representation
        boundary = sub_x[local_query]  # (hidden_dim,)

        # Step 4: K layers of message passing on small subgraph only
        h = sub_x
        for layer in self.layers:
            h = layer(h, sub_edge_index, sub_edge_type, boundary)
            h = self.dropout(h)

        # Step 5: Map subgraph results back to full graph size
        full_h = torch.zeros(x.size(0), self.hidden_dim, device=device)
        full_h[node_subset] = h

        return full_h  # (num_nodes, hidden_dim)

    def predict(self, h_source, h_targets):
        """
        Dot-product scoring.
        Args:
            h_source:  (hidden_dim,)
            h_targets: (num_targets, hidden_dim)
        Returns:
            scores (num_targets,)
        """
        return torch.matmul(h_targets, h_source)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing memory-efficient NBFNet...")

    num_nodes     = 1000
    num_relations = 50
    num_edges     = 5000
    hidden_dim    = 512

    x          = torch.randn(num_nodes, 768)       # BioBERT-dim input
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type  = torch.randint(0, num_relations, (num_edges,))
    query_node = torch.tensor(0)

    model = NBFNet(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        num_layers=4,
        num_hops=2
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    full_h = model(x, edge_index, edge_type, query_node)
    print(f"Output shape: {full_h.shape}")  # (1000, 512)

    targets = torch.tensor([5, 10, 20])
    scores  = model.predict(full_h[query_node], full_h[targets])
    print(f"Score shape:  {scores.shape}")  # (3,)

    print("✓ Memory-efficient NBFNet test passed!")

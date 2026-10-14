"""
BioPathNet++ v3 model
=====================

Core: Neural Bellman-Ford Network (Zhu et al., NeurIPS 2021) as used by BioPathNet
(Hu et al., Nat Biomed Eng 2026), adapted for zero-shot drug repurposing:

  * query  = (disease, therapeutic relation)  ->  scores for EVERY drug in one pass
  * boundary condition = indicator: the query embedding sits on the source disease only
  * message = DistMult (h_u * w_r), aggregation = mean + log-degree amplification (PNA-lite)
  * propagation over relation-wise sparse matrices with a custom autograd function that
    recomputes A_r @ X in backward, so memory is O(V * B * d) instead of O(E * B * d).
    This is what makes full-graph NBFNet fit on a 16 GB T4.

"++" (what this project adds on top of BioPathNet):
  * a cross-disease TRANSFER channel: leakage-free prior features per (disease, drug)
    -- drug popularity among *other* diseases and a disease-similarity kNN transfer score --
    fused with the path representation by a learned head (replaces the hand-weighted
    0.40/0.25/0.20/0.15 "hybrid score" of v2, whose neural term was an unnormalised logit).
  * training regime that mirrors zero-shot inference (batch diseases lose all their drug
    edges), handled in kg.MessageGraph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelSpMM(torch.autograd.Function):
    """out[v, b] = sum_r (A_r @ X)[v, b] * w[b, r]   with recompute in backward.

    x: (V, B, d)   w: (B, R, d)   mats/mats_t: list of R sparse CSR (V x V) and transposes.
    """

    @staticmethod
    def forward(ctx, x, w, mats, mats_t):
        V, B, d = x.shape
        x2 = x.reshape(V, B * d)
        out = x.new_zeros(V, B, d)
        for r, a in enumerate(mats):
            if a._nnz() == 0:
                continue
            out += torch.sparse.mm(a, x2).view(V, B, d) * w[:, r].unsqueeze(0)
        ctx.save_for_backward(x, w)
        ctx.mats, ctx.mats_t = mats, mats_t
        return out

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        V, B, d = x.shape
        x2 = x.reshape(V, B * d)
        gx = torch.zeros_like(x2) if ctx.needs_input_grad[0] else None
        gw = torch.zeros_like(w) if ctx.needs_input_grad[1] else None
        for r, (a, at) in enumerate(zip(ctx.mats, ctx.mats_t)):
            if a._nnz() == 0:
                continue
            if gx is not None:
                gx += torch.sparse.mm(at, (g * w[:, r].unsqueeze(0)).reshape(V, B * d))
            if gw is not None:
                y = torch.sparse.mm(a, x2).view(V, B, d)
                gw[:, r] = (y * g).sum(0)
        return (gx.view(V, B, d) if gx is not None else None), gw, None, None


class NBFLayer(nn.Module):
    def __init__(self, dim: int, num_relations: int, num_query_relations: int):
        super().__init__()
        # query-dependent DistMult relation vectors, initialised near identity
        self.rel = nn.Parameter(1.0 + 0.1 * torch.randn(num_query_relations, num_relations, dim))
        self.lin = nn.Linear(3 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, qrel, boundary_idx, boundary_val, mats, mats_t, inv_deg, amp):
        w = self.rel[qrel]                                            # (B, R, d)
        s = RelSpMM.apply(h.float(), w.float(), mats, mats_t)         # (V, B, d)
        mean = s * inv_deg.view(-1, 1, 1)
        mean = mean.index_put(boundary_idx, boundary_val, accumulate=True)  # boundary as self-message
        agg = torch.cat([h, mean, mean * amp.view(-1, 1, 1)], dim=-1)
        out = F.relu(self.norm(self.lin(agg)))
        return out + h                                                # shortcut


class BioPathNetPP(nn.Module):
    def __init__(self, num_relations: int, num_query_relations: int = 2, dim: int = 32,
                 num_layers: int = 6, mlp_hidden: int = 64, num_prior_feats: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.dim, self.num_layers = dim, num_layers
        self.query = nn.Embedding(num_query_relations, dim)
        self.layers = nn.ModuleList(NBFLayer(dim, num_relations, num_query_relations)
                                    for _ in range(num_layers))
        self.num_prior_feats = num_prior_feats
        feat_dim = dim * num_layers + dim
        self.mlp = nn.Sequential(nn.Linear(feat_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_hidden, 1))
        if num_prior_feats:
            # transfer channel: small MLP on prior features, added to the path logit
            self.prior = nn.Sequential(nn.Linear(num_prior_feats, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, graph, src: torch.Tensor, qrel: torch.Tensor, cand: torch.Tensor,
                prior: torch.Tensor | None = None, use_paths: bool = True) -> torch.Tensor:
        """
        graph: (mats, mats_t, deg) from kg.MessageGraph.build
        src:   (B,) query disease node ids     qrel: (B,) query relation ids
        cand:  (C,) candidate drug node ids     prior: (B, C, F) optional transfer features
        returns logits (B, C)
        """
        mats, mats_t, deg = graph
        V, B = deg.numel(), src.numel()
        q = self.query(qrel)                                          # (B, d)
        logits = None
        if use_paths:
            inv_deg = 1.0 / deg.clamp(min=1.0)
            logd = torch.log(deg + 1.0)
            amp = logd / logd.mean().clamp(min=1e-6)
            ar = torch.arange(B, device=src.device)
            bidx, bval = (src, ar), q
            h = q.new_zeros(V, B, self.dim).index_put((src, ar), q)  # indicator boundary
            hs = []
            for layer in self.layers:
                h = layer(h, qrel, bidx, bval, mats, mats_t, inv_deg, amp)
                hs.append(h[cand])                                    # keep only candidate rows
            feat = torch.cat(hs, dim=-1).transpose(0, 1)              # (B, C, L*d)
            qx = q.unsqueeze(1).expand(-1, cand.numel(), -1)
            logits = self.mlp(torch.cat([feat, qx], dim=-1)).squeeze(-1)
        if self.num_prior_feats and prior is not None:
            p = self.prior(prior).squeeze(-1)
            logits = p if logits is None else logits + p
        return logits


def listwise_loss(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """Multi-positive softmax cross-entropy over ALL candidate drugs (InfoNCE with every
    non-indicated drug as a negative). pos_mask: (B, C) bool, >= 1 positive per row."""
    logp = logits.log_softmax(dim=-1)
    per_row = -(logp * pos_mask).sum(-1) / pos_mask.sum(-1).clamp(min=1)
    return per_row.mean()


def bce_loss(logits: torch.Tensor, pos_mask: torch.Tensor, adv_temp: float = 0.5) -> torch.Tensor:
    """NBFNet-style BCE with self-adversarial weighting of negatives (ablation)."""
    pos = pos_mask.float()
    lp = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    ln = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    with torch.no_grad():
        wneg = torch.softmax(logits.masked_fill(pos_mask, -1e9) / adv_temp, dim=-1)
    lpos = (lp * pos).sum(-1) / pos.sum(-1).clamp(min=1)
    lneg = (ln * wneg).sum(-1)
    return (lpos + lneg).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

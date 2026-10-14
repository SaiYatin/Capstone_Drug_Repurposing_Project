"""
Compact knowledge-graph format and leakage-controlled graph construction.

On-disk format (one directory):
  meta.json        node_types, relation names, therapeutic relation names, split info
  nodes.npz        node_type (int16, V), node names in meta.json["node_names"] is avoided for size;
                   names are stored in nodes.json
  edges.npz        head, rel, tail (int32) -- *context* edges (non-therapeutic), one direction
  therapeutic.npz  disease, drug, trel (int32) -- drug-disease therapeutic pairs, deduplicated
  splits.json      train / valid / test diseases and held-out families

Leakage rules enforced here (not left to the training script):
  1. Therapeutic edges of every valid/test *family* disease are never in the message graph.
  2. Therapeutic edges of the diseases in the current training batch are removed for that
     batch, so training mirrors zero-shot inference.
  3. Every relation gets an explicit inverse relation, so information can flow both ways
     (fixes the incoming-only k_hop_subgraph bug of v1/v2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# On-disk dataset
# --------------------------------------------------------------------------------------
@dataclass
class CompactKG:
    root: Path
    meta: dict
    node_type: np.ndarray            # (V,) int16
    node_names: list                 # (V,) str
    head: np.ndarray                 # (E,) int32 context edges
    rel: np.ndarray
    tail: np.ndarray
    t_disease: np.ndarray            # (T,) therapeutic pairs
    t_drug: np.ndarray
    t_rel: np.ndarray
    splits: dict = field(default_factory=dict)

    # ---------------- convenience ----------------
    @property
    def num_nodes(self) -> int:
        return len(self.node_type)

    @property
    def relation_names(self) -> list:
        return self.meta["relations"]

    @property
    def therapeutic_names(self) -> list:
        return self.meta["therapeutic_relations"]

    def type_id(self, name: str) -> int:
        return self.meta["node_types"].index(name)

    def drug_ids(self) -> np.ndarray:
        return np.flatnonzero(self.node_type == self.type_id(self.meta["drug_type"]))

    def disease_ids(self) -> np.ndarray:
        return np.flatnonzero(self.node_type == self.type_id(self.meta["disease_type"]))

    def positives(self, trel_name: str = "indication") -> dict:
        """disease -> set(drug) for one therapeutic relation."""
        k = self.therapeutic_names.index(trel_name)
        m = self.t_rel == k
        out: dict = {}
        for d, g in zip(self.t_disease[m].tolist(), self.t_drug[m].tolist()):
            out.setdefault(d, set()).add(g)
        return out

    # ---------------- io ----------------
    @staticmethod
    def load(root) -> "CompactKG":
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        nodes = np.load(root / "nodes.npz")
        names = json.loads((root / "nodes.json").read_text())
        e = np.load(root / "edges.npz")
        t = np.load(root / "therapeutic.npz")
        splits = json.loads((root / "splits.json").read_text()) if (root / "splits.json").exists() else {}
        return CompactKG(root, meta, nodes["node_type"], names, e["head"], e["rel"], e["tail"],
                         t["disease"], t["drug"], t["trel"], splits)

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2))
        np.savez_compressed(self.root / "nodes.npz", node_type=self.node_type.astype(np.int16))
        (self.root / "nodes.json").write_text(json.dumps(self.node_names))
        np.savez_compressed(self.root / "edges.npz", head=self.head.astype(np.int32),
                            rel=self.rel.astype(np.int32), tail=self.tail.astype(np.int32))
        np.savez_compressed(self.root / "therapeutic.npz", disease=self.t_disease.astype(np.int32),
                            drug=self.t_drug.astype(np.int32), trel=self.t_rel.astype(np.int32))
        if self.splits:
            (self.root / "splits.json").write_text(json.dumps(self.splits, indent=2))


# --------------------------------------------------------------------------------------
# Device-side graph with per-batch therapeutic edges
# --------------------------------------------------------------------------------------
def _csr(rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, n: int) -> torch.Tensor:
    """Sparse CSR (n x n) with A[row, col] = val. Rows = message targets, cols = sources."""
    if rows.numel() == 0:
        crow = torch.zeros(n + 1, dtype=torch.int64, device=rows.device)
        return torch.sparse_csr_tensor(crow, rows.new_zeros(0), vals.new_zeros(0), (n, n))
    a = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()
    return a.to_sparse_csr()


class MessageGraph:
    """
    Holds the static context relations (binary CSR, both directions, built ONCE) and the
    therapeutic edges allowed in the graph (train-family diseases only). For each batch it
    rebuilds only the small therapeutic matrices, without the batch diseases' own edges.

    Relation layout of the returned list (R_total = 2 * (R_ctx + R_ther)):
        [ctx_fwd (R_ctx), ther_fwd (R_ther), ctx_inv (R_ctx), ther_inv (R_ther)]
    Matrices are binary (A[v, u] = 1 for a message u -> v). The model applies mean
    normalisation diag(1/deg) after the sum, plus PNA-style log-degree amplification, so the
    static matrices never need rebuilding when the batch changes degrees.
    """

    def __init__(self, kg: CompactKG, allowed_ther_diseases, device="cpu", include_relations=None):
        self.device = torch.device(device)
        self.n = kg.num_nodes
        rel_names = kg.relation_names
        keep = np.ones(len(kg.rel), dtype=bool)
        if include_relations is not None:
            keep_ids = [rel_names.index(r) for r in include_relations]
            keep = np.isin(kg.rel, keep_ids)
        self.num_ctx = len(rel_names)  # keep full relation id space for stable parameter shapes
        self.num_ther = len(kg.therapeutic_names)
        self.num_relations = 2 * (self.num_ctx + self.num_ther)

        h = torch.as_tensor(kg.head[keep], dtype=torch.int64)
        r = torch.as_tensor(kg.rel[keep], dtype=torch.int64)
        t = torch.as_tensor(kg.tail[keep], dtype=torch.int64)

        allowed = np.isin(kg.t_disease, np.asarray(sorted(allowed_ther_diseases), dtype=np.int64))
        self.td = torch.as_tensor(kg.t_disease[allowed], dtype=torch.int64, device=self.device)
        self.tg = torch.as_tensor(kg.t_drug[allowed], dtype=torch.int64, device=self.device)
        self.tr = torch.as_tensor(kg.t_rel[allowed], dtype=torch.int64, device=self.device)

        # static in-degree (both directions of context edges)
        deg = torch.bincount(t, minlength=self.n) + torch.bincount(h, minlength=self.n)
        self.static_deg = deg.to(self.device).float()

        # static binary CSR per (relation, direction), and transposes for the backward pass
        self.ctx_fwd, self.ctx_fwd_t, self.ctx_inv, self.ctx_inv_t = [], [], [], []
        for k in range(self.num_ctx):
            m = r == k
            hk, tk = h[m].to(self.device), t[m].to(self.device)
            ones = torch.ones(hk.numel(), device=self.device)
            self.ctx_fwd.append(_csr(tk, hk, ones, self.n))     # message h -> t (row = t)
            self.ctx_fwd_t.append(_csr(hk, tk, ones, self.n))
            self.ctx_inv.append(_csr(hk, tk, ones, self.n))     # message t -> h (row = h)
            self.ctx_inv_t.append(_csr(tk, hk, ones, self.n))
        self.num_ctx_edges = int(keep.sum())

    # ------------------------------------------------------------------
    def build(self, drop_diseases: torch.Tensor | None = None):
        """Return (mats, mats_T, deg) for one batch.

        drop_diseases: diseases whose therapeutic edges must be removed for this batch
        (the batch's own query diseases during training; nothing at evaluation, because
        valid/test families were never allowed in the graph)."""
        keep = torch.ones_like(self.td, dtype=torch.bool)
        if drop_diseases is not None and drop_diseases.numel() > 0:
            keep = ~torch.isin(self.td, drop_diseases.to(self.device))
        td, tg, tr = self.td[keep], self.tg[keep], self.tr[keep]

        deg = self.static_deg.clone()
        deg += torch.bincount(td, minlength=self.n).float() + torch.bincount(tg, minlength=self.n).float()

        ther_fwd, ther_fwd_t, ther_inv, ther_inv_t = [], [], [], []
        for k in range(self.num_ther):
            m = tr == k
            dk, gk = td[m], tg[m]
            ones = torch.ones(dk.numel(), device=self.device)
            ther_fwd.append(_csr(dk, gk, ones, self.n))      # drug -> disease (row = disease)
            ther_fwd_t.append(_csr(gk, dk, ones, self.n))
            ther_inv.append(_csr(gk, dk, ones, self.n))      # disease -> drug (row = drug)
            ther_inv_t.append(_csr(dk, gk, ones, self.n))
        mats = self.ctx_fwd + ther_fwd + self.ctx_inv + ther_inv
        mats_t = self.ctx_fwd_t + ther_fwd_t + self.ctx_inv_t + ther_inv_t
        return mats, mats_t, deg

"""
Task definition shared by the model and the baselines: candidates, labels, exclusion sets,
leakage-free transfer priors and the non-neural baselines.

Leakage rule for priors and baselines: when scoring disease d, the label pool is the TRAIN
diseases minus E(d) = {d} U disease-disease neighbours(d). At test time d's whole strict family
is held out anyway; applying the same exclusion to training queries keeps train and test
conditions identical (no "copy the sibling" shortcut).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from bpp.kg import CompactKG


class RepurposingTask:
    def __init__(self, kg: CompactKG, split: dict, knn_k: int = 20):
        self.kg, self.split, self.knn_k = kg, split, knn_k
        self.drugs = kg.drug_ids()
        self.C = len(self.drugs)
        self.col = {int(d): i for i, d in enumerate(self.drugs)}
        self.ind = kg.positives("indication")
        self.off = kg.positives("off-label use")
        self.train = [int(d) for d in split["train_diseases"]]
        self.valid = [int(d) for d in split["valid_diseases"]]
        self.train_row = {d: i for i, d in enumerate(self.train)}

        rel = kg.relation_names
        def edges(name):
            if name not in rel:
                return np.zeros(0, int), np.zeros(0, int)
            m = kg.rel == rel.index(name)
            return kg.head[m], kg.tail[m]

        # disease-disease neighbours
        self.nb: dict = {}
        h, t = edges("disease_disease")
        for a, b in zip(h.tolist(), t.tolist()):
            self.nb.setdefault(a, set()).add(b); self.nb.setdefault(b, set()).add(a)

        V = kg.num_nodes
        # disease feature matrix (proteins + phenotypes), tf-idf, L2-normalised
        rows, cols = [], []
        for name in ("disease_protein", "disease_phenotype_positive"):
            h, t = edges(name)
            dis_is_h = kg.node_type[h] == kg.type_id(kg.meta["disease_type"]) if len(h) else np.zeros(0, bool)
            rows.append(np.where(dis_is_h, h, t)); cols.append(np.where(dis_is_h, t, h))
        r, c = np.concatenate(rows), np.concatenate(cols)
        F = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(V, V))
        df = np.asarray((F > 0).sum(0)).ravel()
        idf = np.log((F.shape[0] + 1) / (df + 1))
        F = F.multiply(idf).tocsr()
        norm = np.sqrt(np.asarray(F.multiply(F).sum(1)).ravel()) + 1e-12
        self.F = sp.diags(1 / norm) @ F
        self.F_train = self.F[self.train]

        # drug targets and disease genes -> shared-target counts
        h, t = edges("drug_protein")
        drug_is_h = kg.node_type[h] == kg.type_id(kg.meta["drug_type"])
        dg, pr = np.where(drug_is_h, h, t), np.where(drug_is_h, t, h)
        self.DT = sp.csr_matrix((np.ones(len(dg)), ([self.col[x] for x in dg], pr)), shape=(self.C, V))
        h, t = edges("disease_protein")
        dis_is_h = kg.node_type[h] == kg.type_id(kg.meta["disease_type"])
        ds, pr = np.where(dis_is_h, h, t), np.where(dis_is_h, t, h)
        self.DG = sp.csr_matrix((np.ones(len(ds)), (ds, pr)), shape=(V, V))

        # train label matrix
        yr, yc = [], []
        for d in self.train:
            for g in self.ind.get(d, ()):
                yr.append(self.train_row[d]); yc.append(self.col[g])
        self.Y = sp.csr_matrix((np.ones(len(yr)), (yr, yc)), shape=(len(self.train), self.C))
        self.pop_total = np.asarray(self.Y.sum(0)).ravel()

        # context degree of each drug (for degree-matched negatives)
        deg = np.bincount(kg.head, minlength=V) + np.bincount(kg.tail, minlength=V)
        self.drug_degree = deg[self.drugs].astype(float)

        # autoimmune set and hard-negative pool (all-KG labels, evaluation only)
        auto = set(split.get("area_diseases", []))
        for fam in split.get("rich_families", {}).values():
            auto |= set(fam)
        for fam in split.get("targets", {}).values():
            auto |= set(fam)
        self.autoimmune = auto
        pool = set()
        for d in auto:
            pool |= self.ind.get(d, set())
        self.auto_drug_pool = pool
        auto_train = [d for d in self.train if d in auto]
        self.auto_freq_total = np.zeros(self.C)
        for d in auto_train:
            self.auto_freq_total[self.Y[self.train_row[d]].indices] += 1
        self.auto_train = auto_train

    # ------------------------------------------------------------------ labels
    def exclusion(self, d: int) -> set:
        return {d} | self.nb.get(d, set())

    def eval_labels(self, d: int, include_offlabel: bool) -> np.ndarray:
        s = set(self.ind.get(d, set()))
        if include_offlabel:
            s |= self.off.get(d, set())
        return np.array(sorted(self.col[g] for g in s), dtype=int)

    def hard_negatives(self, pos_cols: np.ndarray) -> np.ndarray:
        pool = {self.col[g] for g in self.auto_drug_pool} - set(pos_cols.tolist())
        return np.array(sorted(pool), dtype=int)

    def eval_sets(self):
        """name -> list of (disease_label, node, include_offlabel)."""
        sets = {"valid": [(self.kg.node_names[d], d, False) for d in self.valid if d in self.ind]}
        sets["targets"] = [(n, int(p), True) for n, p in self.split["target_primary"].items()]
        if "eval_area" in self.split:          # transfer protocols (bpp/extra_splits.py)
            sets["area"] = [(self.kg.node_names[d], d, False) for d in self.split["eval_area"] if d in self.ind]
        elif self.split["protocol"] == "no_autoimmune":
            sets["area"] = [(self.kg.node_names[d], d, False) for d in self.split["area_diseases"]
                            if d in self.ind]
        if "test_diseases" in self.split:      # selection-free test set: never trained on, never used to stop
            sets["test"] = [(self.kg.node_names[d], d, False) for d in self.split["test_diseases"] if d in self.ind]
        return sets

    # ------------------------------------------------------------------ baselines / priors
    def _excl_rows(self, d):
        return [self.train_row[e] for e in self.exclusion(d) if e in self.train_row]

    def popularity(self, d: int) -> np.ndarray:
        p = self.pop_total.copy()
        rows = self._excl_rows(d)
        if rows:
            p -= np.asarray(self.Y[rows].sum(0)).ravel()
        return p

    def autoimmune_frequency(self, d: int) -> np.ndarray:
        p = self.auto_freq_total.copy()
        for e in self.exclusion(d):
            if e in self.train_row and e in self.autoimmune:
                p[self.Y[self.train_row[e]].indices] -= 1
        return p

    def knn(self, d: int) -> np.ndarray:
        sim = np.asarray((self.F_train @ self.F[d].T).todense()).ravel()
        for r in self._excl_rows(d):
            sim[r] = 0.0
        k = min(self.knn_k, len(sim))
        top = np.argpartition(-sim, k - 1)[:k]
        return np.asarray(self.Y[top].T @ sim[top]).ravel()

    def shared_targets(self, d: int) -> np.ndarray:
        return np.asarray((self.DT @ self.DG[d].T).todense()).ravel()

    def prior_features(self, diseases) -> torch.Tensor:
        """(B, C, 4) leakage-free transfer features for the BioPathNet++ transfer channel."""
        out = np.zeros((len(diseases), self.C, 4), dtype=np.float32)
        for i, d in enumerate(diseases):
            out[i, :, 0] = np.log1p(self.popularity(d))
            k = self.knn(d)
            out[i, :, 1] = k / (k.max() + 1e-9)
            out[i, :, 2] = np.log1p(self.shared_targets(d))
            out[i, :, 3] = np.log1p(self.autoimmune_frequency(d))
        return torch.from_numpy(out)

    def baselines(self, d: int, rng=None) -> dict:
        rng = rng or np.random.default_rng(0)
        pop = self.popularity(d)
        tie = rng.random(self.C) * 1e-6          # random tie-breaking, same for all baselines
        return {
            "random": rng.random(self.C),
            "drug_degree": self.drug_degree + tie,
            "popularity": pop + tie,
            "autoimmune_frequency": self.autoimmune_frequency(d) + 1e-3 * pop + tie,
            "shared_targets": self.shared_targets(d) + 1e-3 * pop + tie,
            "disease_knn": self.knn(d) + 1e-3 * pop + tie,
        }

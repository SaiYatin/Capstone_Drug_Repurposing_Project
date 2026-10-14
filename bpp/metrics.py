"""
Full-ranking evaluation for zero-shot drug repurposing.

For each query disease, every drug in the KG is ranked (no negative sampling at evaluation).
Per positive drug p, the rank is *filtered*: other positives of the same disease are removed
before ranking p (standard KG-completion protocol).

Reported per disease:  MRR, Hits@{1,10,50,100}, mean rank, AUROC, AUPRC (all drugs),
                       AUROC vs hard negatives (drugs indicated for other autoimmune diseases),
                       AUROC vs degree-matched negatives
Aggregates: macro mean over diseases with a bootstrap 95% CI (resampling diseases), and
            micro (pooled over positives) for MRR/Hits.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HITS = (1, 10, 50, 100)


def filtered_ranks(scores: np.ndarray, pos_idx: np.ndarray) -> np.ndarray:
    """1-based filtered rank of each positive. Ties are broken pessimistically-averaged."""
    pos_scores = scores[pos_idx]
    neg_mask = np.ones(len(scores), dtype=bool)
    neg_mask[pos_idx] = False
    neg = np.sort(scores[neg_mask])
    greater = len(neg) - np.searchsorted(neg, pos_scores, side="right")
    equal = np.searchsorted(neg, pos_scores, side="right") - np.searchsorted(neg, pos_scores, side="left")
    return 1 + greater + equal / 2.0


def _auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def disease_metrics(scores: np.ndarray, pos_idx: np.ndarray, hard_neg_idx=None,
                    degree=None, rng=None) -> dict:
    ranks = filtered_ranks(scores, pos_idx)
    y = np.zeros(len(scores)); y[pos_idx] = 1
    out = {"n_pos": int(len(pos_idx)), "mrr": float(np.mean(1.0 / ranks)),
           "mean_rank": float(np.mean(ranks)),
           "auroc": float(roc_auc_score(y, scores)), "auprc": float(average_precision_score(y, scores))}
    for k in HITS:
        out[f"hits@{k}"] = float(np.mean(ranks <= k))
    if hard_neg_idx is not None and len(hard_neg_idx):
        out["auroc_hard"] = _auroc(scores[pos_idx], scores[hard_neg_idx])
    if degree is not None:
        rng = rng or np.random.default_rng(0)
        negs = np.setdiff1d(np.arange(len(scores)), pos_idx)
        order = negs[np.argsort(degree[negs])]
        dsorted = degree[order]
        matched = []
        for p in pos_idx:               # 5 nearest-degree negatives per positive
            j = np.searchsorted(dsorted, degree[p])
            lo, hi = max(0, j - 5), min(len(order), j + 5)
            matched.extend(rng.choice(order[lo:hi], min(5, hi - lo), replace=False).tolist())
        out["auroc_degmatch"] = _auroc(scores[pos_idx], scores[np.unique(matched)])
    out["_ranks"] = ranks.tolist()
    return out


def aggregate(per_disease: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Macro mean over diseases + bootstrap CI (resampling diseases); micro MRR/Hits."""
    keys = [k for k in next(iter(per_disease.values())) if not k.startswith("_") and k != "n_pos"]
    names = list(per_disease)
    rng = np.random.default_rng(seed)
    agg = {"n_diseases": len(names), "n_pos": int(sum(per_disease[n]["n_pos"] for n in names))}
    for k in keys:
        v = np.array([per_disease[n].get(k, np.nan) for n in names], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        boots = [np.mean(rng.choice(v, len(v))) for _ in range(n_boot)] if len(v) > 1 else [v[0]]
        agg[k] = {"mean": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    ranks = np.concatenate([np.asarray(per_disease[n]["_ranks"]) for n in names])
    agg["micro_mrr"] = float(np.mean(1.0 / ranks))
    for k in HITS:
        agg[f"micro_hits@{k}"] = float(np.mean(ranks <= k))
    return agg


def paired_bootstrap(per_a: dict, per_b: dict, key: str = "mrr", n_boot: int = 5000, seed: int = 0):
    """P(mean_a <= mean_b) under disease resampling: small value => A reliably better than B."""
    names = [n for n in per_a if n in per_b]
    a = np.array([per_a[n][key] for n in names]); b = np.array([per_b[n][key] for n in names])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(names), (n_boot, len(names)))
    diff = (a[idx] - b[idx]).mean(1)
    return {"mean_diff": float((a - b).mean()), "p_a_not_better": float(np.mean(diff <= 0)), "n": len(names)}

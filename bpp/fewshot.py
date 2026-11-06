"""
Few-shot evaluation: how much do a few known drugs of an unseen disease help?

For every evaluation disease with at least K_MAX + 1 indications we draw a random order of its
indications (DRAWS draws, seeded by draw and disease). The first K_MAX drugs are the reveal pool;
the remaining ones are the evaluation positives and stay the SAME for every k, so k = 0..K_MAX are
compared on identical targets. For k revealed drugs:

  * BioPathNet++ (no retraining): the k (disease, drug) indication edges are added to the
    message graph for this query only (MessageGraph.build(extra=...)), so the path channel can
    reason through them. Prior features are left unchanged.
  * baselines get the same k drugs:
      disease_knn       zero-shot disease-similarity transfer (ignores the revealed drugs)
      co_indication     item-based collaborative filtering: cosine similarity of drug indication
                        profiles over training diseases, summed over the revealed drugs
      target_similarity Jaccard similarity of protein-target sets, summed over the revealed drugs
      knn_plus_coind    disease_knn + co_indication, each scaled to max 1
Revealed drugs are removed from the ranking (they are known). Ranks are filtered over the other
positives, exactly as in the main evaluation.

  python -m bpp.fewshot --runs <run dir> [<run dir> ...] --data datasets/bpp_primekg --out outputs/bpp/fewshot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from bpp.metrics import filtered_ranks
from bpp.train import Runner, build_parser

K_MAX, DRAWS = 3, 3
METHODS = ["bpp", "bpp_rrf_coind", "disease_knn", "co_indication", "target_similarity", "knn_plus_coind"]


def load_runner(run: Path, ckpt: str, data: str, device: str) -> Runner:
    saved = json.loads((run / "results.json").read_text())["args"]
    args = build_parser().parse_args(["--data", data, "--out", str(run / "_fewshot_tmp"), "--split", saved["split"],
                                      "--device", device, "--seed", str(saved["seed"]),
                                      "--dim", str(saved["dim"]), "--layers", str(saved["layers"]),
                                      "--loss", saved["loss"], "--relations", saved["relations"]]
                                     + (["--prior"] if saved["prior"] else []) + (["--no-paths"] if saved["no_paths"] else []))
    r = Runner(args)
    ck = torch.load(run / ckpt, map_location=r.dev, weights_only=False)
    r.model.load_state_dict(ck["model"])
    r.model.eval()
    return r


def metrics(scores: np.ndarray, revealed: np.ndarray, pos: np.ndarray) -> dict:
    s = scores.astype(np.float64).copy()
    s[revealed] = -np.inf                      # known drugs are not candidates
    ranks = filtered_ranks(s, pos)
    return {"mrr": float(np.mean(1.0 / ranks)), "hits1": float(np.mean(ranks <= 1)),
            "hits10": float(np.mean(ranks <= 10)), "any10": float(np.any(ranks <= 10))}


def rrf(*score_vectors, c: float = 60.0) -> np.ndarray:
    """Reciprocal-rank fusion (Cormack et al. 2009, constant 60): parameter-free late fusion."""
    out = np.zeros(len(score_vectors[0]))
    for s in score_vectors:
        rank = np.empty(len(s)); rank[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        out += 1.0 / (c + rank)
    return out


class BaselineScorer:
    def __init__(self, task):
        self.t = task
        Yc = task.Y.T.tocsr().astype(np.float64)              # drugs x train diseases
        self.Yc = Yc
        self.ynorm = np.sqrt(np.asarray(Yc.multiply(Yc).sum(1)).ravel()) + 1e-12
        T = (task.DT > 0).astype(np.float64).tocsr()            # drugs x proteins
        self.T = T
        self.tcount = np.asarray(T.sum(1)).ravel()

    def scores(self, d: int, revealed: np.ndarray) -> dict:
        t = self.t
        pop = t.popularity(d)
        tie = 1e-3 * pop / (pop.max() + 1e-9)
        knn = t.knn(d)
        out = {"disease_knn": knn + 1e-3 * pop}
        if len(revealed) == 0:
            coind = np.zeros(t.C); tsim = np.zeros(t.C)
        else:
            Yc = self.Yc
            rows = t._excl_rows(d)                              # never use the query family's labels
            if rows:
                mask = np.ones(Yc.shape[1]); mask[rows] = 0
                Yc = Yc @ sp.diags(mask)
            ynorm = np.sqrt(np.asarray(Yc.multiply(Yc).sum(1)).ravel()) + 1e-12
            dots = np.asarray((Yc @ Yc[revealed].T).todense())  # C x k
            coind = (dots / ynorm[:, None] / ynorm[revealed][None, :]).sum(1)
            inter = np.asarray((self.T @ self.T[revealed].T).todense())
            union = self.tcount[:, None] + self.tcount[revealed][None, :] - inter
            tsim = np.where(union > 0, inter / np.maximum(union, 1), 0.0).sum(1)
        out["co_indication"] = coind + tie
        out["target_similarity"] = tsim + tie
        out["knn_plus_coind"] = knn / (knn.max() + 1e-9) + coind / (coind.max() + 1e-9) + tie
        return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--from-records", help="recompute the summary from a saved records file only")
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--set", default="test", help="evaluation set name from RepurposingTask.eval_sets()")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.from_records:
        recs = json.loads(Path(a.from_records).read_text())
        (out / f"fewshot_{a.set}_summary.json").write_text(json.dumps(summarise(recs), indent=1))
        return

    records, base_done = [], False
    for run in map(Path, a.runs):
        r = load_runner(run, a.ckpt, a.data, a.device)
        seed = r.a.seed
        t = r.task
        if not base_done:
            bs_all = BaselineScorer(t)                       # same split for every run: build once
        bs = bs_all if not base_done else None             # record baseline rows only once
        items = [(n, d) for n, d, _ in t.eval_sets()[a.set]]
        ind_rel = r.qrel_ind
        n_used = 0
        for name, d in items:
            pos_all = t.eval_labels(d, include_offlabel=False)
            if len(pos_all) < K_MAX + 1:
                continue
            n_used += 1
            for draw in range(DRAWS):
                order = np.random.default_rng([draw, int(d)]).permutation(pos_all)
                pool, evalpos = order[:K_MAX], order[K_MAX:]
                for k in range(K_MAX + 1):
                    if k == 0 and draw > 0:
                        continue                                  # k = 0 does not depend on the draw ...
                    rev = pool[:k]
                    drugs = torch.as_tensor(t.drugs[rev], dtype=torch.int64)
                    extra = (torch.full_like(drugs, int(d)), drugs, torch.full_like(drugs, ind_rel))
                    graph = r.mg.build(extra=extra)
                    src = torch.as_tensor([int(d)], device=r.dev)
                    qrel = torch.full_like(src, ind_rel)
                    logits = r.model(graph, src, qrel, r.cand, prior=r.priors([int(d)]),
                                     use_paths=not r.a.no_paths)[0].float().cpu().numpy()
                    draws = range(DRAWS) if k == 0 else [draw]   # ... but its eval positives do, so score all draws
                    for dr in draws:
                        o = np.random.default_rng([dr, int(d)]).permutation(pos_all)
                        p_, e_ = o[:K_MAX], o[K_MAX:]
                        rv = p_[:k]
                        records.append({"method": "bpp", "seed": seed, "disease": int(d), "name": name,
                                        "draw": dr, "k": k, "n_eval": int(len(e_)), **metrics(logits, rv, e_)})
                        base = bs_all.scores(int(d), rv)
                        # nothing to fuse without revealed drugs: the fused ranking is the model's own
                        fused = rrf(logits, base["co_indication"]) if k > 0 else logits
                        records.append({"method": "bpp_rrf_coind", "seed": seed, "disease": int(d), "name": name,
                                        "draw": dr, "k": k, "n_eval": int(len(e_)), **metrics(fused, rv, e_)})
                        if bs is not None:
                            for m, sc in base.items():
                                records.append({"method": m, "seed": -1, "disease": int(d), "name": name,
                                                "draw": dr, "k": k, "n_eval": int(len(e_)), **metrics(sc, rv, e_)})
        base_done = True
        print(f"{run.name}: seed {seed}, {n_used} diseases with >= {K_MAX + 1} indications", flush=True)
        del r
        torch.cuda.empty_cache()

    (out / f"fewshot_{a.set}_records.json").write_text(json.dumps(records))
    summ = summarise(records)
    (out / f"fewshot_{a.set}_summary.json").write_text(json.dumps(summ, indent=1))
    for k in range(K_MAX + 1):
        print(f"k={k}: " + "  ".join(f"{m} {summ['mean'][m][str(k)]['mrr']:.3f}" for m in METHODS))


def summarise(records) -> dict:
    """Per disease: mean over draws (and model seeds); then mean over diseases. Paired tests vs. bpp."""
    from collections import defaultdict
    from scipy.stats import wilcoxon

    per = defaultdict(lambda: defaultdict(list))   # (method, k) -> disease -> values
    for rec in records:
        per[(rec["method"], rec["k"])][rec["disease"]].append(rec)
    ks = sorted({rec["k"] for rec in records})
    out = {"k_max": K_MAX, "draws": DRAWS, "n_diseases": len({rec["disease"] for rec in records}),
           "mean": {}, "paired_vs_bpp": {}, "bpp_gain_over_k0": {}}
    disease_means = {}
    for m in METHODS:
        out["mean"][m] = {}
        for k in ks:
            dm = {d: {key: float(np.mean([x[key] for x in v])) for key in ("mrr", "hits1", "hits10", "any10")}
                  for d, v in per[(m, k)].items()}
            disease_means[(m, k)] = dm
            out["mean"][m][str(k)] = {key: float(np.mean([v[key] for v in dm.values()]))
                                      for key in ("mrr", "hits1", "hits10", "any10")}
    def paired(a: dict, b: dict, key="mrr", n_boot=5000):
        names = sorted(n for n in a if n in b)
        diffs = np.array([a[n][key] - b[n][key] for n in names])
        idx = np.random.default_rng(0).integers(0, len(names), (n_boot, len(names)))
        boot = diffs[idx].mean(1)
        pw = float(wilcoxon(diffs).pvalue) if np.any(np.abs(diffs) > 1e-12) else 1.0
        return {"mean_diff": float(diffs.mean()), "ci_lo": float(np.percentile(boot, 2.5)),
                "ci_hi": float(np.percentile(boot, 97.5)), "wilcoxon_p": pw, "n": len(names),
                "wins": int((diffs > 1e-9).sum()), "losses": int((diffs < -1e-9).sum()),
                "significant": bool((np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0) and pw < 0.05)}

    for m in METHODS[1:]:
        out["paired_vs_bpp"][m] = {str(k): paired(disease_means[("bpp", k)], disease_means[(m, k)]) for k in ks}
    for k in ks[1:]:
        out["bpp_gain_over_k0"][str(k)] = paired(disease_means[("bpp", k)], disease_means[("bpp", 0)])
    # the fused ranking against the strongest baseline that also uses the revealed drugs
    out["rrf_vs_knn_plus_coind"] = {str(k): paired(disease_means[("bpp_rrf_coind", k)], disease_means[("knn_plus_coind", k)])
                                    for k in ks[1:]}
    return out


if __name__ == "__main__":
    main()

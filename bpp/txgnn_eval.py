"""
Score BioPathNet++ and TxGNN on the same TxGNN disease-area split, under BOTH ranking protocols.

  python -m bpp.txgnn_eval --kg <converted dir> --ours <npz from bpp.score> --txgnn <npz from run_txgnn.py> --out results.json

Protocols
  txgnn : TxGNN's disease-centric evaluation (txgnn/utils.py, disease_centric_evaluation): for each test disease,
          rank the drugs that have >= 1 indication anywhere in TxGNN's KG, excluding the disease's train/valid
          indications; positives = the disease's test indications. Recall@k = share of positives in the top k;
          AUROC/AUPRC over those candidates. Means over test diseases (as in TxGNN's summary()).
  full  : our protocol, every drug ranked, filtered ranks (bpp/metrics.py).
Both npz files hold scores aligned to the converted KG's drug order (see run_txgnn.py).
Several npz files per model = several training seeds: paired tests then compare per-disease means over the seeds
(as for the main results); 'seeds' gives each summary metric as mean +- sd over seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bpp.analyze import compare, macro
from bpp.kg import CompactKG
from bpp.metrics import disease_metrics

KS = (10, 50, 100)


def txgnn_protocol(scores: np.ndarray, pos_cols: np.ndarray, cand_cols: np.ndarray) -> dict:
    """One disease. scores: all drugs; pos_cols / cand_cols: indices into that vector."""
    cand = np.asarray(sorted(set(cand_cols.tolist()) | set(pos_cols.tolist())))   # positives are always candidates
    s = scores[cand]
    y = np.isin(cand, pos_cols).astype(int)
    order = np.argsort(-s, kind="stable")
    hit = y[order]
    out = {"n_pos": int(y.sum()), "n_candidates": int(len(cand)),
           "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
    for k in KS:
        out[f"recall@{k}"] = float(hit[:k].sum() / y.sum())
        first = np.flatnonzero(hit[:k])
        out[f"mrr@{k}"] = float(1.0 / (first[0] + 1)) if len(first) else 0.0     # TxGNN's MRR@k (first hit)
    return out


def evaluate(kg: CompactKG, proto: dict, npz: Path) -> dict:
    z = np.load(npz)
    drugs = kg.drug_ids()
    col = {int(d): i for i, d in enumerate(drugs)}
    assert np.array_equal(z["drugs"], drugs), "score file is not aligned to this KG's drug order"
    cand_cols = np.array([col[d] for d in proto["candidate_drugs"] if d in col])
    pos = {}
    for d, g in proto["test_pairs"]:
        pos.setdefault(int(d), set()).add(col[int(g)])
    per_tx, per_full = {}, {}
    name = kg.node_names
    for d, sc in zip(z["diseases"].tolist(), z["scores"]):
        if d not in pos:
            continue
        pc = np.array(sorted(pos[d]))
        per_tx[name[d]] = txgnn_protocol(sc, pc, cand_cols)
        per_full[name[d]] = disease_metrics(sc, pc)
    summ = {"n_diseases": len(per_tx)}
    for k in ("auroc", "auprc", *(f"recall@{k}" for k in KS), *(f"mrr@{k}" for k in KS)):
        summ[f"txgnn_{k}"] = macro(per_tx, k)
    for k in ("mrr", "hits@1", "hits@10", "hits@100", "auroc"):
        summ[f"full_{k}"] = macro({d: {k: m[k]} for d, m in per_full.items()}, k)
    return {"summary": summ, "per_disease_txgnn": per_tx,
            "per_disease_full": {d: {k: v for k, v in m.items() if not k.startswith("_")} for d, m in per_full.items()}}


def seed_mean(runs: dict, part: str) -> dict:
    """Per-disease metrics averaged over one model's seeds (identical to the single run when there is one seed)."""
    per = [r[part] for r in runs.values()]
    out = {}
    for d, m in per[0].items():
        if all(d in p for p in per):
            out[d] = {k: float(np.mean([p[d][k] for p in per])) for k, v in m.items()
                      if isinstance(v, (int, float)) and all(k in p[d] for p in per)}
    return out


def seed_summary(runs: dict) -> dict:
    """Each summary metric: mean and sd over seeds of the per-seed macro means."""
    out = {}
    for k in next(iter(runs.values()))["summary"]:
        v = [r["summary"][k]["mean"] for r in runs.values() if isinstance(r["summary"].get(k), dict)]
        if v:
            out[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "seeds": v}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--ours", nargs="*", default=[])
    ap.add_argument("--txgnn", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kg = CompactKG.load(a.kg)
    proto = json.loads((Path(a.kg) / "txgnn_protocol.json").read_text())
    res = {"ours": {Path(p).stem: evaluate(kg, proto, Path(p)) for p in a.ours},
           "txgnn": {Path(p).stem: evaluate(kg, proto, Path(p)) for p in a.txgnn}}
    res["seeds"] = {who: seed_summary(res[who]) for who in ("ours", "txgnn") if res[who]}
    if a.ours and a.txgnn:          # paired, per disease, seed-averaged
        res["n_seeds"] = {"ours": len(res["ours"]), "txgnn": len(res["txgnn"])}
        o, t = seed_mean(res["ours"], "per_disease_txgnn"), seed_mean(res["txgnn"], "per_disease_txgnn")
        res["paired"] = {k: compare(o, t, k) for k in ("recall@100", "auroc", "auprc")}
        of, tf = seed_mean(res["ours"], "per_disease_full"), seed_mean(res["txgnn"], "per_disease_full")
        res["paired_full"] = {k: compare(of, tf, k) for k in ("mrr", "hits@10", "hits@100", "auroc")}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for who, d in (("ours", res["ours"]), ("txgnn", res["txgnn"])):
        for run, r in d.items():
            s = r["summary"]
            print(f"{who:6s} {run:22s} n={s['n_diseases']}  TxGNN-protocol Recall@100 {s['txgnn_recall@100']['mean']:.3f} "
                  f"AUROC {s['txgnn_auroc']['mean']:.3f} AUPRC {s['txgnn_auprc']['mean']:.3f} | full MRR {s['full_mrr']['mean']:.3f}")


if __name__ == "__main__":
    main()

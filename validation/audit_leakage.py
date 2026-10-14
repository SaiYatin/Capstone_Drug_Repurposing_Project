"""
Leakage & shortcut audit for BioPathNet++ (Phase 0)
===================================================

Reproduces the evaluation-validity checks behind the October 2026 audit.
CPU only; needs numpy, pandas, scipy, scikit-learn (no torch).

Checks
  1. Target leakage   : are labelled test/val positives edges of the message-passing graph?
  2. Degree shortcut  : AUROC/AUPRC of zero-parameter degree heuristics on each split
  3. Subgraph reach   : out-degree of each target disease (PyG k_hop_subgraph default
                        flow='source_to_target' only follows incoming edges, so a disease
                        with out-degree 0 never enters a drug's subgraph)
  4. Hard-negative degree baseline per test disease (positives vs drugs indicated for
     other autoimmune diseases), comparable to validation/metrics_v2.py
  5. Label provenance : share of positives that are text-mined GNBR edges
  6. Spurious relation types (HPO frequency strings baked into relation names)

Usage (from project root):
    python validation/audit_leakage.py
    python validation/audit_leakage.py --processed datasets/processed --out outputs/audit
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = {
    "Disease::MESH:D008105": "Primary Biliary Cholangitis",
    "Disease::MESH:D012595": "Systemic Sclerosis",
    "Disease::MESH:D009157": "Myasthenia Gravis",
    "Disease::MESH:D019693": "Autoimmune Hepatitis",
    "Disease::MESH:D003424": "Crohn's Disease",
    "Disease::MESH:D012859": "Sjogren's Syndrome",
}
TRAIN_DISEASES = {
    "Disease::MESH:D001172": "Rheumatoid Arthritis",
    "Disease::MESH:D008180": "Systemic Lupus Erythematosus",
    "Disease::MESH:D009103": "Multiple Sclerosis",
}


def drug_disease(row):
    h, t = str(row["head"]), str(row["tail"])
    return (h, t) if h.startswith("Compound::") else (t, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="datasets/processed")
    ap.add_argument("--out", default="outputs/audit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    P, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    print("Loading graph TSV ...")
    g = pd.read_csv(P / "enriched_drkg.tsv", sep="\t", header=None, names=["h", "r", "t"], dtype=str)
    e2i = json.load(open(P / "enriched_entity2id.json"))
    n = len(e2i)
    hi = g["h"].map(e2i)
    ti = g["t"].map(e2i)
    ok = hi.notna() & ti.notna()
    row = hi[ok].astype(np.int64).values
    col = ti[ok].astype(np.int64).values
    outdeg = np.bincount(row, minlength=n)
    indeg = np.bincount(col, minlength=n)
    deg = outdeg + indeg

    graph_pairs = set(zip(g["h"], g["t"])) | set(zip(g["t"], g["h"]))
    splits = {s: pd.read_csv(P / f"{s}_pairs.csv") for s in ["train", "val", "test"]}

    # 1. target leakage
    print("\n[1] Target leakage (labelled positives present as graph edges)")
    report["leakage"] = {}
    for s, df in splits.items():
        pos = [drug_disease(r) for _, r in df[df["label"] == 1].iterrows()]
        uniq = set(pos)
        inside = sum(1 for p in uniq if p in graph_pairs)
        report["leakage"][s] = {"unique_positive_pairs": len(uniq), "in_graph": inside}
        print(f"    {s:5s}: {inside}/{len(uniq)} unique positive pairs are edges of the message-passing graph")

    # 2. degree shortcut
    print("\n[2] Degree-only heuristics (no learning)")
    report["degree_baselines"] = {}
    for s, df in splits.items():
        dd, sd = [], []
        for _, r in df.iterrows():
            d, x = drug_disease(r)
            dd.append(deg[e2i[d]] if d in e2i else 0)
            sd.append(deg[e2i[x]] if x in e2i else 0)
        dd, sd, y = np.array(dd), np.array(sd), df["label"].values
        res = {}
        for name, sc in [("drug_degree", dd), ("disease_degree", sd),
                         ("log_drug+log_disease", np.log1p(dd) + np.log1p(sd))]:
            res[name] = {"auroc": float(roc_auc_score(y, sc)), "auprc": float(average_precision_score(y, sc))}
            print(f"    {s:5s} {name:22s} AUROC {res[name]['auroc']:.3f}  AUPRC {res[name]['auprc']:.3f}")
        report["degree_baselines"][s] = res

    # 3. subgraph reachability
    print("\n[3] Disease out-degree (0 => never inside a drug's incoming k-hop subgraph)")
    report["disease_degree"] = {}
    for ent, name in {**TRAIN_DISEASES, **TARGETS}.items():
        i = e2i.get(ent)
        od, idg = (int(outdeg[i]), int(indeg[i])) if i is not None else (None, None)
        report["disease_degree"][name] = {"out_degree": od, "in_degree": idg}
        flag = "  <-- unreachable" if od == 0 else ""
        print(f"    {name:30s} out={od:>5} in={idg:>5}{flag}")

    # 4. hard-negative degree baseline per test disease
    print("\n[4] Per-disease drug-degree baseline vs other-autoimmune-drug negatives")
    rng = np.random.default_rng(args.seed)
    te = splits["test"]
    pos = pd.DataFrame([drug_disease(r) for _, r in te[te["label"] == 1].iterrows()], columns=["drug", "disease"])
    report["hardneg_degree_auroc"] = {}
    for dis, grp in pos.groupby("disease"):
        tgt = set(grp["drug"])
        others = np.setdiff1d(pos.loc[pos["disease"] != dis, "drug"].unique(), list(tgt))
        neg = rng.choice(others, min(len(grp), len(others)), replace=False)
        s = np.r_[[deg[e2i[d]] for d in grp["drug"]], [deg[e2i[d]] for d in neg]]
        y = np.r_[np.ones(len(grp)), np.zeros(len(neg))]
        a = float(roc_auc_score(y, s))
        report["hardneg_degree_auroc"][TARGETS.get(dis, dis)] = {"n_pos": int(len(grp)), "auroc": a}
        print(f"    {TARGETS.get(dis, dis):30s} n={len(grp):3d}  AUROC {a:.3f}")

    # 5. label provenance
    print("\n[5] Label provenance")
    report["gnbr_share"] = {}
    for s, df in splits.items():
        p = df[df["label"] == 1]
        share = float(p["relation"].astype(str).str.startswith("GNBR").mean())
        report["gnbr_share"][s] = share
        print(f"    {s:5s}: {share:.1%} of positives are text-mined GNBR edges")

    # 6. spurious relations
    freq = g["r"].astype(str).str.contains("_freq_")
    report["hpo_freq_relations"] = {"relation_types": int(g.loc[freq, "r"].nunique()),
                                    "edges": int(freq.sum()), "total_relation_types": int(g["r"].nunique())}
    print(f"\n[6] {report['hpo_freq_relations']['relation_types']} of "
          f"{report['hpo_freq_relations']['total_relation_types']} relation types are HPO frequency strings "
          f"({report['hpo_freq_relations']['edges']:,} edges)")

    with open(out / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out / 'audit_report.json'}")


if __name__ == "__main__":
    main()

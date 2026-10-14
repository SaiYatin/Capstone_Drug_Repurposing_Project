"""
Build the compact BioPathNet++ dataset from the project's enriched DRKG (secondary dataset,
also used for development/scale tests). Fixes applied relative to v1/v2:
  * every Compound-Disease edge is a *therapeutic* edge (removed for held-out / batch diseases),
    including GNBR text-mined ones, so no label can leak through the message graph
  * labels ("indication") = curated only: DRUGBANK::treats + Hetionet::CtD
  * the 2,325 'HPO::has_phenotype_freq_*' relation strings are collapsed into one relation

Usage:
  python -m bpp.build_drkg --tsv datasets/processed/enriched_drkg.tsv --out datasets/bpp_drkg
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bpp.kg import CompactKG
from bpp.splits import TARGETS_MESH, make_split

CURATED = {"DRUGBANK::treats::Compound:Disease", "Hetionet::CtD::Compound:Disease"}
PALLIATES = {"Hetionet::CpD::Compound:Disease"}
THERAPEUTIC = ["indication", "contraindication", "off-label use", "text_mined"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    g = pd.read_csv(a.tsv, sep="\t", header=None, names=["h", "r", "t"], dtype=str)
    g["r"] = g["r"].str.replace(r"HPO::has_phenotype_freq_[^:]*::", "HPO::has_phenotype::", regex=True)
    ents = pd.unique(pd.concat([g.h, g.t]))
    e2i = {e: i for i, e in enumerate(ents)}
    types_all = [e.split("::")[0] for e in ents]
    types = sorted(set(types_all))
    node_type = np.array([types.index(t) for t in types_all], dtype=np.int16)

    hi, ti = g.h.map(e2i).values, g.t.map(e2i).values
    hty = np.array(types_all, dtype=object)[hi]
    tty = np.array(types_all, dtype=object)[ti]
    cd = ((hty == "Compound") & (tty == "Disease")) | ((hty == "Disease") & (tty == "Compound"))

    # therapeutic pairs
    sub = g[cd]
    drug = np.where(hty[cd] == "Compound", hi[cd], ti[cd])
    dis = np.where(hty[cd] == "Compound", ti[cd], hi[cd])
    trel = np.where(sub.r.isin(CURATED), 0, np.where(sub.r.isin(PALLIATES), 2, 3))
    th = np.unique(np.stack([dis, drug, trel], 1), axis=0)

    # context edges
    ctx = g[~cd]
    rel_names = sorted(ctx.r.unique())
    r2i = {r: i for i, r in enumerate(rel_names)}
    h, t, r = ctx.h.map(e2i).values, ctx.t.map(e2i).values, ctx.r.map(r2i).values
    e = np.unique(np.stack([h, r, t], 1), axis=0)

    meta = {"source": "DRKG-enriched", "node_types": types, "drug_type": "Compound",
            "disease_type": "Disease", "relations": rel_names, "therapeutic_relations": THERAPEUTIC}
    kg = CompactKG(Path(a.out), meta, node_type, list(ents), e[:, 0], e[:, 1], e[:, 2],
                   th[:, 0], th[:, 1], th[:, 2])

    families = {n: {e2i[m]} for n, m in TARGETS_MESH.items() if m in e2i}
    primary = {n: next(iter(f)) for n, f in families.items()}
    pos = kg.positives("indication")
    s = make_split("targets", families, primary, kg.disease_ids().tolist(), pos, 0.05, a.seed)
    kg.splits = s
    kg.save()
    print(f"nodes={kg.num_nodes:,} ctx_edges={len(e):,} relations={len(rel_names)} therapeutic={len(th):,}")
    print({n: int((th[:, 2] == i).sum()) for i, n in enumerate(THERAPEUTIC)})
    print(f"train={len(s['train_diseases'])} valid={len(s['valid_diseases'])} held_out={len(s['held_out_diseases'])}")
    for n, f in families.items():
        d = next(iter(f)); print(f"  {n:28s} curated indications: {len(pos.get(d, ()))}")


if __name__ == "__main__":
    main()

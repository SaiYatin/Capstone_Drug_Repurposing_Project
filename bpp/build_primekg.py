"""
Build the compact BioPathNet++ dataset from PrimeKG (Chandak et al., Sci Data 2023).

Input : kg.csv from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
        + TxGNN disease-area files (HumanDO.obo, mondo_references.csv,
          kg_grouped_diseases_bert_map.csv) in --txgnn-dir, from github.com/mims-harvard/TxGNN
          (txgnn/data_splits). They define the autoimmune area exactly as TxGNN does (DOID:417).
Output: --out directory in bpp.kg.CompactKG format, one sub-split per protocol:
          splits.json         protocol "targets": the 6 target families held out (zero-shot);
                              every other disease, incl. RA/SLE/MS, can be used for training
          splits_noauto.json  protocol "no_autoimmune": the WHOLE TxGNN autoimmune area (DOID:417)
                              held out. Evaluated on the 6 targets (E2 transfer test: does
                              autoimmune training data help?) and on the full area (comparable
                              to TxGNN's autoimmune disease-area split).

Memory: reads kg.csv in chunks (~1.5 GB peak), fine on a laptop, Kaggle CPU or Colab.

Usage:
  python -m bpp.build_primekg --kg datasets/raw/primekg/kg.csv \
         --txgnn-dir datasets/raw/txgnn_splits --out datasets/bpp_primekg
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bpp.kg import CompactKG
from bpp.splits import RICH_MONDO, TARGETS_MONDO, make_split

THERAPEUTIC = ["indication", "contraindication", "off-label use"]
USECOLS = ["relation", "x_index", "x_id", "x_type", "x_name", "x_source",
           "y_index", "y_id", "y_type", "y_name", "y_source"]


# --------------------------------------------------------------------------------------
def read_kg(path, chunksize=1_000_000):
    nodes = {}                                        # index -> (type, name, id, source)
    ctx = {}                                          # relation -> list of (h, t) arrays
    ther = []
    rel_orient = {}
    for ch in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize, dtype=str, low_memory=False):
        for side in ("x", "y"):
            sub = ch[[f"{side}_index", f"{side}_type", f"{side}_name", f"{side}_id", f"{side}_source"]]
            sub = sub.drop_duplicates(f"{side}_index")
            for i, ty, nm, nid, src in sub.itertuples(index=False):
                nodes.setdefault(int(i), (ty, nm, nid, src))
        xi = ch["x_index"].astype(np.int64).values
        yi = ch["y_index"].astype(np.int64).values
        rel = ch["relation"].values
        xt, yt = ch["x_type"].values, ch["y_type"].values
        for r in np.unique(rel):
            m = rel == r
            if r in THERAPEUTIC:
                drug_is_x = xt[m] == "drug"
                drug = np.where(drug_is_x, xi[m], yi[m])
                dis = np.where(drug_is_x, yi[m], xi[m])
                ther.append(np.stack([dis, drug, np.full(m.sum(), THERAPEUTIC.index(r))], 1))
                continue
            # canonical orientation: heterogeneous -> type order fixed at first sight;
            # homogeneous -> (min, max). Both directions are listed in kg.csv, so dedup later.
            if r not in rel_orient:
                rel_orient[r] = (xt[m][0], yt[m][0])
            a, b = rel_orient[r]
            hx, tx = xi[m], yi[m]
            if a == b:
                h, t = np.minimum(hx, tx), np.maximum(hx, tx)
            else:
                fwd = xt[m] == a
                h, t = np.where(fwd, hx, tx), np.where(fwd, tx, hx)
            ctx.setdefault(r, []).append(np.stack([h, t], 1))
        print(f"  read {len(nodes):,} nodes so far", flush=True)
    return nodes, ctx, ther, rel_orient


def load_do_descendants(obo_path, root="DOID:417"):
    """All descendants (is_a) of a DOID term, including itself; returns ids without prefix."""
    children, cur = {}, None
    for line in Path(obo_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("[Term]"):
            cur = None
        elif line.startswith("id: DOID:"):
            cur = line[4:].strip()
        elif line.startswith("is_a: DOID:") and cur:
            parent = line.split()[1]
            children.setdefault(parent, set()).add(cur)
    out, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in out:
                out.add(c); stack.append(c)
    return {x.split(":")[1] for x in out}


def area_nodes(node_df, txgnn_dir, doid="417"):
    """Exact re-implementation of TxGNN DataSplitter.get_nodes_for_doid."""
    d = Path(txgnn_dir)
    doids = load_do_descendants(d / "HumanDO.obo", f"DOID:{doid}")
    xref = pd.read_csv(d / "mondo_references.csv", dtype=str)
    grp = pd.read_csv(d / "kg_grouped_diseases_bert_map.csv", dtype=str)
    mondo = set(xref[(xref.ontology == "DOID") & (xref.ontology_id.isin(doids))].mondo_id.str.lstrip("0"))
    idx1 = node_df[(node_df.source == "MONDO") & (node_df.nid.str.lstrip("0").isin(mondo))].index
    mgrp = set(grp[(grp.node_source == "MONDO") & (grp.node_id.isin(mondo))].group_id_bert)
    idx2 = node_df[(node_df.source == "MONDO_grouped") & (node_df.nid.isin(mgrp))].index
    return set(idx1.tolist()) | set(idx2.tolist())


def target_family(node_df, grp, mondo_id):
    """Primary node + every PrimeKG disease node in the same TxGNN/BERT disease group
    (catches subtypes such as 'primary biliary cholangitis 1..5')."""
    dis = node_df[node_df.type == "disease"]
    ids = dis.nid.astype(str)
    primary = dis[(dis.source == "MONDO") & (ids.str.lstrip("0") == mondo_id)].index.tolist()
    grouped = dis[(dis.source == "MONDO_grouped") & ids.apply(lambda s: mondo_id in s.split("_"))].index.tolist()
    fam = set(primary) | set(grouped)
    g = grp[grp.node_id == mondo_id]
    if len(g):
        gid = g.group_id_bert.iloc[0]
        members = set(grp[grp.group_id_bert == gid].node_id)
        fam |= set(dis[(dis.source == "MONDO") & ids.str.lstrip("0").isin(members)].index)
        fam |= set(dis[(dis.source == "MONDO_grouped") & (ids == gid)].index)
    prim = (primary or grouped or [None])[0]
    return prim, fam


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--txgnn-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    a = ap.parse_args()

    print("Reading kg.csv in chunks ...")
    nodes, ctx, ther, orient = read_kg(a.kg)
    V = max(nodes) + 1
    assert len(nodes) == V, "PrimeKG node indices are expected to be contiguous"
    node_df = pd.DataFrame([nodes[i] for i in range(V)], columns=["type", "name", "nid", "source"])
    types = sorted(node_df.type.unique())
    node_type = node_df.type.map({t: i for i, t in enumerate(types)}).values.astype(np.int16)

    rel_names = sorted(ctx)
    H, R, T = [], [], []
    for k, r in enumerate(rel_names):
        e = np.unique(np.concatenate(ctx[r]), axis=0)
        H.append(e[:, 0]); T.append(e[:, 1]); R.append(np.full(len(e), k))
        print(f"  {r:32s} {len(e):>10,} edges")
    th = np.unique(np.concatenate(ther), axis=0)
    print(f"  therapeutic pairs: {len(th):,} "
          + str({n: int((th[:, 2] == i).sum()) for i, n in enumerate(THERAPEUTIC)}))

    meta = {"source": "PrimeKG", "node_types": types, "drug_type": "drug", "disease_type": "disease",
            "relations": rel_names, "relation_orientation": orient, "therapeutic_relations": THERAPEUTIC}
    names = [f"{nm} [{src}:{nid}]" for nm, nid, src in zip(node_df.name, node_df.nid, node_df.source)]
    kg = CompactKG(Path(a.out), meta, node_type, names, np.concatenate(H), np.concatenate(R),
                   np.concatenate(T), th[:, 0], th[:, 1], th[:, 2])

    # ---------------- splits ----------------
    grp = pd.read_csv(Path(a.txgnn_dir) / "kg_grouped_diseases_bert_map.csv", dtype=str)
    # disease-disease neighbours: near-duplicates/subtypes ("Crohn's colitis", "adult-onset
    # myasthenia gravis", "scleroderma (disease)") carry the SAME drugs, so they are held out
    # together with the target (strict family). Verified: without this, 16/16 Crohn positives
    # stayed reachable through "Crohn's colitis".
    dd = rel_names.index("disease_disease")
    nb = {}
    for h_, t_ in np.unique(np.concatenate(ctx["disease_disease"]), axis=0):
        nb.setdefault(int(h_), set()).add(int(t_)); nb.setdefault(int(t_), set()).add(int(h_))

    def strict(fam):
        out = set(fam)
        for x in fam:
            out |= nb.get(x, set())
        return out

    families, primary = {}, {}
    for name, mid in TARGETS_MONDO.items():
        p, fam = target_family(node_df, grp, mid)
        if p is None:
            print(f"  WARNING: {name} (MONDO:{mid}) not found in PrimeKG"); continue
        families[name], primary[name] = strict(fam), p
        print(f"  target {name:28s} primary={names[p]}  strict family={len(families[name])} nodes")
    rich = {}
    for name, mid in RICH_MONDO.items():
        p, fam = target_family(node_df, grp, mid)
        rich[name] = sorted(strict(fam))
        print(f"  rich   {name:28s} strict family={len(rich[name])} nodes")
    area = area_nodes(node_df, a.txgnn_dir)
    print(f"  autoimmune area (TxGNN DOID:417): {len(area)} disease nodes")
    autoimmune = set(area) | set().union(*families.values()) | set().union(*map(set, rich.values()))
    print(f"  autoimmune set for E2 (area + 9 strict families): {len(autoimmune)} nodes")

    pos = kg.positives("indication")
    dis_ids = kg.disease_ids().tolist()
    s_main = make_split("targets", families, primary, dis_ids, pos, a.valid_frac, a.seed, area=area, neighbours=nb)
    s_noauto = make_split("no_autoimmune", families, primary, dis_ids, pos,
                          a.valid_frac, a.seed, area=area, exclude_from_train=autoimmune,
                          neighbours=nb)
    for s in (s_main, s_noauto):
        s["rich_families"] = rich
    kg.splits = s_main
    kg.save()
    (kg.root / "splits_noauto.json").write_text(json.dumps(s_noauto, indent=2))
    for s in (s_main, s_noauto):
        print(f"  [{s['protocol']}] train={len(s['train_diseases'])} valid={len(s['valid_diseases'])} "
              f"held_out={len(s['held_out_diseases'])}")
    print(f"Saved to {kg.root}")


if __name__ == "__main__":
    main()

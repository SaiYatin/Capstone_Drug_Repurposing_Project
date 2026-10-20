"""
Convert a TxGNN disease-area split into BioPathNet++'s CompactKG, so both models are trained and
evaluated on exactly the same information (head-to-head experiment, kaggle/run_txgnn.py).

  python -m bpp.build_txgnn --export <dir with kg_directed.csv, train.csv, test_processed.csv> --out <dir>

What each model sees
  message graph : TxGNN's training graph (train.csv), forward relations only -- our MessageGraph adds the
                  inverses itself, TxGNN's 'rev_' rows are dropped and homogeneous relations deduplicated
  drug labels   : therapeutic edges of train.csv. TxGNN's random 12.5% validation edges (valid.csv) are
                  used by neither model's message graph; we drop them, and carve our own validation
                  DISEASES (5%, with families) out of the training diseases for early stopping, so our
                  model trains on slightly less data than TxGNN
  test          : the area diseases' indications from TxGNN's processed test set (test_processed.csv)
Node names keep TxGNN's ids, "<name> [<type>:<id>]", so predictions of both models can be aligned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bpp.kg import CompactKG
from bpp.splits import make_split

THER = ["indication", "contraindication", "off-label use"]


def convert2str(x) -> str:
    """Exactly TxGNN's id normalisation (txgnn/utils.py): numeric ids become float strings ('1234' -> '1234.0'),
    grouped ids ('12_34') stay as they are. Needed to align TxGNN's predictions with our node ids."""
    try:
        if "_" not in str(x):
            x = float(x)
    except (TypeError, ValueError):
        pass
    return str(x)


def _norm(series: pd.Series) -> pd.Series:
    uniq = pd.unique(series)
    m = {u: convert2str(u) for u in uniq}
    return series.map(m)


def _nodes(df: pd.DataFrame):
    """Global node ids: per-type offset + TxGNN's per-type contiguous index."""
    x = df[["x_type", "x_idx", "x_id"]].set_axis(["type", "idx", "id"], axis=1)
    y = df[["y_type", "y_idx", "y_id"]].set_axis(["type", "idx", "id"], axis=1)
    n = pd.concat([x, y]).drop_duplicates(["type", "idx"])
    n["idx"] = n["idx"].astype(int)
    types = sorted(n["type"].unique())
    size = n.groupby("type")["idx"].max().to_dict()
    offset, o = {}, 0
    for t in types:
        offset[t] = o
        o += int(size[t]) + 1
    n["gid"] = n["type"].map(offset) + n["idx"]
    return types, offset, o, n


def gid(frame: pd.DataFrame, side: str, offset: dict) -> np.ndarray:
    return (frame[f"{side}_type"].map(offset) + frame[f"{side}_idx"].astype(int)).to_numpy()


def build(export: Path, out: Path, names_csv: Path | None = None, valid_frac: float = 0.05, seed: int = 0) -> dict:
    df = pd.read_csv(export / "kg_directed.csv", low_memory=False)
    tr = pd.read_csv(export / "train.csv", low_memory=False)
    te = pd.read_csv(export / "test_processed.csv", low_memory=False)
    for f in (df, tr, te):
        f["x_id"] = _norm(f["x_id"]); f["y_id"] = _norm(f["y_id"])

    types, offset, n_nodes, nodes = _nodes(df)
    node_type = np.zeros(n_nodes, dtype=np.int16)
    node_type[nodes["gid"].to_numpy()] = nodes["type"].map({t: i for i, t in enumerate(types)}).to_numpy()
    label = {}
    if names_csv is not None and names_csv.exists():
        kg = pd.read_csv(names_csv, usecols=["x_type", "x_id", "x_name", "y_type", "y_id", "y_name"], low_memory=False)
        for s in ("x", "y"):
            sub = kg[[f"{s}_type", f"{s}_id", f"{s}_name"]].drop_duplicates([f"{s}_type", f"{s}_id"])
            label.update({(t, convert2str(i)): nm for t, i, nm in sub.itertuples(index=False)})
    names = [""] * n_nodes
    for t, i, g in nodes[["type", "id", "gid"]].itertuples(index=False):
        names[int(g)] = f"{label.get((t, str(i)), str(i))} [{t}:{i}]"

    # message-graph (context) edges = TxGNN's training graph, forward rows only
    ctx = tr[~tr.relation.str.startswith("rev_") & ~tr.relation.isin(THER)].copy()
    ctx["h"], ctx["t"] = gid(ctx, "x", offset), gid(ctx, "y", offset)
    homo = ctx.x_type == ctx.y_type                      # undirected within-type: keep one direction
    a, b = np.minimum(ctx.h, ctx.t), np.maximum(ctx.h, ctx.t)
    ctx.loc[homo, "h"], ctx.loc[homo, "t"] = a[homo], b[homo]
    ctx = ctx.drop_duplicates(["h", "relation", "t"])
    relations = sorted(ctx.relation.unique())
    rel_id = {r: i for i, r in enumerate(relations)}

    # therapeutic edges: training graph + test labels (TxGNN's validation edges are used by neither model)
    ther = pd.concat([tr[tr.relation.isin(THER)].assign(part="train"), te[te.relation.isin(THER)].assign(part="test")])
    ther = ther[(ther.x_type == "drug") & (ther.y_type == "disease")]
    ther = ther.assign(drug=gid(ther, "x", offset), disease=gid(ther, "y", offset)).drop_duplicates(["drug", "disease", "relation"])

    meta = {"source": "TxGNN disease-area split converted by bpp/build_txgnn.py", "node_types": types,
            "drug_type": "drug", "disease_type": "disease", "relations": relations,
            "therapeutic_relations": THER}
    kg_out = CompactKG(out, meta, node_type, names,
                       ctx.h.to_numpy(), ctx.relation.map(rel_id).to_numpy(), ctx.t.to_numpy(),
                       ther.disease.to_numpy(), ther.drug.to_numpy(), ther.relation.map({r: i for i, r in enumerate(THER)}).to_numpy())

    # evaluation diseases = TxGNN's test diseases (rev_indication rows kept by process_disease_area_split)
    ev = te[te.relation == "rev_indication"]
    eval_dis = sorted(set(gid(ev, "x", offset).tolist()))
    area = sorted(set(ther[ther.part == "test"].disease.tolist()))          # every disease with a test edge
    test_pairs = {(int(d), int(g)) for d, g in zip(gid(ev, "x", offset), gid(ev, "y", offset))}

    # training-graph indications only decide which diseases are trainable / validation candidates
    pos_train = {}
    for d, g, r in ther[ther.part == "train"][["disease", "drug", "relation"]].itertuples(index=False):
        if r == "indication":
            pos_train.setdefault(int(d), set()).add(int(g))
    dd = ctx[ctx.relation == "disease_disease"]
    nb = {}
    for h, t in zip(dd.h.tolist(), dd.t.tolist()):
        nb.setdefault(h, set()).add(t); nb.setdefault(t, set()).add(h)
    split = make_split("txgnn_area", {}, {}, kg_out.disease_ids().tolist(), pos_train, valid_frac, seed,
                       extra_held_out=area, area=area, neighbours=nb)
    split["eval_area"] = eval_dis
    split["targets"], split["target_primary"] = {}, {}
    # leakage checks: no test pair is a training-graph edge; no test disease is trained on
    train_pairs = {(int(d), int(g)) for d, g in ther[ther.part == "train"][["disease", "drug"]].itertuples(index=False)}
    assert not (test_pairs & train_pairs), f"{len(test_pairs & train_pairs)} test pairs are training edges"
    assert not (set(eval_dis) & set(split["train_diseases"])), "test diseases among training diseases"
    kg_out.splits = split
    kg_out.save()
    (out / "splits.json").write_text(json.dumps(split, indent=2))

    # TxGNN's own ranking protocol (bpp/txgnn_eval.py): candidates = drugs with >=1 indication anywhere in
    # TxGNN's full KG (kg_directed, all splits); exclude a disease's train/valid indications (none here)
    ind_all = df[df.relation == "indication"]
    cand = sorted(set(gid(ind_all, "x", offset).tolist()))
    (out / "txgnn_protocol.json").write_text(json.dumps({"candidate_drugs": cand, "test_pairs": sorted(test_pairs),
                                                         "eval_diseases": eval_dis}))
    return {"nodes": n_nodes, "context_edges": len(ctx), "relations": len(relations), "therapeutic": len(ther),
            "train_diseases": len(split["train_diseases"]), "valid_diseases": len(split["valid_diseases"]),
            "eval_diseases": len(eval_dis), "test_pairs": len(test_pairs), "candidate_drugs": len(cand)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None, help="TxGNN kg.csv, for readable node names")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(build(Path(a.export), Path(a.out), Path(a.names) if a.names else None, seed=a.seed), indent=1))


if __name__ == "__main__":
    main()

"""Unit tests for BioPathNet++ v3. Run:  python -m pytest -q tests/test_bpp.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpp.kg import CompactKG, MessageGraph, _csr          # noqa: E402
from bpp.model import BioPathNetPP, RelSpMM, listwise_loss  # noqa: E402


def toy_kg(tmp_path=None, seed=0):
    """6 diseases, 8 drugs, 10 genes. Types: 0 disease, 1 drug, 2 gene."""
    rng = np.random.default_rng(seed)
    nd, ng, nge = 6, 8, 10
    node_type = np.array([0] * nd + [1] * ng + [2] * nge, dtype=np.int16)
    names = [f"dis{i}" for i in range(nd)] + [f"drug{i}" for i in range(ng)] + [f"gene{i}" for i in range(nge)]
    dis, drugs, genes = np.arange(nd), np.arange(nd, nd + ng), np.arange(nd + ng, nd + ng + nge)
    h, r, t = [], [], []
    for d in dis:                       # disease-gene (rel 0)
        for g in rng.choice(genes, 3, replace=False):
            h.append(d); r.append(0); t.append(g)
    for g in drugs:                     # drug-gene (rel 1)
        for x in rng.choice(genes, 2, replace=False):
            h.append(g); r.append(1); t.append(x)
    td, tg, tr = [], [], []
    for d in dis:                       # indications (trel 0)
        for g in rng.choice(drugs, 2, replace=False):
            td.append(d); tg.append(g); tr.append(0)
    meta = {"node_types": ["disease", "drug", "gene"], "drug_type": "drug", "disease_type": "disease",
            "relations": ["disease_gene", "drug_gene"], "therapeutic_relations": ["indication", "contraindication"]}
    return CompactKG(Path(tmp_path or "."), meta, node_type, names, np.array(h), np.array(r), np.array(t),
                     np.array(td), np.array(tg), np.array(tr))


def test_relspmm_matches_dense_and_gradcheck():
    torch.manual_seed(0)
    V, B, d, R = 7, 2, 3, 3
    mats, mats_t, dense = [], [], []
    for _ in range(R):
        m = (torch.rand(V, V) < 0.3).double()
        idx = m.nonzero().t()
        mats.append(_csr(idx[0], idx[1], torch.ones(idx.shape[1], dtype=torch.float64), V))
        mats_t.append(_csr(idx[1], idx[0], torch.ones(idx.shape[1], dtype=torch.float64), V))
        dense.append(m)
    x = torch.randn(V, B, d, dtype=torch.float64, requires_grad=True)
    w = torch.randn(B, R, d, dtype=torch.float64, requires_grad=True)
    out = RelSpMM.apply(x, w, mats, mats_t)
    ref = sum((dense[r] @ x.reshape(V, B * d)).view(V, B, d) * w[:, r].unsqueeze(0) for r in range(R))
    assert torch.allclose(out, ref)
    assert torch.autograd.gradcheck(lambda a, b: RelSpMM.apply(a, b, mats, mats_t), (x, w))


def test_batch_diseases_lose_their_drug_edges():
    kg = toy_kg()
    allowed = [0, 1, 2, 3]                      # diseases 4,5 = held-out family
    g = MessageGraph(kg, allowed)
    mats, _, _ = g.build(drop_diseases=torch.tensor([0]))
    nctx, nther = g.num_ctx, g.num_ther
    ther_fwd = mats[nctx + 0].to_dense()        # indication drug->disease, row = disease
    ther_inv = mats[2 * nctx + nther + 0].to_dense()
    assert ther_fwd[0].sum() == 0 and ther_inv[:, 0].sum() == 0      # batch disease removed
    for held in (4, 5):
        assert ther_fwd[held].sum() == 0 and ther_inv[:, held].sum() == 0  # never in graph
    assert ther_fwd[1].sum() > 0                # other train diseases keep their edges


def test_inverse_edges_reach_diseases_without_out_edges():
    """v1/v2 bug: diseases with only incoming edges were unreachable. Here every
    disease must receive messages through inverse relations."""
    kg = toy_kg()
    g = MessageGraph(kg, list(range(6)))
    mats, _, deg = g.build()
    incoming = sum(m.to_dense() for m in mats).sum(1)
    assert (incoming[:6] > 0).all()


def test_model_forward_and_learns_toy_task():
    torch.manual_seed(0)
    kg = toy_kg()
    g = MessageGraph(kg, list(range(6)))
    model = BioPathNetPP(g.num_relations, num_query_relations=2, dim=8, num_layers=3, mlp_hidden=16)
    cand = torch.as_tensor(kg.drug_ids())
    pos = kg.positives("indication")
    src = torch.tensor([0, 1, 2, 3]); q = torch.zeros(4, dtype=torch.long)
    mask = torch.zeros(4, cand.numel(), dtype=torch.bool)
    for i, s in enumerate(src.tolist()):
        for drug in pos[s]:
            mask[i, (cand == drug).nonzero()[0, 0]] = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        graph = g.build(drop_diseases=src)
        logits = model(graph, src, q, cand)
        assert logits.shape == (4, cand.numel())
        loss = listwise_loss(logits, mask)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.1, (losses[0], losses[-1])


DATA = Path(__file__).resolve().parents[1] / "datasets" / "bpp_primekg"


def test_extra_splits_never_train_on_evaluation_families():
    """Transfer / test splits (bpp/extra_splits.py): no evaluation disease, and no disease-disease
    neighbour of one, may be a training disease (its drug edges would enter the message graph)."""
    import json
    import pytest
    if not (DATA / "splits_tr_A.json").exists():
        pytest.skip("run `python -m bpp.extra_splits` first")
    from bpp.task import RepurposingTask
    kg = CompactKG.load(str(DATA))
    base = json.loads((DATA / "splits.json").read_text())
    auto = RepurposingTask(kg, base).autoimmune
    a_ind = None
    for name in ["splits_tr_A.json", "splits_tr_CE.json", "splits_test.json"] + \
                [f"splits_tr_Actl{j}.json" for j in range(3)]:
        s = json.loads((DATA / name).read_text())
        task = RepurposingTask(kg, s)
        train = set(task.train)
        evaluated = set(s["valid_diseases"]) | set(s.get("eval_area", [])) | set(s.get("test_diseases", []))
        # strict family = disease + direct disease-disease neighbours (same rule as splits.json);
        # target families are stored already expanded
        forbidden = set().union(*map(set, s["targets"].values()))
        for d in evaluated:
            forbidden |= task.exclusion(d)
        assert not (train & forbidden), (name, len(train & forbidden))
        if "Actl" in name:
            assert not any(task.exclusion(d) & auto for d in train), name      # controls are non-autoimmune
            n_ind = sum(len(task.ind[d]) for d in train)
            assert abs(n_ind - a_ind) <= 0.1 * a_ind, (name, n_ind, a_ind)   # matched label budget
        if name == "splits_tr_A.json":
            a_ind = sum(len(task.ind[d]) for d in train)
            assert train <= set().union(*map(set, s["rich_families"].values()))


def test_explain_replica_matches_model_and_paths_exist():
    """bpp/explain.py must reproduce the model's logits exactly and only return paths made of real edges."""
    from bpp.explain import best_paths, edge_importance, forward_with_states
    torch.manual_seed(0)
    kg = toy_kg()
    g = MessageGraph(kg, list(range(6)))
    model = BioPathNetPP(g.num_relations, num_query_relations=2, dim=8, num_layers=3, mlp_hidden=16).eval()
    cand = torch.as_tensor(kg.drug_ids())
    graph = g.build()
    path_logits, _, xs, ss = forward_with_states(model, graph, 0, 0, cand)
    ref = model(graph, torch.tensor([0]), torch.tensor([0]), cand)[0]
    assert torch.allclose(ref, path_logits, atol=1e-5)
    path_logits[0].backward()
    edges = edge_importance(model, graph, xs, ss, 0, top_k=500)
    real = set()
    for r, a in enumerate(graph[0]):
        d = a.to_dense()
        for v, u in zip(*torch.nonzero(d, as_tuple=True)):
            real.add((int(u), r, int(v)))
    assert edges and all((u, r, v) in real for _, r, u, v, _ in edges)
    for _, nodes, rels in best_paths(edges, 0, int(cand[0]), max_len=4):
        assert nodes[0] == 0 and nodes[-1] == int(cand[0])
        assert all((a, r, b) in real for a, r, b in zip(nodes[:-1], rels, nodes[1:]))


def test_validation_helpers():
    from bpp.validate import bh, grade
    q = bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)
    assert grade([]) == "D"
    assert grade([{"phase": "PHASE2/PHASE3"}]) == "A"
    assert grade([{"phase": "PHASE2"}]) == "B"
    assert grade([{"phase": "NA"}]) == "C"


# toy disease-area split: diseases 16-19 are the area; 2 and 5 are NON-area diseases that also lost one drug edge
# each to TxGNN's random test edges (2: indication, 5: contraindication); 15 is a disease_disease neighbour of 16
TOY_AREA, TOY_RANDOM_TEST = (16, 17, 18, 19), {2: "indication", 5: "contraindication"}


def _toy_txgnn_export(out: Path):
    """Tiny export in TxGNN's file format: kg_directed.csv, train.csv / test_processed.csv with 'rev_' rows.
    Like TxGNN's process_disease_area_split, reverse therapeutic test rows are kept only for area diseases."""
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = []

    def add(xt, xi, rel, yt, yi, split="train"):
        rows.append(dict(x_type=xt, x_id=f"{xi}" if xt == "disease" else f"DB{xi:05d}", relation=rel, y_type=yt,
                         y_id=f"{yi}" if yt == "disease" else (f"DB{yi:05d}" if yt == "drug" else f"g{yi}"),
                         split=split, x_idx=xi, y_idx=yi))
    for g in range(30):
        for p in rng.choice(40, 3, replace=False):
            add("drug", g, "drug_protein", "gene/protein", int(p))
    for d in range(20):
        for p in rng.choice(40, 4, replace=False):
            add("disease", d, "disease_protein", "gene/protein", int(p))
    for d in range(0, 19, 3):
        add("disease", d, "disease_disease", "disease", d + 1)
    for d in range(20):
        drugs = rng.choice(30, 3, replace=False)
        for g in drugs:
            add("drug", int(g), "indication", "disease", d, "test" if d in TOY_AREA else "train")
        if d in TOY_RANDOM_TEST:
            add("drug", int(min(set(range(30)) - set(drugs.tolist()))), TOY_RANDOM_TEST[d], "disease", d, "test")
    df = pd.DataFrame(rows).drop_duplicates(["x_type", "x_idx", "relation", "y_type", "y_idx"])
    df.to_csv(out / "kg_directed.csv", index=False)

    def with_rev(f):
        rev = f.rename(columns={"x_type": "y_type", "x_id": "y_id", "x_idx": "y_idx",
                                "y_type": "x_type", "y_id": "x_id", "y_idx": "x_idx"})
        het = rev.x_type != rev.y_type
        rev.loc[het, "relation"] = "rev_" + rev.loc[het, "relation"]
        return pd.concat([f, rev])
    with_rev(df[df.split == "train"]).to_csv(out / "train.csv", index=False)
    te = with_rev(df[df.split == "test"])
    rev_ther = te.relation.isin(["rev_indication", "rev_contraindication", "rev_off-label use"])
    te[~rev_ther | te.x_idx.isin(TOY_AREA)].to_csv(out / "test_processed.csv", index=False)


def test_txgnn_conversion_and_protocol(tmp_path):
    import json
    from bpp.build_txgnn import build, convert2str
    from bpp.txgnn_eval import txgnn_protocol
    assert [convert2str(x) for x in ("1234", "12_34", "DB00001", "0005388")] == ["1234.0", "12_34", "DB00001", "5388.0"]
    _toy_txgnn_export(tmp_path)
    info = build(tmp_path, tmp_path / "kg")          # asserts no test pair / test disease leaks into training
    kg = CompactKG.load(tmp_path / "kg")
    proto = json.loads((tmp_path / "kg" / "txgnn_protocol.json").read_text())
    assert info["eval_diseases"] == 4 and len(proto["eval_diseases"]) == 4
    split = json.loads((tmp_path / "kg" / "splits.json").read_text())
    assert not set(proto["eval_diseases"]) & (set(split["train_diseases"]) | set(split["valid_diseases"]))
    # the message graph never contains a test disease's drug edges
    mg = MessageGraph(kg, sorted(split["train_diseases"]), device="cpu")
    assert mg.num_ctx_edges > 0
    # protocol: a perfect scorer gets recall/AUROC/AUPRC/MRR 1, a reversed one AUROC 0
    s = np.arange(10.0)
    m = txgnn_protocol(s, np.array([9, 8]), np.arange(10))
    assert m["auroc"] == 1 and m["auprc"] == 1 and m["recall@10"] == 1 and m["mrr@10"] == 1
    assert txgnn_protocol(-s, np.array([9, 8]), np.arange(10))["auroc"] == 0
    # strict coverage (default): the non-area diseases with a random test edge are held out too
    assert set(TOY_RANDOM_TEST) <= set(split["held_out_diseases"]) and split["coverage"] == "strict"
    assert info["dropped_test_edges"] == 0


def test_txgnn_full_coverage(tmp_path):
    """--coverage full: non-area diseases with random test edges are trained on, minus exactly those edges."""
    import json
    from bpp.build_txgnn import build
    from bpp.task import RepurposingTask
    _toy_txgnn_export(tmp_path)
    info = build(tmp_path, tmp_path / "kg", coverage="full")      # its own leakage asserts run here
    kg = CompactKG.load(tmp_path / "kg")
    proto = json.loads((tmp_path / "kg" / "txgnn_protocol.json").read_text())
    split = json.loads((tmp_path / "kg" / "splits.json").read_text())
    dis = {int(n.split("[disease:")[1].rstrip("]").split(".")[0]): i
           for i, n in enumerate(kg.node_names) if "[disease:" in n}
    drug = {int(n.split("[drug:DB")[1].rstrip("]")): i for i, n in enumerate(kg.node_names) if "[drug:" in n}
    area = sorted(dis[d] for d in TOY_AREA)
    assert split["coverage"] == "full" and split["area_diseases"] == area and proto["eval_diseases"] == area
    # area diseases AND their families are held out (15 is 16's neighbour), nothing else is
    seen = set(split["train_diseases"]) | set(split["valid_diseases"])
    assert set(area) | {dis[15]} <= set(split["held_out_diseases"]) and not (set(area) | {dis[15]}) & seen
    assert {dis[d] for d in TOY_RANDOM_TEST} <= seen
    assert info["dropped_test_edges"] == 2 and info["diseases_with_dropped_test_edges"] == 2
    # the dropped pairs are exactly the random test edges, and they are nowhere in our therapeutic table ...
    raw = pd_read(tmp_path / "test_processed.csv")
    rnd = raw[(raw.y_type == "disease") & raw.y_idx.isin(list(TOY_RANDOM_TEST))]
    rnd_pairs = {(dis[int(r.y_idx)], drug[int(r.x_idx)]) for r in rnd.itertuples()}
    assert {tuple(p) for p in proto["dropped_test_pairs"]} == rnd_pairs
    table = set(zip(kg.t_disease.tolist(), kg.t_drug.tolist()))
    assert not rnd_pairs & table
    # ... so neither the message graph nor the training labels contain them; the train edges are still there
    mg = MessageGraph(kg, sorted(split["train_diseases"]), device="cpu")
    graph_pairs = set(zip(mg.td.tolist(), mg.tg.tolist()))
    assert not rnd_pairs & graph_pairs and not {tuple(p) for p in proto["test_pairs"]} & graph_pairs
    task = RepurposingTask(kg, split)
    for d, g in rnd_pairs:
        assert g not in task.ind.get(d, set()) and g not in task.off.get(d, set())
        assert len(task.ind[d]) == 3                                  # its 3 training indications are kept
    # the evaluation labels of the area diseases are unchanged
    assert all(len(task.eval_labels(d, False)) == 3 for d in area)
    # strict and full give the same evaluation set and candidates
    build(tmp_path, tmp_path / "kg_strict")
    strict = json.loads((tmp_path / "kg_strict" / "txgnn_protocol.json").read_text())
    for k in ("test_pairs", "eval_diseases", "candidate_drugs"):
        assert strict[k] == proto[k]


def pd_read(path):
    import pandas as pd
    return pd.read_csv(path)

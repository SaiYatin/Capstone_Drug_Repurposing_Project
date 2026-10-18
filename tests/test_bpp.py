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

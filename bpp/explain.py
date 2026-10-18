"""
Path explanations for BioPathNet++ predictions (NBFNet-style interpretability).

  python -m bpp.explain --run <run dir with best.pt + results.json> --out outputs/bpp/validation/explanations.json

Method (after Zhu et al., NeurIPS 2021): run the model's own forward pass for one (disease, drug) query
while keeping each layer's message input x_t and aggregated message s_t; back-propagate the drug's PATH
logit; the importance of edge u -r-> v at layer t is the gradient-times-input of its message,
    imp_t(u,r,v) = < dLogit/ds_t[v] , w_{t,r} * x_{t-1}[u] >,
i.e. how much that edge's message moved the score. Edges with the largest importance (any layer) form a
small graph in which a beam search finds the highest-scoring directed paths disease -> ... -> drug
(score = product of max-normalised edge importances, length <= max_len). The forward pass is checked
against model(...) to make sure the explanation describes the trained model and nothing else.
The transfer-channel (prior) part of the score is not edge-based and is reported separately.
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
from pathlib import Path

import numpy as np
import torch

from bpp.model import RelSpMM


def relation_names(kg) -> list:
    ctx, ther = list(kg.relation_names), list(kg.therapeutic_names)
    # layout from bpp/kg.MessageGraph.build: [ctx_fwd, ther_fwd, ctx_inv, ther_inv]
    return ctx + [f"{t} (drug->disease)" for t in ther] + [f"{c}^-1" for c in ctx] + [f"{t} (disease->drug)" for t in ther]


def forward_with_states(model, graph, src: int, qrel: int, cand: torch.Tensor, prior=None):
    """Exact replica of BioPathNetPP.forward + NBFLayer.forward for one query, keeping x_t and s_t."""
    mats, mats_t, deg = graph
    dev = cand.device
    srcs = torch.tensor([src], device=dev)
    q = model.query(torch.tensor([qrel], device=dev))
    V = deg.numel()
    inv_deg = 1.0 / deg.clamp(min=1.0)
    logd = torch.log(deg + 1.0)
    amp = logd / logd.mean().clamp(min=1e-6)
    ar = torch.arange(1, device=dev)
    h = q.new_zeros(V, 1, model.dim).index_put((srcs, ar), q)
    xs, ss, hs = [], [], []
    for layer in model.layers:
        w = layer.rel[torch.tensor([qrel], device=dev)]
        x = h.float()
        s = RelSpMM.apply(x, w.float(), mats, mats_t)
        s.retain_grad()
        xs.append(x.detach()); ss.append(s)
        mean = s * inv_deg.view(-1, 1, 1)
        mean = mean.index_put((srcs, ar), q, accumulate=True)
        agg = torch.cat([h, mean, mean * amp.view(-1, 1, 1)], dim=-1)
        h = torch.relu(layer.norm(layer.lin(agg))) + h
        hs.append(h[cand])
    feat = torch.cat(hs, dim=-1).transpose(0, 1)
    qx = q.unsqueeze(1).expand(-1, cand.numel(), -1)
    path_logits = model.mlp(torch.cat([feat, qx], dim=-1)).squeeze(-1)[0]
    prior_logits = model.prior(prior).squeeze(-1)[0] if (model.num_prior_feats and prior is not None) else None
    return path_logits, prior_logits, xs, ss


def edge_importance(model, graph, xs, ss, qrel: int, top_k: int = 3000, chunk: int = 2_000_000):
    """Top-k edges per layer by gradient x input. Returns list of (layer, rel, src, dst, importance)."""
    mats = graph[0]
    out = []
    for t, (x, s) in enumerate(zip(xs, ss)):
        g = s.grad[:, 0, :]
        xx = x[:, 0, :]
        w = model.layers[t].rel[qrel].detach()                   # (R, d)
        vals, meta = [], []
        for r, a in enumerate(mats):
            if a._nnz() == 0:
                continue
            crow, col = a.crow_indices(), a.col_indices()
            rows = torch.repeat_interleave(torch.arange(a.shape[0], device=col.device), crow[1:] - crow[:-1])
            for i in range(0, col.numel(), chunk):
                rr, cc = rows[i:i + chunk], col[i:i + chunk]
                imp = (g[rr] * w[r] * xx[cc]).sum(-1)
                k = min(top_k, imp.numel())
                v, idx = torch.topk(imp, k)
                vals.append(v); meta.append(torch.stack([torch.full_like(idx, r), cc[idx], rr[idx]], 1))
        v = torch.cat(vals); m = torch.cat(meta)
        k = min(top_k, v.numel())
        v, idx = torch.topk(v, k)
        m = m[idx]
        out += [(t, int(r), int(u), int(dst), float(val)) for (r, u, dst), val in zip(m.tolist(), v.tolist()) if val > 0]
    return out


def best_paths(edges, src: int, dst: int, max_len: int = 4, n_paths: int = 3, beam: int = 200):
    """Beam search over directed edges (message direction) maximising the product of normalised importances."""
    best = {}
    for t, r, u, v, imp in edges:                                 # keep each edge's max importance over layers
        if imp > best.get((u, r, v), 0):
            best[(u, r, v)] = imp
    if not best:
        return []
    top = max(best.values())
    adj = {}
    for (u, r, v), imp in best.items():
        adj.setdefault(u, []).append((v, r, np.log(imp / top)))
    found, frontier = [], [(0.0, [src], [])]
    for _ in range(max_len):
        nxt = []
        for score, nodes, rels in frontier:
            for v, r, lw in adj.get(nodes[-1], []):
                if v in nodes:
                    continue
                cand = (score + lw, nodes + [v], rels + [r])
                (found if v == dst else nxt).append(cand)
        frontier = heapq.nlargest(beam, nxt, key=lambda c: c[0])
    found.sort(key=lambda c: -c[0])
    return found[:n_paths]


def clean(n):
    return re.sub(r"\s*\[.*\]$", "", n)


def explain_query(runner, disease: int, drug_col: int, max_len=4, n_paths=3, top_k=3000):
    model = runner.model
    model.eval()
    graph = runner.mg.build(None)
    qrel = runner.qrel_ind
    prior = runner.priors([disease]) if (runner.a.prior or runner.a.no_paths) else None
    path_logits, prior_logits, xs, ss = forward_with_states(model, graph, disease, qrel, runner.cand, prior)
    with torch.no_grad():                                         # the replica must equal the model itself
        ref = model(graph, torch.tensor([disease], device=runner.dev), torch.tensor([qrel], device=runner.dev),
                    runner.cand, prior=prior)[0]
        mine = path_logits + (prior_logits if prior_logits is not None else 0)
        assert torch.allclose(ref, mine, atol=1e-4), float((ref - mine).abs().max())
    model.zero_grad(set_to_none=True)
    path_logits[drug_col].backward()
    edges = edge_importance(model, graph, xs, ss, qrel, top_k=top_k)
    drug = int(runner.task.drugs[drug_col])
    paths = best_paths(edges, disease, drug, max_len=max_len, n_paths=n_paths)
    rel = relation_names(runner.kg)
    names = runner.kg.node_names
    return {"disease": clean(names[disease]), "drug": clean(names[drug]),
            "path_logit": float(path_logits[drug_col]), "prior_logit": float(prior_logits[drug_col]) if prior_logits is not None else None,
            "paths": [{"log_score": float(s), "nodes": [clean(names[n]) for n in nodes], "relations": [rel[r] for r in rels],
                       "text": " -> ".join([clean(names[nodes[0]])] + [f"[{rel[r]}] {clean(names[n])}" for r, n in zip(rels, nodes[1:])])}
                      for s, nodes, rels in paths]}


def main():
    from bpp.train import Runner, build_parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--top-drugs", type=int, default=5)
    ap.add_argument("--extra", default="Myasthenia Gravis:Amifampridine", help="target:drug pairs, comma-separated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    saved = json.loads((run / "results.json").read_text())["args"]
    args = build_parser().parse_args(["--data", a.data, "--out", str(run / "_explain_tmp"), "--split", saved["split"],
                                      "--device", a.device, "--seed", str(saved["seed"])] + (["--prior"] if saved["prior"] else []))
    r = Runner(args)
    ck = torch.load(run / a.ckpt, map_location=r.dev, weights_only=False)
    r.model.load_state_dict(ck["model"])
    r.model.eval()                                   # no dropout: rankings must match bpp/score.py exactly
    targets = {n: int(p) for n, p in r.task.split["target_primary"].items()}
    jobs = []
    for t, d in targets.items():
        with torch.no_grad():
            sc = r.forward([d], train=False)[0].float().cpu().numpy()
        jobs += [(t, d, int(c)) for c in np.argsort(-sc)[:a.top_drugs]]
    for pair in filter(None, a.extra.split(",")):
        t, drug = pair.split(":")
        col = next(i for i, g in enumerate(r.task.drugs) if clean(r.kg.node_names[g]).lower() == drug.lower())
        jobs.append((t, targets[t], col))
    res = []
    for t, d, c in jobs:
        e = explain_query(r, d, c)
        e["target"] = t
        res.append(e)
        print(t, "|", e["drug"], "|", e["paths"][0]["text"] if e["paths"] else "no path found", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

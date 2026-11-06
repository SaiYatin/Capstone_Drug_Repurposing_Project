"""
Knowledge-graph embedding baselines (TransE, DistMult, RotatE) on exactly the same split and graph.

The training triples are the message graph of the split: all PrimeKG context edges plus the
drug-disease edges (indication, contraindication, off-label) of TRAINING diseases only. Held-out
validation/test families therefore keep their context edges (proteins, phenotypes, ...) but never
their drug edges, as for BioPathNet++. A test disease is scored as (disease, indication, ?) over all
7,957 drugs, with filtered ranks and the metrics of bpp/metrics.py.

Training follows Sun et al. (RotatE, ICLR 2019): 256 uniformly drawn negatives shared by each batch, head-
and tail-corruption batches alternating,
self-adversarial negative weighting (temperature 1), Adam. Hyperparameters are fixed in advance
(dimension 128 real numbers per entity for every model, margin 9 for TransE/RotatE, lr 1e-3, batch
1024); the number of epochs is chosen by validation MRR (evaluated every EVAL_EVERY epochs), exactly
as early stopping selects BioPathNet++'s epoch.

  python -m bpp.kge --data datasets/bpp_primekg --split splits_test.json --model rotate --seed 0 --out outputs/bpp/kge
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bpp.kg import CompactKG
from bpp.metrics import aggregate, disease_metrics
from bpp.task import RepurposingTask

DIM, NEG, BATCH, LR, GAMMA, ADV_T = 128, 256, 1024, 1e-3, 9.0, 1.0


class KGE(torch.nn.Module):
    def __init__(self, kind: str, n_ent: int, n_rel: int, dim: int = DIM):
        super().__init__()
        self.kind = kind
        self.ent = torch.nn.Embedding(n_ent, dim)
        rdim = dim // 2 if kind == "rotate" else dim
        self.rel = torch.nn.Embedding(n_rel, rdim)
        bound = (GAMMA + 2.0) / dim                      # RotatE initialisation range
        torch.nn.init.uniform_(self.ent.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel.weight, -bound, bound)
        self.bound = bound

    def score(self, h, r, t):
        """h, t: (..., dim) entity vectors; r: relation ids broadcastable. Higher = more plausible."""
        rv = self.rel(r)
        if self.kind == "transe":
            return GAMMA - (h + rv - t).abs().sum(-1)
        if self.kind == "distmult":
            return (h * rv * t).sum(-1)
        # rotate: complex rotation, phase from relation embedding
        hr, hi = h.chunk(2, -1); tr, ti = t.chunk(2, -1)
        ph = rv / (self.bound / math.pi)
        cr, ci = torch.cos(ph), torch.sin(ph)
        re = hr * cr - hi * ci - tr
        im = hr * ci + hi * cr - ti
        return GAMMA - torch.sqrt(re ** 2 + im ** 2 + 1e-9).sum(-1)


def triples(kg: CompactKG, train_diseases) -> tuple[np.ndarray, int, int]:
    """(N, 3) head, relation, tail; therapeutic relations get ids after the context relations."""
    R = len(kg.relation_names)
    ctx = np.stack([kg.head, kg.rel, kg.tail], 1).astype(np.int64)
    keep = np.isin(kg.t_disease, np.asarray(sorted(train_diseases)))
    ther = np.stack([kg.t_disease[keep], R + kg.t_rel[keep], kg.t_drug[keep]], 1).astype(np.int64)
    return np.concatenate([ctx, ther]), R, R + len(kg.therapeutic_names)


def train_epoch(model, opt, T: torch.Tensor, n_ent: int, gen: torch.Generator):
    model.train()
    perm = torch.randperm(len(T), generator=gen, device=T.device)
    tot = 0.0
    for step, i in enumerate(range(0, len(T), BATCH)):
        b = T[perm[i:i + BATCH]]
        h, r, t = b[:, 0], b[:, 1], b[:, 2]
        he, te = model.ent(h), model.ent(t)
        pos = model.score(he, r, te)
        # NEG negatives shared by the batch; head- and tail-corruption batches alternate, as in the
        # RotatE reference implementation
        ne = model.ent(torch.randint(0, n_ent, (NEG,), generator=gen, device=T.device))[None]   # (1, NEG, d)
        if step % 2 == 0:
            neg = model.score(he[:, None, :], r[:, None], ne)        # corrupt tails
        else:
            neg = model.score(ne, r[:, None], te[:, None, :])        # corrupt heads
        w = torch.softmax(neg.detach() * ADV_T, -1)
        loss = -F.logsigmoid(pos).mean() - (w * F.logsigmoid(-neg)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    return tot / len(T)


@torch.no_grad()
def evaluate(model, task: RepurposingTask, items, rel_ind: int, dev) -> dict:
    model.eval()
    drugs = torch.as_tensor(task.drugs, device=dev)
    dv = model.ent(drugs)
    per = {}
    for name, d, offl in items:
        pos = task.eval_labels(d, offl)
        if len(pos) == 0:
            continue
        h = model.ent(torch.as_tensor([d], device=dev))
        s = model.score(h.expand_as(dv), torch.as_tensor(rel_ind, device=dev), dv).float().cpu().numpy()
        per[name] = disease_metrics(s, pos, hard_neg_idx=task.hard_negatives(pos), degree=task.drug_degree)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--split", default="splits_test.json")
    ap.add_argument("--model", choices=["transe", "distmult", "rotate"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="outputs/bpp/kge")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    kg = CompactKG.load(a.data)
    split = json.loads((Path(a.data) / a.split).read_text())
    task = RepurposingTask(kg, split)
    T_np, R, n_rel = triples(kg, task.train)
    rel_ind = R + kg.therapeutic_names.index("indication")
    T = torch.as_tensor(T_np, device=dev)
    model = KGE(a.model, kg.num_nodes, n_rel).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    sets = task.eval_sets()
    print(f"{a.model} seed {a.seed}: {len(T_np):,} training triples, {kg.num_nodes:,} entities", flush=True)

    best, best_state, log = -1.0, None, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, opt, T, kg.num_nodes, gen)
        rec = {"epoch": ep, "loss": loss, "sec": time.time() - t0}
        if ep % a.eval_every == 0 or ep == a.epochs:
            v = aggregate(evaluate(model, task, sets["valid"], rel_ind, dev))
            rec["valid_mrr"] = v["mrr"]["mean"]
            if v["mrr"]["mean"] > best:
                best, best_state = v["mrr"]["mean"], {k: x.detach().clone() for k, x in model.state_dict().items()}
                rec["best"] = True
        log.append(rec)
        print(json.dumps(rec), flush=True)

    model.load_state_dict(best_state)
    res = {"model": a.model, "seed": a.seed, "split": a.split, "best_valid_mrr": best,
           "hyper": {"dim": DIM, "neg": NEG, "batch": BATCH, "lr": LR, "gamma": GAMMA, "adv_t": ADV_T,
                     "epochs": a.epochs, "eval_every": a.eval_every},
           "log": log, "per_disease": {}, "summary": {}}
    for s in ("valid", "test", "targets"):
        if s in sets:
            per = evaluate(model, task, sets[s], rel_ind, dev)
            res["per_disease"][s] = per
            res["summary"][s] = aggregate(per)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = Path(a.split).stem.replace("splits_", "").replace("splits", "targets")
    (out / f"kge_{a.model}_{tag}_s{a.seed}.json").write_text(json.dumps(res))
    print({s: round(v["mrr"]["mean"], 4) for s, v in res["summary"].items()})


if __name__ == "__main__":
    main()

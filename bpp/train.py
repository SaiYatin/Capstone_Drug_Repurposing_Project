"""
Train / evaluate BioPathNet++ v3 (CPU or CUDA; resumable, so it survives Kaggle session limits).

Examples
  # main model on the 6-target zero-shot protocol
  python -m bpp.train --data datasets/bpp_primekg --split splits.json --out outputs/bpp/full_s0 --prior
  # E2: no autoimmune diseases in training
  python -m bpp.train --data datasets/bpp_primekg --split splits_noauto.json --out outputs/bpp/noauto_s0 --prior
  # ablations: --no-prior (paths only), --no-paths (transfer channel only), --loss bce, --relations brg
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from bpp.kg import CompactKG, MessageGraph
from bpp.metrics import aggregate, disease_metrics
from bpp.model import BioPathNetPP, bce_loss, count_parameters, listwise_loss
from bpp.task import RepurposingTask

# Background-regulatory-graph preset (BioPathNet idea): drop the two largest, least
# mechanistic relation families (tissue expression, drug-drug interactions).
BRG_EXCLUDE = {"anatomy_protein_present", "anatomy_protein_absent", "drug_drug"}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--loss", choices=["listwise", "bce"], default="listwise")
    ap.add_argument("--prior", action="store_true", help="enable the transfer channel (++)")
    ap.add_argument("--no-paths", action="store_true", help="ablation: transfer channel only")
    ap.add_argument("--relations", choices=["all", "brg"], default="all")
    ap.add_argument("--max-steps", type=int, default=0, help="steps per epoch cap (smoke tests)")
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--time-budget-h", type=float, default=11.0, help="stop cleanly before Kaggle's 12 h limit")
    return ap


class Runner:
    def __init__(self, a):
        self.a = a
        set_seed(a.seed)
        self.dev = torch.device(a.device)
        self.out = Path(a.out); self.out.mkdir(parents=True, exist_ok=True)
        self.kg = CompactKG.load(a.data)
        self.split = json.loads((Path(a.data) / a.split).read_text())
        self.task = RepurposingTask(self.kg, self.split)
        allowed = sorted(set(self.task.train))       # ONLY train diseases' drug edges may be in the graph
        rels = None
        if a.relations == "brg":
            rels = [r for r in self.kg.relation_names if r not in BRG_EXCLUDE]
        t0 = time.time()
        self.mg = MessageGraph(self.kg, allowed, device=self.dev, include_relations=rels)
        print(f"graph: {self.kg.num_nodes:,} nodes, {self.mg.num_ctx_edges:,} context edges, "
              f"{self.mg.num_relations} relations (incl. inverse) [{time.time() - t0:.1f}s]", flush=True)
        self.cand = torch.as_tensor(self.task.drugs, device=self.dev)
        self.model = BioPathNetPP(self.mg.num_relations, num_query_relations=len(self.kg.therapeutic_names),
                                  dim=a.dim, num_layers=a.layers,
                                  num_prior_feats=4 if (a.prior or a.no_paths) else 0).to(self.dev)
        print(f"parameters: {count_parameters(self.model):,}", flush=True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr, weight_decay=a.wd)
        self.qrel_ind = self.kg.therapeutic_names.index("indication")
        self.prior_cache = {}
        self.start_epoch, self.best, self.bad = 0, -1.0, 0
        self.t_start = time.time()
        self._maybe_resume()

    # -------------------------------------------------------------------------------
    def priors(self, diseases):
        if not (self.a.prior or self.a.no_paths):
            return None
        miss = [d for d in diseases if d not in self.prior_cache]
        if miss:
            feats = self.task.prior_features(miss)
            for d, f in zip(miss, feats):
                self.prior_cache[d] = f
        return torch.stack([self.prior_cache[d] for d in diseases]).to(self.dev)

    def drop_set(self, diseases):
        s = set()
        for d in diseases:
            s |= self.task.exclusion(d)
        return torch.as_tensor(sorted(s), device=self.dev)

    def forward(self, diseases, train: bool):
        src = torch.as_tensor(diseases, device=self.dev)
        qrel = torch.full_like(src, self.qrel_ind)
        graph = self.mg.build(drop_diseases=self.drop_set(diseases) if train else None)
        return self.model(graph, src, qrel, self.cand, prior=self.priors(diseases),
                          use_paths=not self.a.no_paths)

    # -------------------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        order = list(self.task.train)
        random.Random(self.a.seed * 1000 + epoch).shuffle(order)
        B = self.a.batch
        steps = math.ceil(len(order) / B)
        if self.a.max_steps:
            steps = min(steps, self.a.max_steps)
        tot, t0 = 0.0, time.time()
        for s in range(steps):
            batch = order[s * B:(s + 1) * B]
            mask = torch.zeros(len(batch), self.task.C, dtype=torch.bool)
            for i, d in enumerate(batch):
                mask[i, self.task.Y[self.task.train_row[d]].indices] = True
            mask = mask.to(self.dev)
            logits = self.forward(batch, train=True)
            loss = listwise_loss(logits, mask) if self.a.loss == "listwise" else bce_loss(logits, mask)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            tot += loss.item()
            if s == 0 or (s + 1) % 25 == 0 or s == steps - 1:
                mem = torch.cuda.max_memory_allocated(self.dev) / 1e9 if self.dev.type == "cuda" else 0
                print(f"  ep{epoch} step {s + 1}/{steps} loss {loss.item():.4f} "
                      f"{(time.time() - t0) / (s + 1):.2f}s/step peak_mem {mem:.2f}GB", flush=True)
        return tot / steps

    @torch.no_grad()
    def evaluate(self, sets=None, save_rankings=False):
        self.model.eval()
        res, rankings = {}, {}
        for set_name, items in self.task.eval_sets().items():
            if sets and set_name not in sets:
                continue
            items = [(l, d, o) for l, d, o in items if len(self.task.eval_labels(d, o))]
            per = {}
            for i in range(0, len(items), self.a.eval_batch):
                chunk = items[i:i + self.a.eval_batch]
                logits = self.forward([d for _, d, _ in chunk], train=False).float().cpu().numpy()
                for (label, d, offl), sc in zip(chunk, logits):
                    pos = self.task.eval_labels(d, offl)
                    per[label] = disease_metrics(sc, pos, self.task.hard_negatives(pos),
                                                 self.task.drug_degree, np.random.default_rng(0))
                    if save_rankings and set_name == "targets":
                        top = np.argsort(-sc)[:100]
                        ind, off = self.task.ind.get(d, set()), self.task.off.get(d, set())
                        rankings[label] = [{"rank": r + 1, "drug": self.kg.node_names[self.task.drugs[j]],
                                            "score": float(sc[j]),
                                            "label": "indication" if self.task.drugs[j] in ind else
                                                     ("off-label" if self.task.drugs[j] in off else "")}
                                           for r, j in enumerate(top)]
            res[set_name] = {"aggregate": aggregate(per), "per_disease": per}
        return res, rankings

    # -------------------------------------------------------------------------------
    def _ckpt(self, name, epoch):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "epoch": epoch,
                    "best": self.best, "bad": self.bad, "args": vars(self.a)}, self.out / name)

    def _maybe_resume(self):
        p = self.out / "last.pt"
        if p.exists():
            ck = torch.load(p, map_location=self.dev, weights_only=False)
            self.model.load_state_dict(ck["model"]); self.opt.load_state_dict(ck["opt"])
            self.start_epoch, self.best, self.bad = ck["epoch"] + 1, ck["best"], ck["bad"]
            print(f"resumed from epoch {ck['epoch']} (best valid MRR {self.best:.4f})", flush=True)

    def run(self):
        log = []
        for ep in range(self.start_epoch, self.a.epochs):
            loss = self.train_epoch(ep)
            v, _ = self.evaluate(sets={"valid"})
            mrr = v["valid"]["aggregate"]["mrr"]["mean"]
            log.append({"epoch": ep, "loss": loss, "valid_mrr": mrr,
                        "valid_hits@50": v["valid"]["aggregate"]["hits@50"]["mean"]})
            print(f"epoch {ep}: loss {loss:.4f}  valid MRR {mrr:.4f}  H@50 {log[-1]['valid_hits@50']:.3f}  "
                  f"[{(time.time() - self.t_start) / 3600:.2f} h]", flush=True)
            if mrr > self.best:
                self.best, self.bad = mrr, 0
                self._ckpt("best.pt", ep)
            else:
                self.bad += 1
            self._ckpt("last.pt", ep)
            with open(self.out / "train_log.jsonl", "a") as f:
                f.write(json.dumps(log[-1]) + "\n")
            if self.bad >= self.a.patience:
                print("early stopping", flush=True); break
            if (time.time() - self.t_start) / 3600 > self.a.time_budget_h:
                print("time budget reached; rerun the same command to resume", flush=True); return
        ck = torch.load(self.out / "best.pt", map_location=self.dev, weights_only=False)
        self.model.load_state_dict(ck["model"])
        res, rankings = self.evaluate(save_rankings=True)
        res["best_epoch"] = ck["epoch"]; res["args"] = vars(self.a)
        (self.out / "results.json").write_text(json.dumps(res, indent=1))
        (self.out / "target_rankings.json").write_text(json.dumps(rankings, indent=1))
        for set_name, r in res.items():
            if isinstance(r, dict) and "aggregate" in r:
                g = r["aggregate"]
                print(f"[{set_name}] MRR {g['mrr']['mean']:.4f} {g['mrr']['ci95']}  H@10 {g['hits@10']['mean']:.3f}  "
                      f"H@50 {g['hits@50']['mean']:.3f}  AUROC {g['auroc']['mean']:.3f}  "
                      f"AUROC-hard {g.get('auroc_hard', {}).get('mean', float('nan')):.3f}", flush=True)
        print(f"saved {self.out}/results.json", flush=True)


def main():
    Runner(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()

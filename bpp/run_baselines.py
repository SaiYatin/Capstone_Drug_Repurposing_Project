"""
Evaluate the non-neural baselines under the same zero-shot protocol as the model.

  python -m bpp.run_baselines --data datasets/bpp_primekg --split splits.json --out outputs/bpp/baselines_targets.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bpp.kg import CompactKG
from bpp.metrics import aggregate, disease_metrics
from bpp.task import RepurposingTask


def evaluate_scorer(task, score_fn, rng_seed=0):
    results = {}
    for set_name, items in task.eval_sets().items():
        per = {}
        for label, d, offl in items:
            pos = task.eval_labels(d, offl)
            if len(pos) == 0:
                continue
            s = score_fn(d)
            per[label] = disease_metrics(s, pos, task.hard_negatives(pos), task.drug_degree,
                                         np.random.default_rng(rng_seed))
        if per:
            results[set_name] = {"aggregate": aggregate(per), "per_disease": per}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kg = CompactKG.load(a.data)
    split = json.loads((Path(a.data) / a.split).read_text())
    task = RepurposingTask(kg, split)
    names = list(task.baselines(task.valid[0]).keys())
    cache = {}

    def scorer(name):
        def f(d):
            if d not in cache:
                cache[d] = task.baselines(d)
            return cache[d][name]
        return f

    out = {name: evaluate_scorer(task, scorer(name)) for name in names}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for set_name in out[names[0]]:
        print(f"\n== {set_name} ({split['protocol']}) ==")
        print(f"{'method':22s} {'MRR':>7s} {'H@10':>6s} {'H@50':>6s} {'AUROC':>6s} {'AUROC-hard':>10s} {'AUROC-deg':>9s}  n_dis n_pos")
        for name in names:
            g = out[name][set_name]["aggregate"]
            f = lambda k: g[k]["mean"] if k in g else float("nan")
            print(f"{name:22s} {f('mrr'):7.3f} {f('hits@10'):6.3f} {f('hits@50'):6.3f} {f('auroc'):6.3f} "
                  f"{f('auroc_hard'):10.3f} {f('auroc_degmatch'):9.3f}  {g['n_diseases']:5d} {g['n_pos']:5d}")


if __name__ == "__main__":
    main()

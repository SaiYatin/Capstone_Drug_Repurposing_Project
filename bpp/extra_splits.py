"""
Extra zero-shot protocols derived from splits.json (the "targets" protocol).

  python -m bpp.extra_splits --data datasets/bpp_primekg

Transfer curve (Aim 1: does knowledge transfer from data-rich autoimmune diseases?)
  splits_tr_A.json        A   train ONLY on the RA / SLE / MS strict families (the original project design)
  splits_tr_Actl{j}.json  A'  j-th control: non-autoimmune training diseases matched to A disease-by-disease
                              on number of indications (same size, ~same label count, different biology)
  splits_tr_CE.json       C_E train on every disease except the evaluation diseases (broad data incl. autoimmune)
  The existing no-autoimmune runs (splits_noauto.json, "D": broad data WITHOUT autoimmune diseases) never
  train on the evaluation diseases either, so they are a further point on the same curve.

  Evaluation set E (`eval_area`, reported as set "area"): TxGNN autoimmune-area diseases with indications
  that are neither targets nor members of the RA/SLE/MS families. E and all its disease-disease neighbours
  are excluded from training in every transfer split. ("All autoimmune diseases" cannot be a separate
  training condition: beyond RA/SLE/MS, PrimeKG's autoimmune diseases ARE the evaluation area.)

Selection-free test split (removes the "validation also picks the epoch" caveat)
  splits_test.json        targets protocol + `test_diseases`: random non-autoimmune labelled diseases,
                          held out with their families, used neither for training nor for model selection.

Leakage rule: the message graph only ever contains therapeutic edges of `train_diseases`
(bpp/train.py builds MessageGraph from task.train), so a correct train list is the whole guarantee.
tests/test_bpp.py checks it for every file written here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bpp.kg import CompactKG
from bpp.task import RepurposingTask

N_CONTROLS = 3


def families(task: RepurposingTask, diseases) -> set:
    out = set()
    for d in diseases:
        out |= task.exclusion(d)
    return out


def matched_controls(task, reference, candidates, rng) -> list:
    """Pick, without replacement, the candidate whose #indications is closest to each reference disease."""
    pool = list(candidates)
    picked = []
    for d in sorted(reference, key=lambda x: -len(task.ind[x])):
        target = len(task.ind[d])
        gaps = np.array([abs(len(task.ind[c]) - target) for c in pool])
        best = np.flatnonzero(gaps == gaps.min())
        c = pool.pop(int(rng.choice(best)))
        picked.append(c)
    return sorted(picked)


def derive(base: dict, condition: str, train, extra: dict) -> dict:
    s = json.loads(json.dumps(base))
    held = set(s["held_out_diseases"]) | (set(s["train_diseases"]) - set(train))
    s.update({"protocol": "transfer" if condition != "test" else "test", "condition": condition,
              "derived_from": "splits.json", "train_diseases": sorted(int(x) for x in train),
              "held_out_diseases": sorted(int(x) for x in held)})
    s.update(extra)
    return s


def build(data: Path, seed: int = 0) -> dict:
    kg = CompactKG.load(str(data))
    base = json.loads((data / "splits.json").read_text())
    task = RepurposingTask(kg, base)
    rich = set().union(*map(set, base["rich_families"].values()))
    target_fam = set().union(*map(set, base["targets"].values()))
    auto = task.autoimmune

    # evaluation set E and the training pool that excludes E's families
    E = sorted(d for d in base["area_diseases"] if d in task.ind and d not in rich and d not in target_fam)
    pool = [d for d in task.train if d not in families(task, E)]
    A = [d for d in pool if d in rich]
    non_auto = [d for d in pool if not (task.exclusion(d) & auto)]

    out = {"splits_tr_A.json": derive(base, "A", A, {"eval_area": E}),
           "splits_tr_CE.json": derive(base, "CE", pool, {"eval_area": E})}
    for j in range(N_CONTROLS):
        ctl = matched_controls(task, A, non_auto, np.random.default_rng(seed + 100 + j))
        out[f"splits_tr_Actl{j}.json"] = derive(base, f"Actl{j}", ctl, {"eval_area": E})

    # selection-free test split: as many test diseases as validation diseases, non-autoimmune
    rng = np.random.default_rng(seed + 1)
    labelled = sorted(non_auto)
    test = sorted(rng.choice(labelled, len(base["valid_diseases"]), replace=False).tolist())
    train_test = [d for d in task.train if d not in families(task, test)]
    out["splits_test.json"] = derive(base, "test", train_test, {"test_diseases": [int(x) for x in test]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data)
    kg = CompactKG.load(str(data))
    for name, s in build(data, a.seed).items():
        (data / name).write_text(json.dumps(s, indent=2))
        task = RepurposingTask(kg, s)
        n_ind = sum(len(task.ind[d]) for d in task.train)
        ev = {k: len([x for x in v if len(task.eval_labels(x[1], x[2]))]) for k, v in task.eval_sets().items()}
        print(f"{name:24s} train {len(task.train):5d} diseases / {n_ind:5d} indications   eval {ev}")


if __name__ == "__main__":
    main()

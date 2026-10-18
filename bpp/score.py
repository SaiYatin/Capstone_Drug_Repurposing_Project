"""
Full drug scores (all 7,957 drugs) for chosen evaluation diseases from a trained checkpoint.

  python -m bpp.score --run <run dir with results.json + best.pt> --sets targets --out scores_main_s0.npz

The saved target_rankings.json only keeps the top 100; validation (bpp/validate.py) needs the full
ranking, e.g. for drugs approved after the PrimeKG snapshot that may rank anywhere.
Output npz: drugs (C,) node ids, diseases (D,) node ids, names (D,), scores (D, C).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bpp.train import Runner, build_parser


@torch.no_grad()
def score(run: Path, ckpt: str, data: str, sets, device: str, batch: int = 2):
    saved = json.loads((run / "results.json").read_text())["args"]
    args = build_parser().parse_args(["--data", data, "--out", str(run / "_score_tmp"), "--split", saved["split"],
                                      "--device", device, "--seed", str(saved["seed"]),
                                      "--dim", str(saved["dim"]), "--layers", str(saved["layers"]),
                                      "--loss", saved["loss"], "--relations", saved["relations"]]
                                     + (["--prior"] if saved["prior"] else []) + (["--no-paths"] if saved["no_paths"] else []))
    r = Runner(args)
    ck = torch.load(run / ckpt, map_location=r.dev, weights_only=False)
    r.model.load_state_dict(ck["model"])
    r.model.eval()
    items = [it for s, its in r.task.eval_sets().items() if s in sets for it in its]
    names, nodes, rows = [], [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        rows.append(r.forward([d for _, d, _ in chunk], train=False).float().cpu().numpy())
        names += [n for n, _, _ in chunk]; nodes += [d for _, d, _ in chunk]
    return {"drugs": np.asarray(r.task.drugs), "diseases": np.asarray(nodes), "names": np.asarray(names),
            "scores": np.concatenate(rows), "epoch": ck["epoch"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--sets", default="targets")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = score(Path(a.run), a.ckpt, a.data, a.sets.split(","), a.device)
    np.savez_compressed(a.out, **res)
    print(f"{a.run}: {res['scores'].shape} -> {a.out}")


if __name__ == "__main__":
    main()

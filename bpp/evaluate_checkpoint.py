"""
Re-evaluate a saved checkpoint (best.pt or last.pt) of a finished run, without training.

  python -m bpp.evaluate_checkpoint --run <run dir with train.log args> --ckpt last.pt --out results_last.json

Used for the sensitivity check of the transfer-curve runs: those train on 11 autoimmune diseases but
early-stop on the 68 general validation diseases, so the selected epoch can be very early.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bpp.train import Runner, build_parser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing results.json (for its args) and the checkpoint")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-batch", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    saved = json.loads((run / "results.json").read_text())["args"]
    args = build_parser().parse_args(["--data", a.data, "--out", str(Path(a.out).parent / "_eval_tmp"),
                                      "--split", saved["split"], "--device", a.device,
                                      "--eval-batch", str(a.eval_batch), "--seed", str(saved["seed"]),
                                      "--dim", str(saved["dim"]), "--layers", str(saved["layers"]),
                                      "--loss", saved["loss"], "--relations", saved["relations"]]
                                     + (["--prior"] if saved["prior"] else []) + (["--no-paths"] if saved["no_paths"] else []))
    r = Runner(args)                                   # fresh out dir: nothing is resumed
    ck = torch.load(run / a.ckpt, map_location=r.dev, weights_only=False)
    r.model.load_state_dict(ck["model"])
    res, _ = r.evaluate()
    res["checkpoint"] = a.ckpt
    res["checkpoint_epoch"] = ck["epoch"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{run.name} {a.ckpt} (epoch {ck['epoch']}):",
          {s: round(v["aggregate"]["mrr"]["mean"], 3) for s, v in res.items() if isinstance(v, dict) and "aggregate" in v})


if __name__ == "__main__":
    main()

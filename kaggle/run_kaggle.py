"""
BioPathNet++ v3 -- Kaggle runner (GPU T4 x2)
============================================

ONE-TIME SETUP
  1. Create a Kaggle dataset named  biopathnet-pp-v3  containing two folders from the project:
        bpp/                    (the code package)
        datasets/bpp_primekg/   (the 9 MB compact PrimeKG build)
  2. New notebook -> Settings: Accelerator = GPU T4 x2, Internet = off (not needed).
     Add Data -> your dataset.  Paste this whole file into one cell and run.

WHAT IT DOES
  stage 0  environment + GPU report
  stage 1  baselines on CPU (seconds)
  stage 2  smoke test on cuda:0: 3 training steps -> measured s/step, peak VRAM, projected epoch time
  stage 3  experiment queue, two experiments at a time (one per T4); each run is resumable
  stage 4  summary table -> /kaggle/working/summary.md

RESUMING AFTER THE 12 h SESSION LIMIT
  "Save Version" (Save & Run All) keeps /kaggle/working. In the next session, add the previous
  notebook version's output as an input; this script copies runs/ back and every unfinished
  experiment resumes from its last.pt. Finished experiments (results.json present) are skipped.

QUOTA PLAN (30 GPU-h per week, both GPUs count as one session-hour)
  Stage 3 runs EXPERIMENTS top to bottom; the first 4 are the paper's core and should finish in
  the first session if the smoke test reports <= ~1.5 s/step. Edit EXPERIMENTS / EPOCHS below.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

EPOCHS = 20
COMMON = ["--dim", "32", "--layers", "6", "--batch", "8", "--epochs", str(EPOCHS), "--patience", "6",
          "--time-budget-h", "11.0"]
EXPERIMENTS = [
    # name,                 extra args                                      purpose
    ("main_s0",            ["--prior", "--seed", "0"]),                    # BioPathNet++ (paths + transfer)
    ("paths_s0",           ["--seed", "0"]),                               # ablation: paths only (BioPathNet-style)
    ("noauto_main_s0",     ["--prior", "--seed", "0", "--split", "splits_noauto.json"]),  # E2 transfer test
    ("prioronly_s0",       ["--prior", "--no-paths", "--seed", "0"]),      # ablation: transfer channel only
    ("main_s1",            ["--prior", "--seed", "1"]),
    ("main_s2",            ["--prior", "--seed", "2"]),
    ("paths_s1",           ["--seed", "1"]),
    ("paths_s2",           ["--seed", "2"]),
    ("noauto_main_s1",     ["--prior", "--seed", "1", "--split", "splits_noauto.json"]),
    ("noauto_main_s2",     ["--prior", "--seed", "2", "--split", "splits_noauto.json"]),
    ("noauto_paths_s0",    ["--seed", "0", "--split", "splits_noauto.json"]),
    ("brg_main_s0",        ["--prior", "--seed", "0", "--relations", "brg"]),  # BRG ablation
    ("bce_main_s0",        ["--prior", "--seed", "0", "--loss", "bce"]),       # loss ablation
]

INPUT = os.environ.get("BPP_INPUT", "/kaggle/input")
WORK = Path(os.environ.get("BPP_WORK", "/kaggle/working"))
MAX_EXP = int(os.environ.get("BPP_MAX_EXPERIMENTS", "999"))  # local dry runs: 0
RUNS = WORK / "runs"


def sh(cmd, log=None, env=None):
    print("$", " ".join(cmd), flush=True)
    if log is None:
        return subprocess.run(cmd, check=False, env=env).returncode
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env).returncode


def locate():
    metas = glob.glob(f"{INPUT}/**/bpp_primekg/meta.json", recursive=True)
    pkgs = glob.glob(f"{INPUT}/**/bpp/train.py", recursive=True)
    assert metas and pkgs, "dataset not found: add the biopathnet-pp-v3 dataset to the notebook"
    data = str(Path(metas[0]).parent)
    code_root = str(Path(pkgs[0]).parent.parent)
    # copy code to a writable place and put it on PYTHONPATH
    shutil.copytree(Path(code_root) / "bpp", WORK / "code" / "bpp", dirs_exist_ok=True)
    return data, str(WORK / "code")


def restore_previous_runs():
    for prev in glob.glob(f"{INPUT}/**/runs/*/last.pt", recursive=True):
        src = Path(prev).parent
        dst = RUNS / src.name
        if not (dst / "last.pt").exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"restored {src.name} from previous session", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    data, code = locate()
    env = dict(os.environ, PYTHONPATH=code + os.pathsep + os.environ.get("PYTHONPATH", ""))
    py = sys.executable
    restore_previous_runs()

    # stage 0
    if shutil.which("nvidia-smi"):
        sh(["nvidia-smi"])
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count(), flush=True)
    ngpu = max(1, torch.cuda.device_count())
    dev = (lambda g: f"cuda:{g}") if torch.cuda.is_available() else (lambda g: "cpu")

    # stage 1: baselines (CPU)
    for split, tag in (("splits.json", "targets"), ("splits_noauto.json", "noauto")):
        out = WORK / f"baselines_{tag}.json"
        if not out.exists():
            sh([py, "-m", "bpp.run_baselines", "--data", data, "--split", split, "--out", str(out)], env=env)

    # stage 2: smoke test
    smoke = RUNS / "_smoke"
    if not (smoke / "smoke_ok").exists():
        shutil.rmtree(smoke, ignore_errors=True)
        rc = sh([py, "-m", "bpp.train", "--data", data, "--out", str(smoke), "--device", dev(0),
                 "--prior", "--epochs", "1", "--max-steps", "3"] + COMMON[:6], log=str(WORK / "smoke.log"), env=env)
        print(open(WORK / "smoke.log").read()[-3000:], flush=True)
        assert rc == 0, "smoke test failed -- see smoke.log"
        (smoke / "smoke_ok").write_text("ok")

    # stage 3: queue, one worker per GPU
    queue = [e for e in EXPERIMENTS if not (RUNS / e[0] / "results.json").exists()][:MAX_EXP]
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                name, extra = queue.pop(0)
            out = RUNS / name
            out.mkdir(parents=True, exist_ok=True)
            args = [py, "-m", "bpp.train", "--data", data, "--out", str(out), "--device", dev(gpu)] + COMMON + extra
            t0 = time.time()
            rc = sh(args, log=str(out / "train.log"), env=env)
            print(f"[gpu{gpu}] {name} finished rc={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(ngpu)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(300)
        for log in sorted(RUNS.glob("*/train.log")):
            lines = open(log).read().strip().splitlines()
            if lines:
                print(f"  {log.parent.name:18s} | {lines[-1][:120]}", flush=True)
    for t in threads:
        t.join()

    # stage 4: summary
    rows = ["| run | set | MRR | 95% CI | Hits@10 | Hits@50 | AUROC | AUROC-hard |", "|---|---|---|---|---|---|---|---|"]
    for res in sorted(RUNS.glob("*/results.json")):
        r = json.loads(res.read_text())
        for s in ("valid", "targets", "area"):
            if s in r:
                g = r[s]["aggregate"]
                ci = g["mrr"]["ci95"]
                rows.append(f"| {res.parent.name} | {s} | {g['mrr']['mean']:.3f} | {ci[0]:.3f}-{ci[1]:.3f} | "
                            f"{g['hits@10']['mean']:.3f} | {g['hits@50']['mean']:.3f} | {g['auroc']['mean']:.3f} | "
                            f"{g.get('auroc_hard', {}).get('mean', float('nan')):.3f} |")
    (WORK / "summary.md").write_text("\n".join(rows))
    print("\n".join(rows))


main()

"""
BioPathNet++ vs TxGNN head-to-head on TxGNN's own autoimmune disease-area split  (Kaggle, GPU T4 x2)
=====================================================================================================

NOTEBOOK SETTINGS: Accelerator = GPU T4 x2, **Internet = ON** (TxGNN code from GitHub, its data from
Harvard Dataverse, DGL from data.dgl.ai). Input: the biopathnet-pp-v3 dataset (version with
bpp/build_txgnn.py and bpp/txgnn_eval.py). Paste this file into one cell, Save Version -> Save & Run All.

STAGES (all heavy work runs in subprocesses, logs in /kaggle/working/txgnn_h2h/logs)
  1 install   TxGNN (pinned-commit tarball) in its OWN Python 3.11 env (uv): torch 2.4.0+cu124, DGL 2.4.0+cu124,
              torch_geometric, goatools; DGL self-test. (DGL has no wheels for Kaggle's Python 3.13; our model
              keeps using Kaggle's own Python.)
  2 split     TxGNN's own code builds the 'autoimmune' disease-area split; exported to export/
  3 convert   bpp.build_txgnn -> our CompactKG with the same training graph and test labels
  4 train     GPU0: TxGNN with the official reproduce/train.py settings  |  GPU1: BioPathNet++ (--prior)
  5 evaluate  both models, TxGNN's disease-centric protocol AND our full-ranking protocol -> h2h.json

Fairness notes (also written to summary.md)
  * TxGNN draws the 5% random test edges of a disease-area split with numpy's global RNG and no seed;
    we fix np.random.seed(SPLIT_RANDOM_SEED) so the split is reproducible. It is therefore a different
    random draw from TxGNN's published runs, but the same protocol.
  * TxGNN pretraining (1 epoch, 2-layer full-neighbour sampling) is capped at PRETRAIN_MAX_H; if the cap
    is hit, the number of pretraining steps is logged and must be reported.
  * If DGL cannot be installed, TxGNN is skipped (split generation still works with a DGL stub) and only
    BioPathNet++ runs, so it can be compared with TxGNN's published numbers on the same protocol.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

TX_COMMIT = "0e0203c3939f7d12da27aaf10e55daaff5424023"   # main as of 2026-10-20
TX_TARBALL = f"https://codeload.github.com/mims-harvard/TxGNN/tar.gz/{TX_COMMIT}"
TX_PYTHON = "3.11"
TX_TORCH = "torch==2.4.0"
TX_DGL = "dgl==2.4.0+cu124"
TX_WHEELS = {"torch": "https://download.pytorch.org/whl/cu124", "dgl": "https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html"}
SPLIT = "autoimmune"
SPLIT_RANDOM_SEED = 0          # numpy global seed for TxGNN's unseeded random test edges
TX_SEED = 1                    # TxGNN train/valid split seed (their reproduce runs used 1-5)
OUR_SEED = 0
PRETRAIN_MAX_H = 3.0
FINETUNE_EPOCHS = 500          # official reproduce/train.py
SESSION_H = 11.0

INPUT = os.environ.get("BPP_INPUT", "/kaggle/input")
WORK = Path(os.environ.get("BPP_WORK", "/kaggle/working")) / "txgnn_h2h"
SCRATCH = Path(os.environ.get("BPP_SCRATCH", "/tmp/txgnn_h2h"))    # big files, NOT saved as notebook output
LOGS, SCRIPTS, OUT = WORK / "logs", WORK / "scripts", WORK / "results"
KG = WORK / "kg_txgnn_autoimmune"
EXPORT, TXDATA, TXREPO, TXENV = SCRATCH / "export", SCRATCH / "txdata", SCRATCH / "TxGNN", SCRATCH / "txenv"
TX_PY = TXENV / "bin" / "python"
T0 = time.time()


def hours_left():
    return SESSION_H - (time.time() - T0) / 3600


def sh(cmd, log=None, env=None, timeout=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    if log is None:
        return subprocess.run(list(map(str, cmd)), env=env, timeout=timeout).returncode
    with open(log, "a") as f:
        try:
            return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n### TIMEOUT after {timeout} s\n")
            return -9


def tail(log, n=25):
    try:
        print("".join(open(log).readlines()[-n:]), flush=True)
    except OSError:
        pass


# ------------------------------------------------------------------ helper scripts (run as subprocesses)
COMPAT = r'''
# --- compatibility shims for TxGNN on current pandas / DGL (applied before importing txgnn) ---
import os, time, types, sys
import pandas as pd
if not hasattr(pd.DataFrame, "append"):          # pandas 2 removed DataFrame.append (TxGNN uses it)
    def _append(self, other, ignore_index=False, **kw):
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _append
if os.environ.get("TX_DGL_STUB") == "1":        # split generation only: TxGNN imports dgl at module level
    import torch                                 # import before the stub is in sys.modules
    class _Any(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):            # inspect/importlib probe __file__, __path__, ...
                raise AttributeError(name)
            return _Any(name)
        def __call__(self, *a, **k):
            raise RuntimeError("DGL stub: no graph operations available")
    for m in ("dgl", "dgl.function", "dgl.ops", "dgl.data", "dgl.data.utils", "dgl.dataloading", "dgl.nn"):
        sys.modules[m] = _Any(m)
else:
    import dgl
    if not hasattr(dgl.DGLGraph, "number_of_nodes"):
        dgl.DGLGraph.number_of_nodes = dgl.DGLGraph.num_nodes
        dgl.DGLGraph.number_of_edges = dgl.DGLGraph.num_edges
    if not hasattr(dgl.dataloading, "EdgeDataLoader"):   # removed in DGL 1.0; same batches via the new API
        _MAX_H = float(os.environ.get("TX_PRETRAIN_MAX_H", "1e9"))
        class EdgeDataLoader:
            def __init__(self, g, eids, sampler, negative_sampler=None, **kw):
                kw.pop("exclude", None); kw.pop("reverse_etypes", None)
                s = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)
                self._dl = dgl.dataloading.DataLoader(g, eids, s, **kw)
            def __len__(self):
                return len(self._dl)
            def __iter__(self):
                t0 = time.time()
                for i, batch in enumerate(self._dl):
                    if (time.time() - t0) / 3600 > _MAX_H:
                        print(f"### PRETRAIN CAP: stopped after {i} of {len(self._dl)} steps ({_MAX_H} h)", flush=True)
                        return
                    yield batch
        dgl.dataloading.EdgeDataLoader = EdgeDataLoader
'''

DGL_TEST = r'''
import sys, torch, dgl, dgl.function as fn
from dgl.ops import edge_softmax
g = dgl.heterograph({("a", "r", "b"): (torch.tensor([0, 1]), torch.tensor([1, 0]))})
# TxGNN pretraining samples minibatches from a CPU graph ...
s = dgl.dataloading.as_edge_prediction_sampler(dgl.dataloading.MultiLayerFullNeighborSampler(1))
next(iter(dgl.dataloading.DataLoader(g, {"r": torch.tensor([0, 1])}, s, batch_size=2)))
# ... and runs full-graph message passing on the GPU
g = g.to("cuda")
g.nodes["a"].data["h"] = torch.ones(2, 4, device="cuda")
g.multi_update_all({"r": (fn.copy_u("h", "m"), fn.mean("m", "h"))}, "sum")
assert torch.allclose(g.nodes["b"].data["h"], torch.ones(2, 4, device="cuda"))
print("DGL_OK", dgl.__version__, torch.__version__, sys.version.split()[0])
'''

SPLIT_SCRIPT = COMPAT + r'''
import random, json, hashlib
import numpy as np, torch
data, export, split, seed, rseed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
random.seed(rseed); np.random.seed(rseed); torch.manual_seed(rseed)
from txgnn import TxData
from txgnn.utils import process_disease_area_split
tx = TxData(data_folder_path=data)                      # downloads kg.csv, node.csv, edges.csv (Dataverse)
np.random.seed(rseed)                                   # TxGNN's random test edges use the global RNG
try:
    tx.prepare_split(split=split, seed=seed)
    df, tr, va, te = tx.df, tx.df_train, tx.df_valid, tx.df_test
except RuntimeError as e:                               # DGL stub: graph construction fails after the split files exist
    print("graph construction skipped:", e)
    df = pd.read_csv(os.path.join(data, split + "_kg", "kg_directed.csv"))
    sd = os.path.join(data, f"{split}_{seed}")
    tr, va, te = (pd.read_csv(os.path.join(sd, f)) for f in ("train.csv", "valid.csv", "test.csv"))
    te = process_disease_area_split(data, df, te, split)
os.makedirs(export, exist_ok=True)
cols = ["x_type", "x_id", "relation", "y_type", "y_id", "x_idx", "y_idx"]
for name, frame in (("kg_directed", df), ("train", tr), ("valid", va), ("test_processed", te)):
    frame[cols].to_csv(os.path.join(export, name + ".csv"), index=False)
info = {n: int(len(f)) for n, f in (("kg_directed", df), ("train", tr), ("valid", va), ("test_processed", te))}
info["test_diseases_indication"] = int(te[te.relation == "rev_indication"].x_idx.nunique())
info["md5"] = {n: hashlib.md5(open(os.path.join(export, n + ".csv"), "rb").read()).hexdigest()
               for n in ("kg_directed", "train", "valid", "test_processed")}
json.dump(info, open(os.path.join(export, "split_info.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
'''

TX_TRAIN = COMPAT + r'''
import json, pickle, numpy as np
data, split, seed, out, ft = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
from txgnn import TxData, TxGNN, TxEval
tx = TxData(data_folder_path=data)
tx.prepare_split(split=split, seed=seed)                 # loads the cached split from stage 2
m = TxGNN(data=tx, weight_bias_track=False, proj_name="h2h", exp_name=f"TxGNN_{seed}_{split}", device="cuda:0")
m.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3, attention=False,
                   sim_measure="all_nodes_profile", agg_measure="rarity")
t = time.time(); m.pretrain(n_epoch=1, learning_rate=1e-3, batch_size=1024, train_print_per_n=20)
print(f"### pretrain {(time.time()-t)/3600:.2f} h", flush=True)
t = time.time(); m.finetune(n_epoch=ft, learning_rate=5e-4, train_print_per_n=5, valid_per_n=20)
print(f"### finetune {(time.time()-t)/3600:.2f} h", flush=True)
ev = TxEval(model=m)
raw = ev.eval_disease_centric(disease_idxs="test_set", show_plot=False, verbose=True, save_result=False,
                              return_raw=True, simulate_random=False)
os.makedirs(out, exist_ok=True)
pickle.dump({"prediction": raw["prediction"]["rev_indication"], "label": raw["label"]["rev_indication"]},
            open(os.path.join(out, "txgnn_raw_indication.pkl"), "wb"))
json.dump({k: [[r[1], float(r[2]), float(r[3])] for r in v] for k, v in raw["summary"].items()},
          open(os.path.join(out, "txgnn_official_summary.json"), "w"), indent=1)
print("### TxGNN done", flush=True)
'''

ALIGN = r'''
# TxGNN's predictions (keyed by TxGNN ids) -> score matrix aligned to the converted KG (bpp.txgnn_eval input)
import sys, json, pickle, re, numpy as np
sys.path.insert(0, sys.argv[4])
from bpp.kg import CompactKG
kg = CompactKG.load(sys.argv[1]); raw = pickle.load(open(sys.argv[2], "rb"))["prediction"]
ids = {}
for i, n in enumerate(kg.node_names):
    m = re.search(r"\[([^:\]]+):([^\]]+)\]$", n)
    if m: ids[(m.group(1), m.group(2))] = i
drugs = kg.drug_ids(); col = {int(d): j for j, d in enumerate(drugs)}
proto = json.load(open(sys.argv[1] + "/txgnn_protocol.json"))
rows, dis, missing = [], [], 0
for d in proto["eval_diseases"]:
    did = re.search(r"\[disease:([^\]]+)\]$", kg.node_names[d]).group(1)
    if did not in raw:
        missing += 1; continue
    s = np.full(len(drugs), -np.inf)
    for drug_id, v in raw[did].items():
        g = ids.get(("drug", str(drug_id)))
        if g is not None: s[col[g]] = float(v)
    rows.append(s); dis.append(d)
s = np.stack(rows); s[np.isinf(s)] = np.nanmin(s[np.isfinite(s)]) - 1
np.savez_compressed(sys.argv[3], drugs=drugs, diseases=np.array(dis), names=np.array([kg.node_names[d] for d in dis]), scores=s)
print(f"aligned {len(dis)} diseases, {missing} missing in TxGNN output")
'''


def write_scripts():
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    for name, body in (("dgl_test.py", DGL_TEST), ("split_export.py", SPLIT_SCRIPT), ("tx_train.py", TX_TRAIN), ("align.py", ALIGN)):
        (SCRIPTS / name).write_text(textwrap.dedent(body))


def locate_bpp():
    pkgs = glob.glob(f"{INPUT}/**/bpp/build_txgnn.py", recursive=True)
    assert pkgs, "biopathnet-pp-v3 dataset with bpp/build_txgnn.py not found: upload the new dataset version"
    code = WORK / "code"
    shutil.copytree(Path(pkgs[0]).parent, code / "bpp", dirs_exist_ok=True)
    return str(code)


def fetch_txgnn(log):
    """TxGNN source at the pinned commit (GitHub tarball: one request, no git needed)."""
    if (TXREPO / "txgnn" / "TxGNN.py").exists():
        return
    import tarfile
    import urllib.request
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tgz = SCRATCH / "txgnn.tar.gz"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(TX_TARBALL, tgz)
            break
        except OSError as e:
            print(f"TxGNN download attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(15)
    else:
        raise RuntimeError(f"could not download {TX_TARBALL}")
    with tarfile.open(tgz) as t:
        top = t.getmembers()[0].name.split("/")[0]
        t.extractall(SCRATCH, filter="data")
    shutil.rmtree(TXREPO, ignore_errors=True)
    shutil.move(str(SCRATCH / top), str(TXREPO))
    for need in ("txgnn/TxGNN.py", f"data/disease_files/{SPLIT}.csv", "txgnn/data_splits/HumanDO.obo"):
        assert (TXREPO / need).exists(), f"TxGNN download incomplete: {need} missing (see install.log)"
    print("TxGNN source:", TXREPO, flush=True)


def install(tx_env):
    """Python 3.11 env for TxGNN (DGL has no wheels for Python >= 3.13). Returns True if DGL works on the GPU."""
    log = LOGS / "install.log"
    fetch_txgnn(log)
    py = sys.executable
    if not TX_PY.exists():
        sh([py, "-m", "pip", "install", "-q", "uv"], log)
        uv = [py, "-m", "uv"]
        sh(uv + ["venv", "--python", TX_PYTHON, TXENV], log)          # uv fetches a standalone CPython if needed
        pip = uv + ["pip", "install", "--python", TX_PY]
        sh(pip + [TX_TORCH, "--index-url", TX_WHEELS["torch"]], log)
        sh(pip + ["numpy<2", "pandas<2.2", "scipy", "scikit-learn", "matplotlib", "seaborn", "tqdm", "requests",
                  "networkx", "psutil", "pydantic", "pyyaml", "packaging", "torch_geometric", "goatools"], log)
        sh(pip + ["--no-deps", TX_DGL, "-f", TX_WHEELS["dgl"]], log)
        sh(pip + ["--no-deps", "torchdata==0.8.0"], log)                # older graphbolt builds import it
    r = subprocess.run([str(TX_PY), str(SCRIPTS / "dgl_test.py")], capture_output=True, text=True, env=tx_env) \
        if TX_PY.exists() else None
    if r is not None:
        with open(log, "a") as f:
            f.write(r.stdout + r.stderr)
    if r is not None and "DGL_OK" in r.stdout:
        print(r.stdout.strip(), flush=True)
        return True
    print("DGL could not be installed: TxGNN will be skipped (see install.log); split via DGL stub", flush=True)
    tail(log, 40)
    sh([py, "-m", "pip", "install", "-q", "torch_geometric", "goatools"], log)   # stub split runs on Kaggle's Python
    return False


def check_internet():
    import urllib.error
    import urllib.request
    for url in ("https://codeload.github.com", "https://pypi.org/simple/", "https://dataverse.harvard.edu"):
        try:
            urllib.request.urlopen(url, timeout=15)
        except urllib.error.HTTPError:
            pass                    # the server answered (e.g. 404): the network is fine
        except Exception as e:      # noqa: BLE001 - DNS/connection failure: no internet
            raise SystemExit(f"\n*** NO INTERNET ({url}: {e}).\n*** In the notebook editor: Settings (right panel) -> "
                             "Internet -> ON (needs a phone-verified Kaggle account), then Save Version again.\n")
    print("internet OK", flush=True)


def main():
    check_internet()
    for d in (LOGS, OUT):
        d.mkdir(parents=True, exist_ok=True)
    write_scripts()
    code = locate_bpp()
    # our model: Kaggle's Python + bpp code. TxGNN: its own Python 3.11 env, which must NOT see Kaggle's
    # (Python 3.13) PYTHONPATH entries, only the TxGNN source.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(TXREPO), code, os.environ.get("PYTHONPATH", "")]),
               TX_PRETRAIN_MAX_H=str(PRETRAIN_MAX_H))
    tx_env = {k: v for k, v in env.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    tx_env.update(PYTHONPATH=str(TXREPO), PYTHONNOUSERSITE="1")
    if shutil.which("nvidia-smi"):
        sh(["nvidia-smi"])
    dgl_ok = install(tx_env)
    py = sys.executable
    tx_py = str(TX_PY) if dgl_ok else py

    # stage 2: TxGNN's own split (stub DGL if the real one is unavailable)
    split_env = dict(tx_env if dgl_ok else env, TX_DGL_STUB="0" if dgl_ok else "1")
    # process_disease_area_split reads <data>/disease_files/<split>.csv, which ships with the repo, not Dataverse
    shutil.copytree(TXREPO / "data" / "disease_files", TXDATA / "disease_files", dirs_exist_ok=True)
    if not (EXPORT / "split_info.json").exists():
        rc = sh([tx_py, SCRIPTS / "split_export.py", TXDATA, EXPORT, SPLIT, TX_SEED, SPLIT_RANDOM_SEED], LOGS / "split.log", split_env)
        tail(LOGS / "split.log")
        assert rc == 0, "split generation failed: see split.log"

    # stage 3: convert
    if not (KG / "splits.json").exists():
        rc = sh([py, "-m", "bpp.build_txgnn", "--export", EXPORT, "--out", KG, "--names", TXDATA / "kg.csv"],
                LOGS / "convert.log", env)
        tail(LOGS / "convert.log")
        assert rc == 0, "conversion failed: see convert.log"

    # stage 4: train both in parallel
    def run_txgnn():
        if not dgl_ok or (OUT / "txgnn_raw_indication.pkl").exists():
            return
        rc = sh([tx_py, SCRIPTS / "tx_train.py", TXDATA, SPLIT, TX_SEED, OUT, FINETUNE_EPOCHS], LOGS / "txgnn_train.log",
                dict(tx_env, CUDA_VISIBLE_DEVICES="0"), timeout=max(600, int(3600 * (hours_left() - 0.5))))
        print("TxGNN rc", rc, flush=True); tail(LOGS / "txgnn_train.log")

    run_dir = WORK / "runs" / f"ours_s{OUR_SEED}"

    def run_ours():
        if (run_dir / "results.json").exists():
            return
        rc = sh([py, "-m", "bpp.train", "--data", KG, "--split", "splits.json", "--out", run_dir, "--device", "cuda:0",
                 "--prior", "--seed", OUR_SEED, "--dim", "32", "--layers", "6", "--batch", "8", "--epochs", "20",
                 "--patience", "6", "--time-budget-h", f"{hours_left() - 0.75:.2f}"],
                LOGS / "ours_train.log", dict(env, CUDA_VISIBLE_DEVICES="1"))
        print("ours rc", rc, flush=True); tail(LOGS / "ours_train.log")

    threads = [threading.Thread(target=run_txgnn), threading.Thread(target=run_ours)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(600)
        for lg in ("txgnn_train.log", "ours_train.log"):
            try:
                print(f"  {lg:18s} | {open(LOGS / lg).read().strip().splitlines()[-1][:140]}", flush=True)
            except (OSError, IndexError):
                pass
    for t in threads:
        t.join()

    # stage 5: evaluate
    args = []
    if (run_dir / "best.pt").exists():      # also when training stopped on the time budget (no results.json)
        sh([py, "-m", "bpp.score", "--run", run_dir, "--data", KG, "--sets", "area", "--device", "cuda:0",
            "--out", OUT / f"ours_s{OUR_SEED}.npz"], LOGS / "eval.log", env)
        args += ["--ours", OUT / f"ours_s{OUR_SEED}.npz"]
    if (OUT / "txgnn_raw_indication.pkl").exists():
        sh([py, SCRIPTS / "align.py", KG, OUT / "txgnn_raw_indication.pkl", OUT / f"txgnn_s{TX_SEED}.npz", code],
           LOGS / "eval.log", env)
        args += ["--txgnn", OUT / f"txgnn_s{TX_SEED}.npz"]
    if args:
        sh([py, "-m", "bpp.txgnn_eval", "--kg", KG, *args, "--out", OUT / "h2h.json"], LOGS / "eval.log", env)
    tail(LOGS / "eval.log", 40)
    for f in ("splits.json", "txgnn_protocol.json", "meta.json"):
        if (KG / f).exists():
            shutil.copy(KG / f, OUT / f)
    shutil.copy(EXPORT / "split_info.json", OUT / "split_info.json")
    (OUT / "settings.json").write_text(json.dumps({
        "tx_commit": TX_COMMIT, "split": SPLIT, "split_random_seed": SPLIT_RANDOM_SEED, "tx_seed": TX_SEED,
        "our_seed": OUR_SEED, "pretrain_max_h": PRETRAIN_MAX_H, "finetune_epochs": FINETUNE_EPOCHS, "dgl_ok": dgl_ok,
        "ours_finished": (run_dir / "results.json").exists()}, indent=1))
    for f in ("train_log.jsonl", "results.json"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, OUT / f"ours_{f}")
    notes = []
    for lg in ("txgnn_train.log", "ours_train.log", "eval.log"):
        try:
            notes += [f"[{lg}] {l.rstrip()}" for l in open(LOGS / lg)
                      if l.startswith(("###", "aligned", "ours ", "txgnn ", "early stopping", "time budget", "epoch "))]
        except OSError:
            pass
    (OUT / "summary.md").write_text("# TxGNN head-to-head\n\n```\n" + (OUT / "settings.json").read_text() + "\n```\n\n```\n"
                                    + "\n".join(notes) + "\n```\n")
    print((OUT / "summary.md").read_text())
    print("done; download /kaggle/working/txgnn_h2h/results and logs", flush=True)


main()

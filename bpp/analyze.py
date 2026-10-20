"""
Aggregate every finished run into the numbers and LaTeX tables used by the papers.

  python -m bpp.analyze                      # reads outputs/bpp/kaggle_session*/, writes:
    outputs/bpp/analysis/summary.json        all aggregated numbers (seed mean/sd, CIs, tests)
    outputs/paper/v3/generated/numbers.tex   \newcommand macros for numbers quoted in the text
    outputs/paper/v3/generated/tab_*.tex     tables included by both papers

Nothing here is typed by hand: re-run after each Kaggle session and recompile the papers.
Later sessions override earlier ones for the same run name (they resume the same run).

Statistics
  * per-disease metrics are first averaged over seeds, then macro-averaged over diseases;
  * 95% CI = percentile bootstrap over diseases (1000 resamples) of the seed-averaged values;
  * model-vs-model: one-sided paired bootstrap over diseases (P[mean diff <= 0]) and the
    two-sided Wilcoxon signed-rank test on the same paired per-disease values.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from bpp.metrics import paired_bootstrap

GROUPS = {  # run-name prefix -> (label, split protocol)
    "main": ("BioPathNet++ (paths + prior)", "targets"),
    "paths": ("Paths only (NBFNet)", "targets"),
    "prioronly": ("Prior only (transfer channel)", "targets"),
    "brg_main": ("BioPathNet++, BRG relations", "targets"),
    "bce_main": ("BioPathNet++, BCE loss", "targets"),
    "noauto_main": ("BioPathNet++, no autoimmune training", "no_autoimmune"),
    "noauto_paths": ("Paths only, no autoimmune training", "no_autoimmune"),
    # transfer curve (bpp/extra_splits.py); "tr_Actl" pools the matched-control draws (draw j = "seed" j)
    "tr_A": ("A: RA/SLE/MS families only", "transfer"),
    "tr_Actl": ("A$'$: matched non-autoimmune controls", "transfer"),
    "tr_CE": ("C$_E$: all diseases except E", "transfer"),
    "tr_A_paths": ("A, paths only (no transfer channel)", "transfer"),
    "tr_Actl_paths": ("A$'$, paths only (no transfer channel)", "transfer"),
    "test_main": ("BioPathNet++ (selection-free test split)", "test"),
}
# extra baseline files written by bpp.run_baselines for the derived splits
EXTRA_BASELINES = {"tr_A": "baselines_tr_A.json", "tr_CE": "baselines_tr_CE.json", "test": "baselines_test.json",
                   **{f"tr_Actl{j}": f"baselines_tr_Actl{j}.json" for j in range(3)}}
BASELINES = {"random": "Random", "drug_degree": "Drug degree", "popularity": "Popularity",
             "autoimmune_frequency": "Autoimmune frequency", "shared_targets": "Shared targets",
             "disease_knn": "Disease kNN"}
METRICS = ["mrr", "hits@1", "hits@10", "hits@50", "hits@100", "auroc", "auprc", "auroc_hard",
           "auroc_degmatch", "mean_rank"]
TARGET_SHORT = {"Primary Biliary Cholangitis": "PBC", "Systemic Sclerosis": "SSc",
                "Sjogren Syndrome": "Sj\\\"ogren", "Autoimmune Hepatitis": "AIH",
                "Myasthenia Gravis": "MG", "Crohn Disease": "Crohn"}


# ----------------------------------------------------------------------------- loading
def collect_runs(root: Path) -> dict:
    """run name -> run directory (latest session wins)."""
    runs = {}
    for sess in sorted(root.glob("kaggle_session*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        for res in (sess / "runs").glob("*/results.json"):
            if not res.parent.name.startswith("_"):
                runs[res.parent.name] = res.parent
    return runs


def group_runs(runs: dict) -> dict:
    g = defaultdict(dict)
    for name, path in runs.items():
        m = re.match(r"(.+)_s(\d+)$", name)
        if not m:
            continue
        grp, key = m.group(1), int(m.group(2))
        c = re.match(r"tr_Actl(\d+)(_paths)?$", grp)
        if c:                                   # control draws pooled: draw index stands in for the seed
            grp, key = "tr_Actl" + (c.group(2) or ""), int(c.group(1)) * 100 + key
        if grp in GROUPS:
            g[grp][key] = json.loads((path / "results.json").read_text())
    return {k: dict(sorted(v.items())) for k, v in g.items()}


def seed_avg_per_disease(results_by_seed: dict, set_name: str) -> dict:
    """{disease: {metric: mean over seeds}} (+ '_any10': share of seeds w/ a known drug in top 10)."""
    per = defaultdict(lambda: defaultdict(list))
    for res in results_by_seed.values():
        if set_name not in res:
            continue
        for d, m in res[set_name]["per_disease"].items():
            for k in METRICS:
                if k in m:
                    per[d][k].append(m[k])
            per[d]["_any10"].append(float(min(m["_ranks"]) <= 10))
            per[d]["n_pos"] = [m["n_pos"]]
    return {d: {k: float(np.mean(v)) for k, v in mm.items()} for d, mm in per.items()}


def macro(per: dict, key: str, n_boot=1000, seed=0):
    v = np.array([m[key] for m in per.values() if key in m and not np.isnan(m[key])])
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = [rng.choice(v, len(v)).mean() for _ in range(n_boot)] if len(v) > 1 else [v[0]]
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(len(v))}


def seed_stats(results_by_seed: dict, set_name: str, key: str):
    vals = [r[set_name]["aggregate"][key]["mean"] for r in results_by_seed.values()
            if set_name in r and key in r[set_name]["aggregate"]]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "seeds": vals}


def compare(per_a: dict, per_b: dict, key="mrr"):
    pb = paired_bootstrap(per_a, per_b, key=key)
    names = [n for n in per_a if n in per_b]
    a = np.array([per_a[n][key] for n in names]); b = np.array([per_b[n][key] for n in names])
    try:
        w = wilcoxon(a, b).pvalue if np.any(a != b) and len(names) >= 5 else float("nan")
    except ValueError:
        w = float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(names), (5000, len(names)))
    diffs = (a[idx] - b[idx]).mean(1)
    return {"mean_diff": pb["mean_diff"], "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_boot_one_sided": pb["p_a_not_better"], "p_wilcoxon_two_sided": float(w), "n": pb["n"],
            "wins": int(np.sum(a > b)), "ties": int(np.sum(a == b)), "losses": int(np.sum(a < b))}


def run_info(run_dir: Path) -> dict:
    """Graph size, last logged step time and peak GPU memory, parsed from train.log."""
    info = {}
    log = run_dir / "train.log"
    if not log.exists():
        return info
    for line in log.read_text(errors="ignore").splitlines():
        m = re.match(r"graph: [\d,]+ nodes, ([\d,]+) context edges", line)
        if m:
            info["edges"] = int(m.group(1).replace(",", ""))
        m = re.search(r"([\d.]+)s/step peak_mem ([\d.]+)GB", line)
        if m:
            info["s_per_step"], info["peak_gb"] = float(m.group(1)), float(m.group(2))
    res = run_dir / "results.json"
    if res.exists():
        info["best_epoch"] = json.loads(res.read_text()).get("best_epoch")
    return info


def selection_gap(run_dir: Path):
    log = run_dir / "train_log.jsonl"
    if not log.exists():
        return None
    m = [json.loads(l)["valid_mrr"] for l in log.read_text().splitlines() if l.strip()]
    return {"best": max(m), "last": m[-1], "gap_best_minus_last": max(m) - m[-1], "epochs": len(m)}


# ----------------------------------------------------------------------------- analysis
def analyse(root: Path, data: Path) -> dict:
    runs = collect_runs(root)
    groups = group_runs(runs)
    base = {"targets": json.loads((root / "baselines_targets.json").read_text()),
            "no_autoimmune": json.loads((root / "baselines_noauto.json").read_text())}
    out = {"runs_found": sorted(runs), "models": {}, "baselines": {}, "tests": {}, "selection_gap": {}}

    per = {}  # (group, set) -> seed-averaged per-disease
    for g, by_seed in groups.items():
        label, proto = GROUPS[g]
        out["models"][g] = {"label": label, "protocol": proto, "seeds": sorted(by_seed), "sets": {}}
        for s in ("valid", "targets", "area", "test"):
            if not any(s in r for r in by_seed.values()):
                continue
            per[(g, s)] = seed_avg_per_disease(by_seed, s)
            out["models"][g]["sets"][s] = {
                "seed_mean_sd": {k: seed_stats(by_seed, s, k) for k in METRICS},
                "macro_seedavg": {k: macro(per[(g, s)], k) for k in METRICS + ["_any10"]},
                "per_disease": per[(g, s)]}
        out["models"][g]["best_epochs"] = [r.get("best_epoch") for r in by_seed.values()]
    for name, path in runs.items():
        sg = selection_gap(path)
        if sg:
            out["selection_gap"][name] = sg

    for proto, b in base.items():
        out["baselines"][proto] = {}
        for bname, sets in b.items():
            out["baselines"][proto][bname] = {}
            for s, r in sets.items():
                pd_ = {d: {**{k: m[k] for k in METRICS if k in m}, "_any10": float(min(m["_ranks"]) <= 10),
                           "n_pos": m["n_pos"]} for d, m in r["per_disease"].items()}
                per[(f"base:{proto}:{bname}", s)] = pd_
                out["baselines"][proto][bname][s] = {"macro": {k: macro(pd_, k) for k in METRICS + ["_any10"]}}
    for key, fname in EXTRA_BASELINES.items():
        if (root / fname).exists():
            for bname, sets in json.loads((root / fname).read_text()).items():
                for s, r in sets.items():
                    per[(f"base:{key}:{bname}", s)] = {
                        d: {**{k: m[k] for k in METRICS if k in m}, "_any10": float(min(m["_ranks"]) <= 10),
                            "n_pos": m["n_pos"]} for d, m in r["per_disease"].items()}

    def test(name, a, b, key="mrr"):
        if a in per and b in per:
            out["tests"][name] = compare(per[a], per[b], key)

    knn_t, knn_n = "base:targets:disease_knn", "base:no_autoimmune:disease_knn"
    for s in ("valid", "targets"):
        test(f"main_vs_knn_{s}", ("main", s), (knn_t, s))
        test(f"paths_vs_knn_{s}", ("paths", s), (knn_t, s))
        test(f"main_vs_paths_{s}", ("main", s), ("paths", s))
        test(f"main_vs_prioronly_{s}", ("main", s), ("prioronly", s))
        test(f"main_vs_brg_{s}", ("main", s), ("brg_main", s))
        test(f"main_vs_bce_{s}", ("main", s), ("bce_main", s))
        test(f"main_vs_knn_{s}_hits10", ("main", s), (knn_t, s), "hits@10")
    test("main_vs_noauto_targets", ("main", "targets"), ("noauto_main", "targets"))
    test("noauto_vs_knn_area", ("noauto_main", "area"), (knn_n, "area"))
    test("noauto_vs_knn_targets", ("noauto_main", "targets"), (knn_n, "targets"))
    test("noauto_main_vs_paths_area", ("noauto_main", "area"), ("noauto_paths", "area"))

    # same-seed comparisons for single-seed ablations (robustness check vs the seed-averaged tests)
    def test_runs(name, a, b, s, key="mrr"):
        if a in runs and b in runs:
            ra = json.loads((runs[a] / "results.json").read_text())
            rb = json.loads((runs[b] / "results.json").read_text())
            if s in ra and s in rb:
                out["tests"][name] = compare(ra[s]["per_disease"], rb[s]["per_disease"], key)
    test_runs("s0_main_vs_bce_valid", "main_s0", "bce_main_s0", "valid")
    test_runs("s0_main_vs_brg_valid", "main_s0", "brg_main_s0", "valid")
    test_runs("s0_noauto_main_vs_paths_area", "noauto_main_s0", "noauto_paths_s0", "area")

    out["run_info"] = {name: run_info(path) for name, path in runs.items()}
    out["sparsity"] = sparsity(per, data)
    out["case_studies"] = case_studies(runs)
    out["metrics_table"] = metrics_table(per, groups, data)
    out["validation"] = load_validation(root)
    out["transfer"] = transfer_curve(per, data, out["tests"])
    out["transfer"]["last_checkpoint"] = transfer_last_checkpoint(root, out.get("transfer", {}), out["tests"])
    out["test_split"] = test_split(per, out["tests"])
    out["txgnn_h2h"] = load_txgnn_h2h(root)
    return out


# TxGNN's own reproduce/result_more_metrics.csv (commit 0e0203c), autoimmune split, indication, 5 seeds:
# TxGNN Recall@100 0.338 +- 0.004. Its AUROC/AUPRC there use sampled balanced negatives -> not comparable.
TXGNN_PUBLISHED_RECALL100 = 0.338


def load_txgnn_h2h(root: Path) -> dict:
    """Head-to-head on TxGNN's autoimmune split (kaggle/run_txgnn.py): copy its results/ to outputs/bpp/txgnn_h2h/."""
    d = root / "txgnn_h2h"
    if not (d / "h2h.json").exists():
        return {}
    h = json.loads((d / "h2h.json").read_text())
    out = {"settings": json.loads((d / "settings.json").read_text()) if (d / "settings.json").exists() else {},
           "split_info": json.loads((d / "split_info.json").read_text()) if (d / "split_info.json").exists() else {},
           "paired": h.get("paired", {}), "paired_full": h.get("paired_full", {})}
    for who in ("ours", "txgnn"):
        if h.get(who):
            out[who] = next(iter(h[who].values()))["summary"]
    off = d / "txgnn_official_summary.json"
    if off.exists():
        rows = json.loads(off.read_text()).get("rev_indication", [])
        out["txgnn_official"] = {name: {"mean": m, "sd": sd} for name, m, sd in rows}
    proto = json.loads((d / "txgnn_protocol.json").read_text()) if (d / "txgnn_protocol.json").exists() else {}
    if proto:
        out["n_candidates"], out["n_test_pairs"] = len(proto["candidate_drugs"]), len(proto["test_pairs"])
        # share of each model's full-ranking top 100 that are drugs with >= 1 indication (TxGNN's candidate set)
        for who, f in (("ours", "ours_s0.npz"), ("txgnn", "txgnn_s1.npz")):
            if (d / f).exists():
                z = np.load(d / f)
                cand = np.isin(z["drugs"], proto["candidate_drugs"])
                out[f"{who}_top100_indicated_share"] = float(np.mean([cand[np.argsort(-s)[:100]].mean() for s in z["scores"]]))
    if (d / "splits.json").exists():
        sp = json.loads((d / "splits.json").read_text())
        out["n_train_diseases"], out["n_valid_diseases"] = len(sp["train_diseases"]), len(sp["valid_diseases"])
    if (d / "coverage.json").exists():
        out["coverage"] = json.loads((d / "coverage.json").read_text())
    log = d / "logs" / "txgnn_train.log"
    if log.exists():
        for stage in ("pretrain", "finetune"):
            m = re.search(rf"### {stage} ([\d.]+) h", log.read_text(errors="replace"))
            if m:
                out[f"txgnn_{stage}_h"] = float(m.group(1))
        out["txgnn_pretrain_capped"] = "### PRETRAIN CAP" in log.read_text(errors="replace")
    return out


# ----------------------------------------------------------------------------- reader-friendly metrics table
TABLE_METRICS = [("_any10", "Any@10"), ("hits@1", "H@1"), ("hits@10", "H@10"), ("hits@50", "H@50"), ("auroc", "AUROC"),
                 ("auroc_hard", r"AUROC$_\text{hard}$"), ("auroc_degmatch", r"AUROC$_\text{deg}$"), ("auprc", "AUPRC"),
                 ("mrr", "MRR"), ("mean_rank", "Mean rank")]


def metrics_table(per: dict, groups: dict, data: Path) -> dict:
    """Model vs baselines on test / valid / targets for every metric, plus the random references a
    non-specialist needs: expected any-hit@10 and AUPRC (= prevalence) of a random ranking, and the
    median rank of a known drug."""
    from bpp.kg import CompactKG
    n_drugs = len(CompactKG.load(str(data)).drug_ids())
    sets = {"test": (("test_main", "test"), "base:test"), "valid": (("main", "valid"), "base:targets"),
            "targets": (("main", "targets"), "base:targets")}
    out = {"n_drugs": n_drugs}
    for s, (mkey, bpref) in sets.items():
        rows = {}
        if mkey in per:
            rows["model"] = {k: (macro(per[mkey], k) or {}).get("mean") for k, _ in TABLE_METRICS}
        for b in ("random", "popularity", "disease_knn"):
            if (f"{bpref}:{b}", s) in per:
                rows[b] = {k: (macro(per[(f"{bpref}:{b}", s)], k) or {}).get("mean") for k, _ in TABLE_METRICS}
        ref = per.get(mkey) or per.get((f"{bpref}:random", s)) or {}
        npos = np.array([m["n_pos"] for m in ref.values()], float)
        if len(npos):
            exp_any10 = np.mean([1 - np.prod([(n_drugs - n - i) / (n_drugs - i) for i in range(10)]) for n in npos])
            out[s] = {"rows": rows, "n_diseases": int(len(npos)), "mean_positives": float(npos.mean()),
                      "prevalence": float(np.mean(npos / n_drugs)), "random_expected_any10": float(exp_any10)}
        grp = groups.get(mkey[0], {})
        ranks = [r for res in grp.values() if s in res for m in res[s]["per_disease"].values() for r in m["_ranks"]]
        if ranks and s in out:
            ranks = np.array(ranks)
            out[s]["median_rank_known_drug"] = float(np.median(ranks))
            out[s]["share_known_drugs_rank1"] = float(np.mean(ranks <= 1))
    return out


def load_validation(root: Path) -> dict:
    vdir = root / "validation"
    if not (vdir / "validation_summary.json").exists():
        return {}
    v = json.loads((vdir / "validation_summary.json").read_text())
    import pandas as pd
    if (vdir / "temporal_validation.csv").exists():
        v["temporal_rows"] = pd.read_csv(vdir / "temporal_validation.csv").to_dict(orient="records")
    if (vdir / "explanations_main_s0.json").exists():
        v["explanations"] = json.loads((vdir / "explanations_main_s0.json").read_text())
    return v


# ----------------------------------------------------------------------------- transfer curve (Aim 1)
TRANSFER_CONDITIONS = [  # key, label, model group, baseline key(s), split file for the training-set size
    ("Actl", "A$'$: 11 matched non-autoimmune diseases", "tr_Actl", [f"tr_Actl{j}" for j in range(3)],
     "splits_tr_Actl0.json"),
    ("A", "A: RA/SLE/MS families (11 diseases)", "tr_A", ["tr_A"], "splits_tr_A.json"),
    ("D", "D: all non-autoimmune diseases", "noauto_main", ["no_autoimmune"], "splits_noauto.json"),
    ("CE", "C$_E$: all diseases except the test set", "tr_CE", ["tr_CE"], "splits_tr_CE.json"),
]


def _restrict(pd_: dict, names) -> dict:
    return {d: m for d, m in pd_.items() if d in names}


def _avg(pds: list) -> dict:
    """Average several per-disease dicts (e.g. the 3 control draws) disease by disease."""
    common = [d for d in pds[0] if all(d in p for p in pds)]      # keep a stable order: bootstraps are seeded
    return {d: {k: float(np.mean([p[d][k] for p in pds])) for k in pds[0][d] if not k.startswith("_") or k == "_any10"}
            for d in common}


def transfer_curve(per: dict, data: Path, tests: dict) -> dict:
    """MRR on E (held-out autoimmune-area diseases) and on the 6 targets for each training condition,
    model and disease-kNN baseline side by side. Missing runs are reported as pending."""
    split_a = data / "splits_tr_A.json"
    if not split_a.exists():
        return {}
    from bpp.kg import CompactKG
    names = CompactKG.load(str(data)).node_names
    E = {names[d] for d in json.loads(split_a.read_text())["eval_area"]}
    res = {"eval_diseases": sorted(E), "conditions": {}}
    paired = {}
    for key, label, grp, bkeys, split_file in TRANSFER_CONDITIONS:
        sp = json.loads((data / split_file).read_text())
        row = {"label": label, "n_train": len(sp["train_diseases"])}
        for s, sname in (("area", "E"), ("targets", "targets")):
            m = per.get((grp, s))
            if m is not None:
                m = _restrict(m, E) if s == "area" else m
                row[f"model_{sname}"] = macro(m, "mrr")
                paired[(key, sname, "model")] = m
            b = [per.get((f"base:{bk}:disease_knn", s)) for bk in bkeys]
            if all(x is not None for x in b):
                b = _avg([_restrict(x, E) if s == "area" else x for x in b])
                row[f"knn_{sname}"] = macro(b, "mrr")
                paired[(key, sname, "knn")] = b
        res["conditions"][key] = row
    for pa, pc, tname in (("tr_A_paths", "tr_Actl_paths", "tr_A_vs_Actl_paths_E"),):
        if (pa, "area") in per and (pc, "area") in per:
            ma, mc = _restrict(per[(pa, "area")], E), _restrict(per[(pc, "area")], E)
            res["paths_only"] = {"A": macro(ma, "mrr"), "Actl": macro(mc, "mrr")}
            tests[tname] = compare(ma, mc)
    for a, b, sname, kind in (("A", "Actl", "E", "model"), ("A", "Actl", "E", "knn"), ("A", "Actl", "targets", "model"),
                              ("CE", "D", "E", "model"), ("CE", "A", "E", "model"), ("D", "A", "E", "model")):
        if (a, sname, kind) in paired and (b, sname, kind) in paired:
            tests[f"tr_{a}_vs_{b}_{sname}_{kind}"] = compare(paired[(a, sname, kind)], paired[(b, sname, kind)])
    return res


def transfer_last_checkpoint(root: Path, transfer: dict, tests: dict) -> dict:
    """Sensitivity check: A vs A' with each run's FINAL checkpoint (bpp/evaluate_checkpoint.py), because
    these runs early-stop on general validation diseases that do not resemble their training data."""
    files = {p.stem: p for p in root.glob("kaggle_session*/last_ckpt_eval/*.json")}
    a = [files[f"tr_A_s{i}"] for i in range(3) if f"tr_A_s{i}" in files]
    c = [files[f"tr_Actl{j}_s0"] for j in range(3) if f"tr_Actl{j}_s0" in files]
    E = set(transfer.get("eval_diseases", []))
    if not (a and c and E):
        return {}
    def load(paths, s):
        return _avg([json.loads(p.read_text())[s]["per_disease"] for p in paths])
    out = {}
    for s, sname in (("area", "E"), ("targets", "targets")):
        pa, pc = load(a, s), load(c, s)
        if s == "area":
            pa, pc = _restrict(pa, E), _restrict(pc, E)
        out[sname] = {"A": macro(pa, "mrr"), "Actl": macro(pc, "mrr")}
        tests[f"tr_A_vs_Actl_{sname}_last"] = compare(pa, pc)
    return out


def test_split(per: dict, tests: dict) -> dict:
    """Selection-free test set: never trained on, never used for early stopping."""
    out = {}
    if ("test_main", "test") in per:
        out["model"] = {k: macro(per[("test_main", "test")], k) for k in ("mrr", "hits@10", "auroc_hard", "_any10")}
        out["model_valid"] = macro(per[("test_main", "valid")], "mrr") if ("test_main", "valid") in per else None
        if ("base:test:disease_knn", "test") in per:
            tests["test_main_vs_knn_test"] = compare(per[("test_main", "test")], per[("base:test:disease_knn", "test")])
    for b in ("random", "popularity", "disease_knn"):
        if (f"base:test:{b}", "test") in per:
            out[b] = {k: macro(per[(f"base:test:{b}", "test")], k) for k in ("mrr", "hits@10", "auroc_hard", "_any10")}
    return out


def sparsity(per: dict, data: Path) -> dict:
    """Does performance depend on how well-connected (data-rich) a validation disease is?"""
    from bpp.kg import CompactKG
    kg = CompactKG.load(str(data))
    split = json.loads((data / "splits.json").read_text())
    deg = np.bincount(kg.head, minlength=kg.num_nodes) + np.bincount(kg.tail, minlength=kg.num_nodes)
    name2deg = {kg.node_names[d]: int(deg[d]) for d in split["valid_diseases"]}
    res = {}
    for key, label in ((("main", "valid"), "BioPathNet++"), (("base:targets:disease_knn", "valid"), "Disease kNN"),
                       (("paths", "valid"), "Paths only")):
        if key not in per:
            continue
        names = [n for n in per[key] if n in name2deg]
        x = np.array([name2deg[n] for n in names]); y = np.array([per[key][n]["mrr"] for n in names])
        q = np.quantile(x, [1 / 3, 2 / 3])
        bins = {"low": x <= q[0], "mid": (x > q[0]) & (x <= q[1]), "high": x > q[1]}
        res[label] = {"spearman_rho": float(spearmanr(x, y).correlation), "spearman_p": float(spearmanr(x, y).pvalue),
                      "tertile_edges": [float(v) for v in q],
                      "tertile_mrr": {b: float(y[m].mean()) for b, m in bins.items()},
                      "tertile_n": {b: int(m.sum()) for b, m in bins.items()}}
    return res


def case_studies(runs: dict) -> dict:
    out = {}
    for name in ("main_s0", "main_s1", "main_s2"):
        if name in runs and (runs[name] / "target_rankings.json").exists():
            out[name] = json.loads((runs[name] / "target_rankings.json").read_text())
    return out


# ----------------------------------------------------------------------------- LaTeX
def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pfmt(p):
    if p is None or np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ptext(p):
    r"""p-value for running text: '$<$0.001' or '$=$0.012' (always written as  $p$\,\Pwil...)."""
    if p is None or np.isnan(p):
        return "n/a"
    return "$<$0.001" if p < 0.001 else f"$=${p:.3f}"


def cell_model(m, s, k):
    st = m["sets"].get(s, {}).get("seed_mean_sd", {}).get(k)
    if not st:
        return "--"
    return f"{st['mean']:.3f}" + (f"$\\pm${st['sd']:.3f}" if st["sd"] is not None else "")


def table_main(S):
    cols = [("valid", "mrr"), ("valid", "hits@1"), ("valid", "hits@10"), ("valid", "auroc"), ("valid", "auroc_hard"),
            ("targets", "mrr"), ("targets", "hits@10"), ("targets", "auroc_hard")]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{5}{c}{68 zero-shot validation diseases} & \multicolumn{3}{c}{6 rare autoimmune targets}\\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
             r"Method & MRR & H@1 & H@10 & AUROC & AUROC$_{\text{hard}}$ & MRR & H@10 & AUROC$_{\text{hard}}$\\", r"\midrule"]
    for b, lab in BASELINES.items():
        r = S["baselines"]["targets"][b]
        lines.append(lab + " & " + " & ".join(f3((r.get(s, {}).get("macro", {}).get(k) or {}).get("mean"))
                                             for s, k in cols) + r"\\")
    lines.append(r"\midrule")
    for g in ("prioronly", "paths", "main"):
        if g in S["models"]:
            m = S["models"][g]
            lab = m["label"] + f" ($n$={len(m['seeds'])})"
            lab = r"\textbf{" + lab + "}" if g == "main" else lab
            lines.append(lab + " & " + " & ".join(cell_model(m, s, k) for s, k in cols) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_targets(S):
    lines = [r"\begin{tabular}{lcccccl}", r"\toprule",
             r"Target & \#known & kNN & Paths only & BioPathNet++ & Best rank & Top-3 predictions (seed 0)\\",
             r"\midrule"]
    knn = S["baselines"]["targets"]["disease_knn"]["targets"]
    knn_pd = json.loads(json.dumps(S["_knn_targets_pd"]))
    main_pd = S["models"].get("main", {}).get("sets", {}).get("targets", {}).get("per_disease", {})
    paths_pd = S["models"].get("paths", {}).get("sets", {}).get("targets", {}).get("per_disease", {})
    rk = S["case_studies"].get("main_s0", {})
    for t, short in TARGET_SHORT.items():
        top = rk.get(t, [])
        best = next((r["rank"] for r in top if r["label"]), None)
        top3 = ", ".join(re.sub(r"\s*\[.*\]", "", r["drug"]) + (r"$^{*}$" if r["label"] else "") for r in top[:3])
        lines.append(f"{short} & {int(main_pd.get(t, {}).get('n_pos', 0))} & {f3(knn_pd.get(t, {}).get('mrr'))} & "
                     f"{f3(paths_pd.get(t, {}).get('mrr'))} & {f3(main_pd.get(t, {}).get('mrr'))} & "
                     f"{best if best else '$>$100'} & {top3}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_transfer(S):
    rows = [(r"BioPathNet++, all diseases in training", "main", "targets"),
            (r"BioPathNet++, 123 autoimmune diseases removed", "noauto_main", "no_autoimmune"),
            (r"Paths only, 123 autoimmune diseases removed", "noauto_paths", "no_autoimmune")]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Training data & \multicolumn{2}{c}{6 targets} & \multicolumn{2}{c}{21 autoimmune-area diseases}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & MRR & H@10 & MRR & H@10\\", r"\midrule"]
    for lab, g, _ in rows:
        if g not in S["models"]:
            lines.append(f"{lab} & \\multicolumn{{4}}{{c}}{{pending}}\\\\"); continue
        m = S["models"][g]
        lines.append(f"{lab} ($n$={len(m['seeds'])}) & {cell_model(m, 'targets', 'mrr')} & {cell_model(m, 'targets', 'hits@10')} & "
                     f"{cell_model(m, 'area', 'mrr')} & {cell_model(m, 'area', 'hits@10')}\\\\")
    kn = S["baselines"]["no_autoimmune"]["disease_knn"]
    lines.append(f"Disease kNN (no-autoimmune pool) & {f3(kn['targets']['macro']['mrr']['mean'])} & "
                 f"{f3(kn['targets']['macro']['hits@10']['mean'])} & {f3(kn['area']['macro']['mrr']['mean'])} & "
                 f"{f3(kn['area']['macro']['hits@10']['mean'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_ablation(S):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Variant (seed 0 unless noted) & Valid MRR & Valid H@10 & Targets MRR & $p$ vs full (valid)\\", r"\midrule"]
    for g, lab in (("main", "Full model ($n$=3)"), ("paths", "$-$ transfer channel ($n$=3)"),
                   ("prioronly", "$-$ path reasoning (prior only)"), ("brg_main", "BRG relation subset"),
                   ("bce_main", "BCE loss instead of listwise")):
        if g not in S["models"]:
            lines.append(f"{lab} & \\multicolumn{{4}}{{c}}{{pending}}\\\\"); continue
        m = S["models"][g]
        key = {"paths": "main_vs_paths_valid", "prioronly": "main_vs_prioronly_valid",
               "brg_main": "main_vs_brg_valid", "bce_main": "main_vs_bce_valid"}.get(g)
        p = pfmt(S["tests"][key]["p_wilcoxon_two_sided"]) if key in S["tests"] else "--"
        lines.append(f"{lab} & {cell_model(m, 'valid', 'mrr')} & {cell_model(m, 'valid', 'hits@10')} & "
                     f"{cell_model(m, 'targets', 'mrr')} & {p}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_metrics(S):
    M = S.get("metrics_table", {})
    head = " & ".join(lab for _, lab in TABLE_METRICS)
    lines = [r"\begin{tabular}{ll" + "c" * len(TABLE_METRICS) + "}", r"\toprule", f"Set & Method & {head}\\\\", r"\midrule"]
    names = {"random": "Random", "popularity": "Popularity", "disease_knn": "Disease $k$NN", "model": r"\textbf{\sysname}"}
    titles = {"test": "Test (68, never selected)", "valid": "Validation (68)", "targets": "Rare targets (6)"}
    for s in ("test", "valid", "targets"):
        if s not in M:
            continue
        first = True
        for k in ("random", "popularity", "disease_knn", "model"):
            r = M[s]["rows"].get(k)
            if not r:
                continue
            cells = []
            for m, _ in TABLE_METRICS:
                v = r.get(m)
                cells.append("--" if v is None else (f"{100 * v:.0f}\\%" if m == "_any10" else
                                                     (f"{v:,.0f}" if m == "mean_rank" else f"{v:.3f}")))
            lines.append((titles[s] if first else "") + f" & {names[k]} & " + " & ".join(cells) + r"\\")
            first = False
        lines.append(r"\midrule" if s != "targets" else "")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(l for l in lines if l)


def table_validation(S):
    V = S.get("validation", {})
    if not V:
        return "% validation pending"
    def pct(d, g):
        x = d.get(g)
        x = x["share"] if isinstance(x, dict) else x
        return "--" if x is None else f"{100 * x:.0f}\\%"
    rows = [("Trial evidence, grade A--C (ClinicalTrials.gov)", V["v2_trial_grade_A_to_C_among_novel"]),
            ("Trial evidence, phase $\\geq$2 (grade A--B)", V["v2_trial_grade_A_or_B_among_novel"]),
            ("Any PubMed co-mention", V["v3_any_pubmed_hit_among_novel"]),
            ("PubMed title supports use", V["v3_supporting_among_novel"]),
            ("PubMed title reports harm / no benefit", V["v3_against_among_novel"]),
            ("$\\geq$1 drug target among disease genes", V["v5_share_any_disease_gene_target"])]
    n = {g: V["v2_trial_grade_A_to_C_among_novel"][g]["n"] for g in ("model", "random", "popularity")}
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             f"Evidence for unlabelled candidates & \\sysname{{}} top-20 ($n$={n['model']}) & Popularity top-20 ($n$={n['popularity']}) & Random ($n$={n['random']})\\\\",
             r"\midrule"]
    for lab, d in rows:
        lines.append(f"{lab} & {pct(d, 'model')} & {pct(d, 'popularity')} & {pct(d, 'random')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_temporal(S):
    rows = [r for r in S.get("validation", {}).get("temporal_rows", []) if r.get("in_primekg") and not r.get("already_labelled")]
    short = {"Primary Biliary Cholangitis": "PBC", "Myasthenia Gravis": "MG", "Crohn Disease": "Crohn", "Systemic Sclerosis": "SSc"}
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Drug & Target & FDA approval & \sysname{} rank & Top \% & $k$NN rank & Popularity rank\\", r"\midrule"]
    for post in (True, False):
        for r in sorted([r for r in rows if r.get("post_snapshot") == post], key=lambda r: r["mean_rank"]):
            lines.append(f"{r['primekg_name']} & {short.get(r['target'], r['target'])} & {str(r.get('fda_approval'))[:7]} & "
                         f"{r['mean_rank']:,.0f} & {r['percentile_top']:.1f} & {int(r['knn_rank']):,} & {int(r['popularity_rank']):,}\\\\")
        if post:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_paths(S, picks=(("Myasthenia Gravis", "Pyridostigmine"), ("Primary Biliary Cholangitis", "Ursodeoxycholic acid"),
                          ("Myasthenia Gravis", "Amifampridine"), ("Crohn Disease", "Ciprofloxacin"))):
    ex = {(e["target"], e["drug"]): e for e in S.get("validation", {}).get("explanations", [])}
    lines = [r"\begin{tabular}{p{0.97\linewidth}}", r"\toprule"]
    for t, d in picks:
        e = ex.get((t, d))
        if not e or not e["paths"]:
            continue
        p = e["paths"][0]
        steps = [p["nodes"][0]] + [f"$\\xrightarrow{{\\text{{{rel.replace('_', ' ').replace('^-1', '$^{-1}$').replace('->', r'$\to$')}}}}}$ {n}"
                                   for rel, n in zip(p["relations"], p["nodes"][1:])]
        lines.append(r"\textbf{" + f"{d} for {t}" + r"}: " + " ".join(steps).replace("&", r"\&") + r"\\[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


PENDING_MACROS = [
    *(f"{c}{f}Mrr" for c in ("TrA", "TrActl", "TrD", "TrCE") for f in ("E", "EKnn", "Targets", "TargetsKnn")),
    *(f"{p}{t}" for t in ("TrAActlE", "TrAActlEKnn", "TrAActlTargets", "TrCEDE", "TrCEAE", "TrDAE", "TestKnn")
      for p in ("Diff", "DiffLo", "DiffHi", "Pwil", "Wins", "Ties", "Losses")),
    "TrAPathsEMrr", "TrActlPathsEMrr",
    *(f"{p}TrAActlPathsE" for p in ("Diff", "DiffLo", "DiffHi", "Pwil", "Wins", "Ties", "Losses")),
    "TestMrr", "TestHitsTen", "TestAnyTen", "TestModelValidMrr", "KnnTestMrr", "KnnTestHitsTen", "KnnTestAnyTen",
    "TxhN", "TxhPairs", "TxhTxOfficialRecall", "TxhCand", "TxhTrainDis", "TxhValidDis", "TxhPretrainH", "TxhFinetuneH",
    "TxhOursIndShare", "TxhTxIndShare", "TxhNonTestDis",
    *(f"Txh{w}{k}" for w in ("Ours", "Tx") for k in ("Recall", "RecallTen", "Auroc", "Auprc", "Mrr", "HitsTen",
                                                       "HitsHund", "FullAuroc")),
    *(f"Txh{p}{k}" for k in ("Recall", "Auroc", "Auprc", "FullMrr", "FullHitsHund", "FullAuroc")
      for p in ("Diff", "DiffLo", "DiffHi", "Pwil", "Wins", "Losses")),
]
TXH_KEYS = (("Recall", "txgnn_recall@100"), ("RecallTen", "txgnn_recall@10"), ("Auroc", "txgnn_auroc"),
            ("Auprc", "txgnn_auprc"), ("Mrr", "full_mrr"), ("HitsTen", "full_hits@10"),
            ("HitsHund", "full_hits@100"), ("FullAuroc", "full_auroc"))


def _ci(r):
    return "pending" if not r else f"{r['mean']:.3f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"


def table_transfer_curve(S):
    T = S.get("transfer", {})
    lines = [r"\begin{tabular}{lrcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{12 held-out autoimmune-area diseases (E)} & \multicolumn{2}{c}{6 rare targets}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Training data & \#dis. & \sysname & disease $k$NN & \sysname & disease $k$NN\\", r"\midrule"]
    for key, *_ in TRANSFER_CONDITIONS:
        c = T.get("conditions", {}).get(key)
        if not c:
            continue
        lines.append(f"{c['label']} & {c['n_train']:,} & {_ci(c.get('model_E'))} & {_ci(c.get('knn_E'))} & "
                     f"{_ci(c.get('model_targets'))} & {_ci(c.get('knn_targets'))}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_test(S):
    T = S.get("test_split", {})
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Method (68 test diseases, never used for training or stopping) & MRR & H@10 & AUROC$_\text{hard}$\\",
             r"\midrule"]
    for k, lab in (("random", "Random"), ("popularity", "Popularity"), ("disease_knn", "Disease $k$NN"),
                   ("model", r"\textbf{\sysname}")):
        r = T.get(k)
        lines.append(f"{lab} & " + (" & ".join(f3((r.get(m) or {}).get("mean")) for m in ("mrr", "hits@10", "auroc_hard"))
                                    if r else r"\multicolumn{3}{c}{pending}") + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_txgnn(S):
    H = S.get("txgnn_h2h", {})
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{TxGNN protocol} & \multicolumn{3}{c}{full ranking}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & R@100 & AUROC & AUPRC & MRR & H@100 & AUROC\\", r"\midrule",
             f"TxGNN, published (5 seeds) & {TXGNN_PUBLISHED_RECALL100:.3f} & -- & -- & -- & -- & --" + r"\\"]
    for who, lab in (("txgnn", "TxGNN, re-run (this split)"), ("ours", r"\sysname (this split)")):
        r = H.get(who)
        lines.append(f"{lab} & " + (" & ".join(f3((r.get(k) or {}).get("mean")) for k in
                                               ("txgnn_recall@100", "txgnn_auroc", "txgnn_auprc", "full_mrr",
                                                "full_hits@100", "full_auroc"))
                                    if r else r"\multicolumn{6}{c}{pending}") + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def macros(S) -> str:
    out = []

    def nc(name, val):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    def g(group, s, k):
        return S["models"].get(group, {}).get("sets", {}).get(s, {}).get("seed_mean_sd", {}).get(k)

    for grp, tag in (("main", "Main"), ("paths", "Paths"), ("prioronly", "Prior"), ("noauto_main", "Noauto"),
                     ("noauto_paths", "NoautoPaths"), ("brg_main", "Brg"), ("bce_main", "Bce")):
        for s, st in (("valid", "Valid"), ("targets", "Targets"), ("area", "Area")):
            for k, kt in (("mrr", "Mrr"), ("hits@1", "HitsOne"), ("hits@10", "HitsTen"), ("hits@100", "HitsHundred"),
                          ("auroc", "Auroc"), ("auprc", "Auprc"), ("auroc_hard", "AurocHard"),
                          ("auroc_degmatch", "AurocDeg")):
                v = g(grp, s, k)
                if v:
                    nc(f"{tag}{st}{kt}", f"{v['mean']:.3f}")
                    nc(f"{tag}{st}{kt}Sd", f"{v['sd']:.3f}" if v["sd"] is not None else "--")
            if grp in S["models"] and s in S["models"][grp]["sets"]:
                a = S["models"][grp]["sets"][s]["macro_seedavg"]["_any10"]
                nc(f"{tag}{st}AnyTen", f"{100 * a['mean']:.0f}")
        if grp in S["models"]:
            nc(f"{tag}Seeds", str(len(S["models"][grp]["seeds"])))
    for proto, ptag in (("targets", ""), ("no_autoimmune", "Noauto")):
        for b, bt in (("disease_knn", "Knn"), ("popularity", "Pop"), ("random", "Rand"), ("drug_degree", "Deg")):
            for s, st in (("valid", "Valid"), ("targets", "Targets"), ("area", "Area")):
                r = S["baselines"][proto][b].get(s)
                if not r:
                    continue
                for k, kt in (("mrr", "Mrr"), ("hits@10", "HitsTen"), ("auroc", "Auroc"), ("auroc_hard", "AurocHard")):
                    if r["macro"].get(k):
                        nc(f"{ptag}{bt}{st}{kt}", f"{r['macro'][k]['mean']:.3f}")
                nc(f"{ptag}{bt}{st}AnyTen", f"{100 * r['macro']['_any10']['mean']:.0f}")
    names = {"main_vs_knn_valid": "MainKnnValid", "main_vs_knn_targets": "MainKnnTargets",
             "main_vs_paths_valid": "MainPathsValid", "main_vs_paths_targets": "MainPathsTargets",
             "main_vs_noauto_targets": "MainNoautoTargets", "noauto_vs_knn_area": "NoautoKnnArea",
             "paths_vs_knn_valid": "PathsKnnValid", "main_vs_prioronly_valid": "MainPriorValid",
             "main_vs_knn_valid_hits10": "MainKnnValidHitsTen", "main_vs_bce_valid": "MainBceValid",
             "main_vs_brg_valid": "MainBrgValid", "main_vs_brg_targets": "MainBrgTargets",
             "tr_A_vs_Actl_E_model": "TrAActlE", "tr_A_vs_Actl_E_knn": "TrAActlEKnn",
             "tr_A_vs_Actl_targets_model": "TrAActlTargets", "tr_CE_vs_D_E_model": "TrCEDE",
             "tr_CE_vs_A_E_model": "TrCEAE", "tr_D_vs_A_E_model": "TrDAE", "test_main_vs_knn_test": "TestKnn",
             "tr_A_vs_Actl_E_last": "TrAActlELast", "tr_A_vs_Actl_paths_E": "TrAActlPathsE", "tr_A_vs_Actl_targets_last": "TrAActlTargetsLast",
             "noauto_main_vs_paths_area": "NoautoPathsArea", "s0_main_vs_bce_valid": "SzeroMainBceValid",
             "s0_main_vs_brg_valid": "SzeroMainBrgValid", "s0_noauto_main_vs_paths_area": "SzeroNoautoPathsArea"}
    for k, t in names.items():
        if k in S["tests"]:
            r = S["tests"][k]
            nc(f"Diff{t}", f"{r['mean_diff']:+.3f}")
            nc(f"Pboot{t}", ptext(r["p_boot_one_sided"]))
            nc(f"Pwil{t}", ptext(r["p_wilcoxon_two_sided"]))
            nc(f"DiffLo{t}", f"{r['diff_ci95'][0]:+.3f}"); nc(f"DiffHi{t}", f"{r['diff_ci95'][1]:+.3f}")
            nc(f"Wins{t}", str(r["wins"])); nc(f"Ties{t}", str(r["ties"])); nc(f"Losses{t}", str(r["losses"]))
    gaps = defaultdict(list)
    for run, sg in S["selection_gap"].items():
        m = re.match(r"(.+)_s\d+$", run)
        if m:
            gaps[m.group(1)].append(sg["gap_best_minus_last"])
    for grp, tag in (("main", "Main"), ("paths", "Paths")):
        if gaps.get(grp):
            nc(f"SelGap{tag}Min", f"{min(gaps[grp]):.3f}"); nc(f"SelGap{tag}Max", f"{max(gaps[grp]):.3f}")
    tags = {"Actl": "TrActl", "A": "TrA", "D": "TrD", "CE": "TrCE"}
    for key, c in S.get("transfer", {}).get("conditions", {}).items():
        nc(f"{tags[key]}NTrain", f"{c['n_train']:,}".replace(",", "{,}"))
        for f, ft in (("model_E", "E"), ("knn_E", "EKnn"), ("model_targets", "Targets"), ("knn_targets", "TargetsKnn")):
            if c.get(f):
                nc(f"{tags[key]}{ft}Mrr", f"{c[f]['mean']:.3f}")
    for sname, st in (("E", "E"), ("targets", "Targets")):
        lc = S.get("transfer", {}).get("last_checkpoint", {}).get(sname)
        if lc:
            nc(f"TrA{st}LastMrr", f"{lc['A']['mean']:.3f}"); nc(f"TrActl{st}LastMrr", f"{lc['Actl']['mean']:.3f}")
    po = S.get("transfer", {}).get("paths_only")
    if po:
        nc("TrAPathsEMrr", f"{po['A']['mean']:.3f}"); nc("TrActlPathsEMrr", f"{po['Actl']['mean']:.3f}")
    T = S.get("test_split", {})
    nc("TestSeeds", str(len(S["models"]["test_main"]["seeds"])) if "test_main" in S["models"] else "0")
    for k, kt in (("model", "Test"), ("disease_knn", "KnnTest"), ("popularity", "PopTest")):
        if T.get(k):
            nc(f"{kt}Mrr", f"{T[k]['mrr']['mean']:.3f}")
            nc(f"{kt}HitsTen", f"{T[k]['hits@10']['mean']:.3f}")
            nc(f"{kt}AnyTen", f"{100 * T[k]['_any10']['mean']:.0f}")
    if T.get("model_valid"):
        nc("TestModelValidMrr", f"{T['model_valid']['mean']:.3f}")
    M = S.get("metrics_table", {})
    for s, st in (("test", "Test"), ("valid", "Valid"), ("targets", "Targets")):
        if s in M:
            nc(f"RandExpAnyTen{st}", f"{100 * M[s]['random_expected_any10']:.1f}")
            nc(f"Prevalence{st}", f"{M[s]['prevalence']:.4f}")
            nc(f"MeanPos{st}", f"{M[s]['mean_positives']:.1f}")
            if "median_rank_known_drug" in M[s]:
                nc(f"MedRank{st}", f"{M[s]['median_rank_known_drug']:.0f}")
                nc(f"RankOneShare{st}", f"{100 * M[s]['share_known_drugs_rank1']:.0f}")
            for k, kt in (("model", "Model"), ("disease_knn", "Knn"), ("popularity", "Pop"), ("random", "Rand")):
                r = M[s]["rows"].get(k)
                if r:
                    nc(f"Tab{kt}{st}AnyTen", f"{100 * r['_any10']:.0f}")
                    if r.get("auprc") is not None:
                        nc(f"Tab{kt}{st}Auprc", f"{r['auprc']:.3f}")
                    if r.get("auroc") is not None:
                        nc(f"Tab{kt}{st}Auroc", f"{r['auroc']:.3f}")
    V = S.get("validation", {})
    if V:
        def sh(d, g):
            x = d[g]
            return f"{100 * (x['share'] if isinstance(x, dict) else x):.0f}"
        for key, tag in (("v2_trial_grade_A_to_C_among_novel", "ValTrial"), ("v2_trial_grade_A_or_B_among_novel", "ValTrialAB"),
                         ("v3_any_pubmed_hit_among_novel", "ValLitHit"), ("v3_supporting_among_novel", "ValLitSup"),
                         ("v3_against_among_novel", "ValLitAgainst"), ("v5_share_any_disease_gene_target", "ValDisGene")):
            for g, gt in (("model", "Model"), ("random", "Rand"), ("popularity", "Pop")):
                nc(f"{tag}{gt}", sh(V[key], g))
        nc("ValNovelN", str(V["v2_trial_grade_A_to_C_among_novel"]["model"]["n"]))
        e = V.get("e_check")
        if e:
            nc("EcheckLabelled", str(e["labelled_in_top10"])); nc("EcheckDiseases", str(e["diseases_with_labelled_in_top10"]))
            nc("EcheckNovel", str(e["unlabelled"])); nc("EcheckNovelTrial", str(e["unlabelled_with_trial_A_to_C"]))
        nc("ValGradeA", str(V["v2_grade_counts_model_novel"].get("A", 0)))
        nc("ValSteroidShare", f"{100 * V['v7']['share_corticosteroid_by_name_all_candidates']:.0f}")
        nc("ValAntibioticShare", f"{100 * V['v7']['share_antibacterial_by_name_all_candidates']:.0f}")
        nc("ValAtcCoverage", f"{100 * V['v7']['atc_coverage']:.0f}")
        for tag, key in (("TempPost", "v4_post_snapshot"), ("TempGap", "v4_label_gap")):
            t = V.get(key)
            if t:
                nc(f"{tag}N", str(t["n"])); nc(f"{tag}MedRank", f"{t['median_model_rank']:,.0f}".replace(",", "{,}"))
                nc(f"{tag}MedPct", f"{t['median_percentile']:.0f}"); nc(f"{tag}Better", str(t["n_better_than_median_rank"]))
                nc(f"{tag}P", ptext(t["sign_test_p_one_sided"])); nc(f"{tag}KnnMedRank", f"{t['median_knn_rank']:,.0f}".replace(",", "{,}"))
                nc(f"{tag}BetterKnn", str(t["n_model_better_than_knn"]))
        for r in V.get("temporal_rows", []):
            if r.get("in_primekg"):
                nc("TempRank" + re.sub(r"[^A-Za-z]", "", r["drug"].title()), f"{r['mean_rank']:,.0f}".replace(",", "{,}"))
    H = S.get("txgnn_h2h", {})
    nc("TxhTxPublishedRecall", f"{TXGNN_PUBLISHED_RECALL100:.3f}")
    n = (H.get("ours") or H.get("txgnn") or {}).get("n_diseases")
    if n:
        nc("TxhN", str(n))
    for who, wt in (("ours", "Ours"), ("txgnn", "Tx")):
        if H.get(who):
            for k, key in TXH_KEYS:
                if H[who].get(key):
                    nc(f"Txh{wt}{k}", f"{H[who][key]['mean']:.3f}")
    if H.get("n_test_pairs"):
        nc("TxhPairs", str(H["n_test_pairs"])); nc("TxhCand", f"{H['n_candidates']:,}".replace(",", "{,}"))
    if H.get("n_train_diseases"):
        nc("TxhTrainDis", str(H["n_train_diseases"])); nc("TxhValidDis", str(H["n_valid_diseases"]))
    if H.get("coverage"):
        nc("TxhNonTestDis", f"{H['coverage']['non_test_diseases_with_indications']:,}".replace(",", "{,}"))
    for stage, tag in (("pretrain", "TxhPretrainH"), ("finetune", "TxhFinetuneH")):
        if H.get(f"txgnn_{stage}_h") is not None:
            nc(tag, f"{H[f'txgnn_{stage}_h']:.2f}")
    for who, tag in (("ours", "TxhOursIndShare"), ("txgnn", "TxhTxIndShare")):
        if H.get(f"{who}_top100_indicated_share") is not None:
            nc(tag, f"{100 * H[f'{who}_top100_indicated_share']:.0f}")
    if H.get("txgnn_official", {}).get("Recall@100"):
        nc("TxhTxOfficialRecall", f"{H['txgnn_official']['Recall@100']['mean']:.3f}")
    for k, key, src in (("Recall", "recall@100", "paired"), ("Auroc", "auroc", "paired"), ("Auprc", "auprc", "paired"),
                        ("FullMrr", "mrr", "paired_full"), ("FullHitsHund", "hits@100", "paired_full"),
                        ("FullAuroc", "auroc", "paired_full")):
        c = H.get(src, {}).get(key)
        if c:
            nc(f"TxhDiff{k}", f"{c['mean_diff']:+.3f}"); nc(f"TxhDiffLo{k}", f"{c['diff_ci95'][0]:+.3f}")
            nc(f"TxhDiffHi{k}", f"{c['diff_ci95'][1]:+.3f}"); nc(f"TxhPwil{k}", ptext(c["p_wilcoxon_two_sided"]))
            nc(f"TxhWins{k}", str(c["wins"])); nc(f"TxhLosses{k}", str(c["losses"]))
    # numbers that only exist after later Kaggle sessions: print "[pending]" so the papers compile meanwhile
    defined = {re.match(r"\\newcommand\{\\(\w+)\}", l).group(1) for l in out}
    for name in PENDING_MACROS:
        if name not in defined:
            nc(name, r"\textbf{[pending]}")
    info = S.get("run_info", {})
    for run, tag in (("main_s0", "Main"), ("brg_main_s0", "Brg")):
        if run in info and info[run].get("edges"):
            nc(f"Edges{tag}", f"{info[run]['edges']:,}".replace(",", "{,}"))
            nc(f"StepSec{tag}", f"{info[run]['s_per_step']:.2f}")
            nc(f"PeakMem{tag}", f"{info[run]['peak_gb']:.2f}")
    if "main_s0" in info and "brg_main_s0" in info:
        nc("BrgEdgeCutPct", f"{100 * (1 - info['brg_main_s0']['edges'] / info['main_s0']['edges']):.0f}")
        nc("BrgSpeedupPct", f"{100 * (1 - info['brg_main_s0']['s_per_step'] / info['main_s0']['s_per_step']):.0f}")
    sp = S["sparsity"]
    for lab, tag in (("BioPathNet++", "Main"), ("Disease kNN", "Knn")):
        if lab in sp:
            nc(f"Rho{tag}", f"{sp[lab]['spearman_rho']:.2f}"); nc(f"RhoP{tag}", ptext(sp[lab]["spearman_p"]))
            for b, bt in (("low", "Low"), ("mid", "Mid"), ("high", "High")):
                nc(f"Tert{tag}{bt}", f"{sp[lab]['tertile_mrr'][b]:.3f}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/bpp")
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--paper", default="outputs/paper/v3/generated")
    a = ap.parse_args()
    root, paper = Path(a.root), Path(a.paper)
    S = analyse(root, Path(a.data))
    S["_knn_targets_pd"] = {d: {"mrr": m["mrr"]} for d, m in
                            json.loads((root / "baselines_targets.json").read_text())["disease_knn"]["targets"]["per_disease"].items()}
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "summary.json").write_text(json.dumps(S, indent=1))
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "numbers.tex").write_text(macros(S))
    (paper / "tab_main.tex").write_text(table_main(S))
    (paper / "tab_targets.tex").write_text(table_targets(S))
    (paper / "tab_transfer.tex").write_text(table_transfer(S))
    (paper / "tab_ablation.tex").write_text(table_ablation(S))
    (paper / "tab_transfer_curve.tex").write_text(table_transfer_curve(S))
    (paper / "tab_test.tex").write_text(table_test(S))
    (paper / "tab_metrics.tex").write_text(table_metrics(S))
    (paper / "tab_validation.tex").write_text(table_validation(S))
    (paper / "tab_temporal.tex").write_text(table_temporal(S))
    (paper / "tab_paths.tex").write_text(table_paths(S))
    (paper / "tab_txgnn.tex").write_text(table_txgnn(S))
    print("runs:", ", ".join(S["runs_found"]))
    for k, v in S["tests"].items():
        print(f"  {k:28s} diff {v['mean_diff']:+.3f} [{v['diff_ci95'][0]:+.3f},{v['diff_ci95'][1]:+.3f}] "
              f"W/T/L {v['wins']}/{v['ties']}/{v['losses']}  p_boot {v['p_boot_one_sided']:.3f}  "
              f"p_wilcoxon {v['p_wilcoxon_two_sided']:.3f}  n={v['n']}")
    print("sparsity:", json.dumps(S["sparsity"], indent=1))
    print(f"wrote {root / 'analysis' / 'summary.json'} and {paper}/")


if __name__ == "__main__":
    main()

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
}
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
        if m and m.group(1) in GROUPS:
            g[m.group(1)][int(m.group(2))] = json.loads((path / "results.json").read_text())
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
        for s in ("valid", "targets", "area"):
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

    out["sparsity"] = sparsity(per, data)
    out["case_studies"] = case_studies(runs)
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
    """p-value for running text: '$<$0.001' or '$=$0.012' (always written as  $p$\,\Pwil...)."""
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


def macros(S) -> str:
    out = []

    def nc(name, val):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    def g(group, s, k):
        return S["models"].get(group, {}).get("sets", {}).get(s, {}).get("seed_mean_sd", {}).get(k)

    for grp, tag in (("main", "Main"), ("paths", "Paths"), ("prioronly", "Prior"), ("noauto_main", "Noauto")):
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
             "main_vs_knn_valid_hits10": "MainKnnValidHitsTen"}
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
    print("runs:", ", ".join(S["runs_found"]))
    for k, v in S["tests"].items():
        print(f"  {k:28s} diff {v['mean_diff']:+.3f} [{v['diff_ci95'][0]:+.3f},{v['diff_ci95'][1]:+.3f}] "
              f"W/T/L {v['wins']}/{v['ties']}/{v['losses']}  p_boot {v['p_boot_one_sided']:.3f}  "
              f"p_wilcoxon {v['p_wilcoxon_two_sided']:.3f}  n={v['n']}")
    print("sparsity:", json.dumps(S["sparsity"], indent=1))
    print(f"wrote {root / 'analysis' / 'summary.json'} and {paper}/")


if __name__ == "__main__":
    main()

"""
Paper figures, drawn from outputs/bpp/analysis/summary.json (run `python -m bpp.analyze` first).

  python -m bpp.figures            -> outputs/paper/v3/figures/*.pdf (+ .png previews)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ACCENT, GREY, INK, INK2, MUTED, GRID = "#2a78d6", "#9a9893", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
ORANGE = "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL1 = 3.5   # IEEE single column width (in)
COL2 = 7.16  # IEEE double column width (in)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ 1. pipeline diagram
def pipeline(out: Path):
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, title, body, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc="#eef4fc" if accent else "#f6f6f4", ec=ACCENT if accent else MUTED, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK2))

    box(0.5, 4, 16, 27, "PrimeKG", "129,375 nodes\n4.0M context edges\n27 relation types\n+ inverse edges\n\n42,631 drug-disease\ntherapeutic pairs")
    box(20, 4, 18, 27, "Zero-shot split", "held-out disease +\nall its disease-disease\nneighbours (strict family)\n\ntheir therapeutic edges\nnever enter the graph;\nper batch: query diseases'\nedges removed too")
    box(41.5, 18.5, 22, 12.5, "Path channel (NBFNet)", "indicator on query disease,\n6 layers, DistMult messages,\nPNA degree scaling, dim 32", accent=True)
    box(41.5, 4, 22, 12.5, "Transfer channel (prior)", "popularity, disease-kNN,\nshared targets, autoimmune\nfrequency (family-excluded)", accent=True)
    box(67, 4, 14, 27, "Score", "path logit\n+ MLP(prior)\n\nlistwise\nsoftmax loss\nover all drugs\n\n~68k params")
    box(84.5, 4, 15, 27, "Evaluation", "rank all 7,957\ndrugs, filtered\n\nMRR, Hits@k,\nAUROC (all /\nhard / degree-\nmatched), AUPRC")
    arrow(17, 17.5, 19.6, 17.5)
    arrow(38.5, 24, 41.1, 24); arrow(38.5, 10, 41.1, 10)
    arrow(64, 24, 66.6, 20); arrow(64, 10, 66.6, 14)
    arrow(81.5, 17.5, 84.1, 17.5)
    ax.text(52.5, 0.6, "both channels see only training diseases' drug labels", ha="center", fontsize=6.4,
            color=MUTED, style="italic")
    save(fig, out, "fig_pipeline")


# ------------------------------------------------------------------ 2. main results (dot + CI)
METHODS = [("base", "random", "Random"), ("base", "popularity", "Popularity"),
           ("base", "shared_targets", "Shared targets"), ("base", "disease_knn", "Disease kNN (best baseline)"),
           ("model", "prioronly", "Prior only"), ("model", "paths", "Paths only"),
           ("model", "main", "BioPathNet++")]


def point(S, kind, key, s, metric="mrr"):
    if kind == "base":
        r = S["baselines"]["targets"][key].get(s, {}).get("macro", {}).get(metric)
    else:
        r = S["models"].get(key, {}).get("sets", {}).get(s, {}).get("macro_seedavg", {}).get(metric)
    return r


def main_results(S, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.0), sharey=True)
    for ax, (s, title) in zip(axes, (("valid", "68 unseen validation diseases"), ("targets", "6 rare autoimmune targets"))):
        for i, (kind, key, lab) in enumerate(METHODS):
            r = point(S, kind, key, s)
            if not r:
                continue
            ours = key == "main"
            c = ACCENT if ours else (INK2 if kind == "model" else GREY)
            ax.plot(r["ci95"], [i, i], color=c, lw=1.2, solid_capstyle="round")
            ax.plot(r["mean"], i, "o", ms=5.5 if ours else 4.5, color=c, mec="white", mew=0.8, zorder=3)
            ax.text(r["ci95"][1] + 0.02, i, f"{r['mean']:.3f}", va="center", fontsize=6.8,
                    color=INK if ours else INK2, weight="bold" if ours else "normal")
        knn = point(S, "base", "disease_knn", s)
        ax.axvline(knn["mean"], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.set_xlim(0, 1.0); ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("MRR (macro over diseases, 95% bootstrap CI)")
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(METHODS))); axes[0].set_yticklabels([m[2] for m in METHODS])
    for t in axes[0].get_yticklabels():
        if t.get_text() == "BioPathNet++":
            t.set_color(ACCENT); t.set_weight("bold")
    save(fig, out, "fig_main_results")


# ------------------------------------------------------------------ 3. AUROC vs MRR
def auroc_vs_mrr(S, out: Path):
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    pts = []
    for kind, key, lab in METHODS + [("base", "drug_degree", "Drug degree")]:
        a, m = point(S, kind, key, "valid", "auroc"), point(S, kind, key, "valid", "mrr")
        if a and m:
            pts.append((lab, a["mean"], m["mean"], key == "main", kind))
    for lab, a, m, ours, kind in pts:
        ax.plot(a, m, "o", ms=6 if ours else 4.5, color=ACCENT if ours else (INK2 if kind == "model" else GREY),
                mec="white", mew=0.8, zorder=3)
    offs = {"Random": (0.012, 0.012), "Drug degree": (-0.095, 0.004), "Popularity": (-0.083, 0.008),
            "Shared targets": (0.012, 0.004), "Disease kNN (best baseline)": (-0.105, -0.006),
            "Prior only": (0.012, -0.022), "Paths only": (-0.072, 0.02), "BioPathNet++": (-0.118, -0.028)}
    for lab, a, m, ours, _ in pts:
        dx, dy = offs.get(lab, (0.01, 0.01))
        ax.text(a + dx, m + dy, lab.replace(" (best baseline)", ""), fontsize=6.6,
                color=ACCENT if ours else INK2, weight="bold" if ours else "normal")
    ax.axvspan(0.78, 0.86, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0.455, 0.49, "grey band: no-learning baselines\nalready reach AUROC 0.78-0.86", fontsize=6.2,
            color=MUTED, va="top")
    ax.set_xlim(0.44, 1.0); ax.set_ylim(-0.02, 0.5)
    ax.set_xlabel("AUROC (68 validation diseases)"); ax.set_ylabel("MRR")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, out, "fig_auroc_vs_mrr")


# ------------------------------------------------------------------ 4. sparsity tertiles
def sparsity(S, out: Path):
    sp = S["sparsity"]
    if "BioPathNet++" not in sp:
        return
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    bins = ["low", "mid", "high"]
    e = sp["BioPathNet++"]["tertile_edges"]; n = sp["BioPathNet++"]["tertile_n"]
    labels = [f"sparse\n(<={e[0]:.0f} edges, n={n['low']})", f"middle\n(n={n['mid']})",
              f"dense\n(>{e[1]:.0f} edges, n={n['high']})"]
    w = 0.36
    for j, (lab, c) in enumerate((("Disease kNN", GREY), ("BioPathNet++", ACCENT))):
        vals = [sp[lab]["tertile_mrr"][b] for b in bins]
        xs = [i + (j - 0.5) * (w + 0.03) for i in range(3)]
        ax.bar(xs, vals, width=w, color=c, label=lab, edgecolor="white", linewidth=0.8)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.6, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("MRR"); ax.set_ylim(0, 0.68)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    save(fig, out, "fig_sparsity")


# ------------------------------------------------------------------ 5. per-target per-seed
def per_target(S, root: Path, out: Path):
    short = {"Primary Biliary Cholangitis": "PBC", "Systemic Sclerosis": "SSc", "Sjogren Syndrome": "Sjögren",
             "Autoimmune Hepatitis": "AIH", "Myasthenia Gravis": "MG", "Crohn Disease": "Crohn"}
    from bpp.analyze import collect_runs
    runs = collect_runs(root)
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    for j, (grp, lab, c) in enumerate((("paths", "Paths only", INK2), ("main", "BioPathNet++", ACCENT))):
        for sd in range(3):
            p = runs.get(f"{grp}_s{sd}")
            if not p:
                continue
            per = json.loads((p / "results.json").read_text())["targets"]["per_disease"]
            for i, t in enumerate(short):
                ax.plot(i + (j - 0.5) * 0.28, per[t]["mrr"], "o", ms=4, color=c, alpha=0.85, mec="white", mew=0.5,
                        label=lab if (sd == 0 and i == 0) else None)
    knn = S["_knn_targets_pd"]
    for i, t in enumerate(short):
        ax.plot([i - 0.32, i + 0.32], [knn[t]["mrr"]] * 2, color=GREY, lw=1.2, label="Disease kNN" if i == 0 else None)
    ax.set_xticks(range(6)); ax.set_xticklabels(list(short.values()))
    ax.set_ylabel("MRR (one dot per seed)"); ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=6.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    save(fig, out, "fig_per_target")


# ------------------------------------------------------------------ 6. transfer curve (Aim 1)
def transfer_curve(S, out: Path):
    T = S.get("transfer", {}).get("conditions", {})
    order = [k for k in ("Actl", "A", "D", "CE") if k in T]
    if not order:
        return
    short = {"Actl": "A$'$ matched\nnon-autoimmune", "A": "A  RA/SLE/MS\n(original design)",
             "D": "D  all non-\nautoimmune", "CE": "C$_E$  all\ndiseases"}
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True)
    for ax, (sname, title) in zip(axes, (("E", "12 held-out autoimmune-area diseases"), ("targets", "6 rare targets"))):
        labelled = set()
        for i, k in enumerate(order):
            c = T[k]
            for dx, f, col, lab in ((-0.12, "knn", GREY, "Disease kNN"), (0.12, "model", ACCENT, "BioPathNet++")):
                r = c.get(f"{f}_{sname}")
                if not r:
                    if f == "model":
                        ax.text(i + dx, 0.03, "pending", rotation=90, fontsize=6, color=MUTED, ha="center")
                    continue
                ax.plot([i + dx] * 2, r["ci95"], color=col, lw=1.2)
                ax.plot(i + dx, r["mean"], "o", ms=5, color=col, mec="white", mew=0.8,
                        label=None if lab in labelled else lab)
                labelled.add(lab)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{short[k]}\n({T[k]['n_train']:,} dis.)" for k in order], fontsize=6.4)
        ax.set_title(title, fontsize=8, color=INK, loc="left"); ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("MRR (95% CI over diseases)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, fontsize=6.6, loc="upper left")
    save(fig, out, "fig_transfer_curve")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/bpp")
    ap.add_argument("--out", default="outputs/paper/v3/figures")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    S = json.loads((root / "analysis" / "summary.json").read_text())
    pipeline(out); main_results(S, out); auroc_vs_mrr(S, out); sparsity(S, out); per_target(S, root, out)
    transfer_curve(S, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()

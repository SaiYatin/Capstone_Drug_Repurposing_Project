"""Design diagrams for the capstone report and slides (matplotlib, 300 dpi). Usage: make_diagrams.py <out dir>"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

INK, MUTED, LINE = "#222222", "#555555", "#8A8A8A"
GREY, BLUE, ORANGE, RED, GREEN = "#F3F3F3", "#E8F0FA", "#FCEBDD", "#FBE3E3", "#E6F3E6"
BLUE_E, ORANGE_E, RED_E, GREEN_E = "#2E75B6", "#C55A11", "#B03A2E", "#3D8B3D"


def box(ax, x, y, w, h, title, body="", fc=GREY, ec=LINE, tsize=10.5, bsize=8.6, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.035 - 0.055 * (tsize / 10.5), body, ha="center", va="top", fontsize=bsize,
                color=MUTED, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)


def arrow(ax, a, b, color=LINE, style="-|>", lw=1.4, ls="-", rad=0.0, text=None, tpos=0.5, tcolor=MUTED, toff=(0, 0.018)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        x = a[0] + (b[0] - a[0]) * tpos + toff[0]
        y = a[1] + (b[1] - a[1]) * tpos + toff[1]
        ax.text(x, y, text, ha="center", va="bottom", fontsize=7.8, color=tcolor, style="italic")


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


# 1. High-level architecture --------------------------------------------------------------------------
fig, ax = canvas(10, 6.2)
lanes = [(0.80, "Data layer"), (0.555, "Protocol layer"), (0.31, "Learning layer"), (0.065, "Evaluation & reporting layer")]
for y, name in lanes:
    ax.add_patch(FancyBboxPatch((0.01, y - 0.02), 0.98, 0.215, boxstyle="round,pad=0,rounding_size=0.01",
                                fc="#FAFAFA", ec="#DDDDDD", lw=1))
    ax.text(0.02, y + 0.183, name, fontsize=8.5, color=MUTED, fontweight="bold", va="top")
box(ax, 0.05, 0.80, 0.20, 0.15, "PrimeKG (raw)", "kg.csv: 8.1M rows\n129,375 nodes", GREY)
box(ax, 0.30, 0.80, 0.22, 0.15, "KG builder", "build_primekg.py\ndedupe, inverses, CSR", ORANGE, ORANGE_E)
box(ax, 0.57, 0.80, 0.18, 0.15, "CompactKG store", "nodes / edges /\ntherapeutic .npz", GREY)
box(ax, 0.79, 0.80, 0.18, 0.15, "TxGNN split", "build_txgnn.py\n(head-to-head)", GREY)
arrow(ax, (0.25, 0.875), (0.30, 0.875))
arrow(ax, (0.52, 0.875), (0.57, 0.875))
arrow(ax, (0.79, 0.875), (0.75, 0.875))
box(ax, 0.10, 0.555, 0.26, 0.15, "Split generator", "splits.py, extra_splits.py\nstrict-family hold-out", ORANGE, ORANGE_E)
box(ax, 0.42, 0.555, 0.26, 0.15, "Task & priors", "task.py: labels, exclusions,\n4 leakage-free prior features", ORANGE, ORANGE_E)
box(ax, 0.74, 0.555, 0.23, 0.15, "Baselines", "run_baselines.py: random ...\nkNN, same protocol", GREY)
arrow(ax, (0.66, 0.80), (0.23, 0.705), rad=0.0)
arrow(ax, (0.36, 0.63), (0.42, 0.63))
arrow(ax, (0.68, 0.63), (0.74, 0.63))
box(ax, 0.06, 0.31, 0.27, 0.15, "Message graph", "kg.py: per-relation sparse CSR\nheld-out edges never present", BLUE, BLUE_E)
box(ax, 0.38, 0.31, 0.27, 0.15, "BioPathNet++", "model.py: NBFNet path channel\n+ transfer channel, RelSpMM", BLUE, BLUE_E)
box(ax, 0.70, 0.31, 0.27, 0.15, "Trainer", "train.py: listwise loss, early stop,\nresumable (Kaggle 2 x T4)", BLUE, BLUE_E)
arrow(ax, (0.23, 0.555), (0.20, 0.46))
arrow(ax, (0.55, 0.555), (0.52, 0.46))
arrow(ax, (0.33, 0.385), (0.38, 0.385))
arrow(ax, (0.65, 0.385), (0.70, 0.385))
box(ax, 0.765, 0.065, 0.205, 0.15, "Metrics", "metrics.py: full ranking,\nMRR, Hits@k, AUROC", GREEN, GREEN_E)
box(ax, 0.51, 0.065, 0.22, 0.15, "Analysis", "analyze.py: paired tests,\nmacros, tables, figures", GREEN, GREEN_E)
box(ax, 0.265, 0.065, 0.21, 0.15, "Validation", "validate.py: trials, PubMed,\ntemporal, explain.py paths", GREEN, GREEN_E)
box(ax, 0.05, 0.065, 0.18, 0.15, "Outputs", "papers, report,\ncandidate lists", GREY)
arrow(ax, (0.835, 0.31), (0.835, 0.215))
arrow(ax, (0.765, 0.14), (0.73, 0.14))
arrow(ax, (0.51, 0.14), (0.475, 0.14))
arrow(ax, (0.265, 0.14), (0.23, 0.14))
save(fig, "diag_architecture.png")

# 2. Strict-family hold-out ---------------------------------------------------------------------------
fig, ax = canvas(10, 4.6)
import math


def node(x, y, label, fc, ec, r=0.045, size=8.5, bold=False):
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec, lw=1.5, transform=ax.transData))
    ax.text(x, y, label, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal", color=INK)


ax.set_aspect("equal")
ax.set_xlim(0, 2.17)
ax.set_ylim(0, 1)


def link(a, b, color, ls="-", lw=1.4):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=lw, zorder=0)


for panel, x0, title in ((0, 0.02, "Naive hold-out: target node only"), (1, 1.12, "Strict-family hold-out (ours)")):
    ax.text(x0 + 0.5, 0.95, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK)
    T, S1, S2 = (x0 + 0.30, 0.52), (x0 + 0.62, 0.70), (x0 + 0.62, 0.34)
    D1, D2, D3 = (x0 + 0.95, 0.78), (x0 + 0.95, 0.52), (x0 + 0.95, 0.26)
    G = (x0 + 0.10, 0.22)
    link(T, S1, LINE); link(T, S2, LINE); link(T, G, LINE)
    for s in (S1, S2):
        for d in (D1, D2, D3):
            if panel == 0:
                link(s, d, RED_E, lw=1.6)
            else:
                link(s, d, "#BBBBBB", ls=":", lw=1.2)
    for d in (D1, D2):
        link(T, d, "#BBBBBB", ls=":", lw=1.2)
    node(T[0], T[1], "Crohn\ndisease", ORANGE, ORANGE_E, r=0.075, size=7.8, bold=True)
    fam_fc, fam_ec = (GREY, LINE) if panel == 0 else (ORANGE, ORANGE_E)
    node(S1[0], S1[1], "Crohn's\ncolitis", fam_fc, fam_ec, r=0.07, size=7.2)
    node(S2[0], S2[1], "Crohn\nileitis", fam_fc, fam_ec, r=0.07, size=7.2)
    node(G[0], G[1], "gene", BLUE, BLUE_E, r=0.05, size=7.5)
    for d, nm in ((D1, "drug"), (D2, "drug"), (D3, "drug")):
        node(d[0], d[1], nm, GREEN, GREEN_E, r=0.05, size=7.5)
    msg = ("Target's own drug edges removed, but its\nsubtypes still link to the same drugs:\nthe answer leaks back in (red)."
           if panel == 0 else
           "Target + all disease-disease neighbours\nheld out together: no drug edge of the\nfamily is in the graph (dotted = removed).")
    ax.text(x0 + 0.5, 0.06, msg, ha="center", va="bottom", fontsize=8.6, color=MUTED, linespacing=1.35)
save(fig, "diag_strict_family.png")

# 3. One training step (LLD sequence) -----------------------------------------------------------------
fig, ax = canvas(10, 3.3)
steps = [("1. Sample batch", "8 training\ndiseases (query u)"),
         ("2. Drop edges", "drug edges of the\nbatch's families"),
         ("3. Path channel", "6 NBFNet layers\nfrom u (RelSpMM)"),
         ("4. Transfer channel", "MLP over 4\nprior features"),
         ("5. Listwise loss", "softmax over all\n7,957 drugs"),
         ("6. Update", "backward (A_r H\nrecomputed), clip, Adam")]
w, gap = 0.132, 0.034
for i, (t, b) in enumerate(steps):
    x = 0.02 + i * (w + gap)
    fc, ec = (BLUE, BLUE_E) if i in (2, 3) else ((ORANGE, ORANGE_E) if i == 1 else (GREY, LINE))
    box(ax, x, 0.36, w, 0.36, t, b, fc, ec, tsize=9.2, bsize=8.2)
    if i < len(steps) - 1:
        arrow(ax, (x + w + 0.006, 0.54), (x + w + gap - 0.006, 0.54))
ax.text(0.5, 0.17, "Per epoch: evaluate validation MRR -> keep best.pt, write last.pt (resume) -> stop after 6 epochs without improvement or when the session budget ends",
        ha="center", fontsize=8.6, color=MUTED, style="italic")
save(fig, "diag_training_step.png")

# 4. Transfer study design ----------------------------------------------------------------------------
fig, ax = canvas(10, 4.0)
conds = [("A'", "11 non-autoimmune\ndiseases, 3 draws", "160 indications", GREY, LINE),
         ("A", "RA / SLE / MS\nfamilies (11 diseases)", "161 indications", ORANGE, ORANGE_E),
         ("D", "all non-autoimmune\ndiseases", "1,194 diseases", BLUE, BLUE_E),
         ("C_E", "all diseases\nexcept test set E", "1,185 diseases", BLUE, BLUE_E)]
bw, bg = 0.138, 0.03
for i, (k, b, n, fc, ec) in enumerate(conds):
    x = 0.03 + i * (bw + bg)
    box(ax, x, 0.55, bw, 0.27, f"Train: {k}", b + "\n" + n, fc, ec, tsize=10, bsize=8.2)
    ax.plot([x + bw / 2, x + bw / 2], [0.545, 0.42], color="#9A9A9A", lw=1.3)
xs = 0.03 + bw / 2
xe = 0.03 + 3 * (bw + bg) + bw / 2
ax.plot([xs, xe], [0.42, 0.42], color="#9A9A9A", lw=1.3)
arrow(ax, (xe, 0.42), (0.745, 0.50), color="#9A9A9A")
box(ax, 0.745, 0.40, 0.235, 0.33, "Fixed test set E", "12 held-out autoimmune-area\ndiseases (+ 6 rare targets),\nheld out with their families\nin EVERY condition", GREEN, GREEN_E, tsize=10.5, bsize=8.4)
ax.text(0.36, 0.25, "Scarce data:  A vs A'  (same size, same #labels: only disease type differs)", fontsize=9, color=INK, ha="center")
ax.text(0.36, 0.16, "Full scale:  C_E vs D  (does autoimmune data add anything on top of broad data?)", fontsize=9, color=INK, ha="center")
ax.text(0.36, 0.95, "Aim 1 design: only the training diseases change", fontsize=11.5, fontweight="bold", ha="center", va="top", color=INK)
save(fig, "diag_transfer_design.png")

# 5. Execution workflow -------------------------------------------------------------------------------
fig, ax = canvas(10, 3.6)
box(ax, 0.02, 0.40, 0.27, 0.48, "Laptop (RTX 3050, 4 GB)",
    "build KG + splits\nunit tests (pytest)\nbaselines (CPU)\nanalysis, figures, papers\nscoring / explanations", GREY, LINE, tsize=10, bsize=8.6)
box(ax, 0.37, 0.40, 0.27, 0.48, "Kaggle notebook (2 x T4)",
    "dataset: code + compact KG\nqueue of runs, 1 per GPU\nresume from last.pt\nstops before 12 h limit", BLUE, BLUE_E, tsize=10, bsize=8.6)
box(ax, 0.72, 0.40, 0.26, 0.48, "Run outputs",
    "results.json, train logs\nbest.pt / last.pt\nsummary.md\n-> outputs/bpp/kaggle_sessionN", GREEN, GREEN_E, tsize=10, bsize=8.6)
arrow(ax, (0.29, 0.66), (0.37, 0.66), text="upload zip", tpos=0.5)
arrow(ax, (0.64, 0.66), (0.72, 0.66), text="download", tpos=0.5)
arrow(ax, (0.85, 0.40), (0.155, 0.40), rad=-0.25, text="analyze.py -> paper numbers (no hand-typed values)", tpos=0.5, toff=(0, -0.155))
arrow(ax, (0.79, 0.88), (0.50, 0.88), rad=0.22, ls="--", color="#AAAAAA", text="runs zip -> resume next session", tpos=0.5, toff=(0, 0.06))
save(fig, "diag_workflow.png")

# 6. Timeline (Phase 3) -------------------------------------------------------------------------------
import datetime as dt

tasks = [("Validation layers v2, paper draft (old pipeline)", "2026-08-15", "2026-08-28", "#C9C9C9"),
         ("Leakage audit of v1/v2; findings F1-F12", "2026-10-13", "2026-10-15", "#E8A39B"),
         ("v3 rebuild: PrimeKG pipeline, model, baselines", "2026-10-13", "2026-10-15", "#F0B07A"),
         ("Kaggle sessions 1-3: main, ablations, no-autoimmune", "2026-10-14", "2026-10-17", "#8DB4E2"),
         ("Session 4: transfer curve + selection-free test split", "2026-10-17", "2026-10-18", "#8DB4E2"),
         ("Validation layers, path explanations, papers", "2026-10-18", "2026-10-19", "#9CCB9C"),
         ("TxGNN head-to-head (runs 1 and 2)", "2026-10-20", "2026-10-21", "#8DB4E2"),
         ("Session 5: test seeds, paths-only transfer", "2026-10-22", "2026-10-22", "#8DB4E2"),
         ("Plagiarism/AI check; report and slides", "2026-10-22", "2026-11-10", "#9CCB9C")]
fig = plt.figure(figsize=(10, 3.8))
ax = fig.add_axes([0.42, 0.14, 0.55, 0.80])
d0 = dt.date(2026, 8, 10)
for i, (name, a, b, c) in enumerate(tasks):
    a, b = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    ax.barh(i, (b - a).days + 1, left=(a - d0).days, color=c, edgecolor="#666666", height=0.6)
    fig.text(0.41, 0.94 - (i + 0.5) * 0.80 / len(tasks), name, ha="right", va="center", fontsize=8.6, color=INK)
ax.set_ylim(len(tasks) - 0.5, -0.5)
ax.set_yticks([])
ticks = [dt.date(2026, 8, 15), dt.date(2026, 9, 1), dt.date(2026, 9, 15), dt.date(2026, 10, 1), dt.date(2026, 10, 15), dt.date(2026, 11, 1)]
ax.set_xticks([(t - d0).days for t in ticks])
ax.set_xticklabels([t.strftime("%d %b") for t in ticks], fontsize=8.5)
ax.set_xlim(0, (dt.date(2026, 11, 15) - d0).days)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#E5E5E5")
ax.set_axisbelow(True)
save(fig, "diag_timeline.png")

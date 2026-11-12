"""Build the SmartCom 2027 (Springer LNNS, svproc class) version of the conference paper.

    python outputs/paper/v3/make_smartcom.py      # writes outputs/paper/v3/bpp_v3_smartcom.tex

The body is taken from bpp_v3_conf.tex (so both versions stay in sync with the same generated numbers) and
converted to Springer's single-column proceedings format, following the SmartCom author guidelines
(smartcomconference.com/author-guidelines-for-paper-submission, read 12 Nov 2026):
  - svproc.cls from the conference's LaTeX package; references author-year, alphabetical (spbasic.bst + natbib)
  - abstract <= 250 words, no citations, no mathematical expressions, no first person; 4-6 keywords
  - corresponding author's e-mail on the title page; acknowledgment as a run-in heading
  - Springer: table captions above, figure captions below, figure lettering >= 6 pt, title-case headings
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
conf = (HERE / "bpp_v3_conf.tex").read_text(encoding="utf-8")
body = conf[conf.index(r"\section{Introduction}"):conf.index(r"\bibliographystyle")]


def rep(old, new, count=1):
    global body
    assert body.count(old) == count, (body.count(old), old[:80])
    body = body.replace(old, new)


# author-year citations in parentheses: "TxGNN (Huang et al., 2024)"
body = body.replace(r"\cite{", r"\citep{")

# architecture diagram: the IEEE version is a full-width strip (7.16 in); Springer's text block is 12.2 cm, so use
# the portrait layout of the same diagram drawn at that width (bpp/figures.py::architecture_tall, labels >= 6 pt),
# upright, as an ordinary float placed just before the Method section
a = body.index("% full-width figure set exactly here")
b = body.index(r"\end{strip}") + len(r"\end{strip}")
strip = body[a:b]
cap = strip[strip.index(r"\captionof{figure}{") + len(r"\captionof{figure}"):strip.index(r"\label{fig:arch}")].strip()
body = body[:a] + (r"""\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{fig_architecture_tall.pdf}
\caption""" + cap + "\n" + r"""\label{fig:arch}
\end{figure}""") + body[b:]

# 12-column results table: upright at \scriptsize with tight columns (no landscape page, no shrinking)
rep(r"""\begin{table*}[t]
\centering
\caption{Aim 2: ranking""", r"""\begin{table}[!htbp]
\centering
\caption{Aim 2: ranking""")
# the "Set" column becomes a group-label row and the long headers take two lines, so the 11 metric columns fit
SETS = {"Test (68, never selected)": "Test (68 diseases)", "Validation (68)": "Validation (68 diseases)",
        "Rare targets (6)": "Rare targets (6 diseases)"}
rows = []
for line in (HERE / "generated" / "tab_metrics.tex").read_text(encoding="utf-8").strip().splitlines():
    if line.startswith(r"\begin{tabular}"):
        assert line == r"\begin{tabular}{llcccccccccc}", line
        rows.append(r"\begin{tabular}{lcccccccccc}")
    elif line.startswith("Set & Method"):
        rows.append(r"Method & Any@10 & H@1 & H@10 & H@50 & AUROC & AUROC & AUROC & AUPRC & MRR & Mean\\"
                    + "\n" + r" & & & & & & hard & deg. & & & rank\\")
    elif any(line.startswith(k + " & ") for k in SETS):
        k = next(k for k in SETS if line.startswith(k + " & "))
        rows.append(r"\multicolumn{11}{l}{\textit{" + SETS[k] + r"}}\\" + "\n" + line[len(k) + 3:])
    elif line.startswith(" & "):
        rows.append(line[3:])
    else:
        rows.append(line)
metrics = "\n".join(rows)
assert metrics.count(r"\multicolumn{11}") == 3
rep(r"""\resizebox{\textwidth}{!}{\input{generated/tab_metrics.tex}}
\end{table*}""", "\\setlength{\\tabcolsep}{2pt}\n" + metrics + "\n\\end{table}")
rep(r"Test: seed means", r"AUROC hard/deg.: against drugs indicated for autoimmune diseases / degree-matched drugs. "
    r"Test diseases were never used for model selection. Test: seed means")

# unbreakable DOI token -> breakable link
rep(r"(Harvard Dataverse, doi:10.7910/DVN/IXA7BM)", r"(Harvard Dataverse, \url{https://doi.org/10.7910/DVN/IXA7BM})")

# Table 1: author-year citations do not fit table cells; the same works are cited in the Related Work text
tab = body[body.index(r"\label{tab:related}"):body.index(r"\end{table}", body.index(r"\label{tab:related}"))]
body = body.replace(tab, re.sub(r"~?\\citep\{[^}]*\}", "", tab))

# single-column page: the compact transfer figure at a readable size; small tables a little larger
rep(r"\includegraphics[width=\columnwidth]{fig_transfer_curve_col.pdf}", r"\includegraphics[width=0.78\textwidth]{fig_transfer_curve_col.pdf}")
body = body.replace(r"\begin{table}[t]", r"\begin{table}[tb]").replace(r"\begin{table}[!h]", r"\begin{table}[tb]").replace(r"\begin{table}[H]", r"\begin{table}[tb]")
body = body.replace(r"\begin{figure}[t]", r"\begin{figure}[tb]")
body = re.sub(r"(\\caption\{[^\n]*\}\n\\label\{tab:(?:related|metricdefs|validation)\}\n)\\scriptsize", r"\1\\small", body)
rep(r"\begin{tabular}{p{1.1cm}p{5.5cm}l}", r"\begin{tabular}{p{1.4cm}p{7.6cm}l}")

# Springer headings: title case
for old, new in ((r"\subsection{Aim 1: transfer to rare autoimmune diseases}", r"\subsection{Aim 1: Transfer to Rare Autoimmune Diseases}"),
                 (r"\subsection{Aim 2: accuracy on unseen diseases}", r"\subsection{Aim 2: Accuracy on Unseen Diseases}"),
                 (r"\subsection{Aim 3: what each component contributes}", r"\subsection{Aim 3: Contribution of Each Component}"),
                 (r"\subsection{Case studies and validation}", r"\subsection{Case Studies and Validation}")):
    rep(old, new)

# acknowledgment: run-in heading, not a numbered section
ack = body[body.index(r"\section*{Acknowledgment}"):]
body = body.replace(ack, "\\paragraph{Acknowledgments.} Generative AI tools assisted with code, analysis and language "
                         "editing; the authors verified all results against the released run logs.\n\n")

ABSTRACT = r"""Rare autoimmune diseases have few approved drugs, so computational drug repurposing must rank candidate drugs for diseases whose treatments were never seen during training. This zero-shot setting is studied with \sysname, a path-based graph neural network in the style of BioPathNet, extended with a channel of four leakage-free prior features, on the PrimeKG biomedical knowledge graph. A strict protocol holds out each test disease together with its related diseases and ranks all 7,957 drugs. On 68 unseen test diseases, a known indication appears in the top ten for \TabModelTestAnyTen\% of diseases, against \TabKnnTestAnyTen\% for the best baseline and \RandExpAnyTenTest\% by chance, and the mean reciprocal rank rises from \KnnTestMrr{} to \TestMrr, a statistically significant gain. When training data are scarce, data-rich autoimmune diseases transfer better than label-matched non-autoimmune diseases; with the full graph they bring no benefit and even lower accuracy. Ablations show that path reasoning carries the accuracy, while the prior channel mainly stabilises training across random seeds. Registered clinical trials test \ValTrialTestedModel\% of the new candidates, against \ValTrialTestedPop\% of popular drugs. The output is a ranked list of hypotheses for expert review, not treatment advice."""

HEAD = r"""% SmartCom 2027 (Springer LNNS) version -- GENERATED by make_smartcom.py from bpp_v3_conf.tex; edit that file, then rerun.
\documentclass{svproc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xspace}
\usepackage[table]{xcolor}
\usepackage[authoryear,round]{natbib}
\usepackage{xurl}
\def\UrlFont{\rmfamily}
\graphicspath{{figures/}}
\newcommand{\sysname}{BioPathNet++\xspace}
\input{generated/numbers.tex}
\emergencystretch=1.5em
\renewcommand{\bibname}{References}
\providecommand{\refname}{References}\renewcommand{\refname}{References}

\begin{document}
\mainmatter
\title{BioPathNet++ for Autoimmune Diseases: Path-Based Transfer Learning for Rare Disease Drug Repurposing}
\titlerunning{BioPathNet++ for Rare Autoimmune Disease Drug Repurposing}
\author{TVNS Yatin\textsuperscript{*} \and Srijan Jha \and A~Charan~KR \and Praveen Kumar Yadav \and Priyanka~H}
\authorrunning{TVNS Yatin et al.}
\tocauthor{TVNS Yatin, Srijan Jha, A Charan KR, Praveen Kumar Yadav, Priyanka H}
\institute{Department of Computer Science and Engineering, PES University, Bengaluru, India\\
\textsuperscript{*}Corresponding author: \email{sumakkiyatin@gmail.com}}
\maketitle

\begin{abstract}
""" + ABSTRACT + r"""
\keywords{Drug repurposing $\cdot$ Knowledge graphs $\cdot$ Graph neural networks $\cdot$ Zero-shot learning $\cdot$ Transfer learning $\cdot$ Autoimmune diseases}
\end{abstract}

"""
TAIL = r"""\bibliographystyle{spbasic}
\bibliography{references}

\end{document}
"""
(HERE / "bpp_v3_smartcom.tex").write_text(HEAD + body + TAIL, encoding="utf-8")
words = len(re.sub(r"\\[A-Za-z]+|[{}$]", " ", ABSTRACT).split())
print(f"wrote bpp_v3_smartcom.tex | abstract ~{words} words")

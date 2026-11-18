"""Build the IEEE Access version of the journal paper.

    python outputs/paper/v3/make_access.py <official template dir>

Writes a self-contained folder outputs/paper/ieee_access/ (bpp_v3_access.tex + the UNMODIFIED official template files
+ figures + generated tables + references.bib). The body is taken from bpp_v3_journal.tex, so both versions keep the
same generated numbers; edit the journal, then rerun. Rules applied (IEEE Access author page + template, 17 Nov 2026):
  - ieeeaccess.cls (double column); compile with pdfLaTeX (the class uses pdfTeX spot colour and font maps)
  - abstract 150-250 words, one paragraph, no abbreviations, footnotes or references
  - keywords in alphabetical order (3-10 at submission)
  - full author names; one corresponding author at submission (more can be added at the proof stage)
  - AI-generated text disclosed in the acknowledgements; short biographies for ALL authors after the references
"""
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
TPL = Path(sys.argv[1])
OUT = HERE.parent / "ieee_access"
journal = (HERE / "bpp_v3_journal.tex").read_text(encoding="utf-8")
body = journal[journal.index(r"\section{Introduction}"):journal.index(r"\bibliographystyle")]


def rep(old, new, count=1):
    global body
    assert body.count(old) == count, (body.count(old), old[:80])
    body = body.replace(old, new)


rep(r"\IEEEPARstart{D}{rug} ", "Drug ")          # IEEE Access has no drop cap
# IEEE Access columns are 85.3 mm (IEEEtran journal: 88.9 mm): narrow Table I's last column so it fits
rep(r"p{2.35cm}>{\raggedright\arraybackslash}p{4.1cm}}", r"p{2.35cm}>{\raggedright\arraybackslash}p{3.7cm}}")

ABSTRACT = (r"Rare autoimmune diseases have few approved drugs, so computational drug repurposing must rank candidate "
            r"drugs for diseases whose treatments are unknown. This zero-shot setting is studied on the PrimeKG "
            r"biomedical knowledge graph under a leakage-controlled protocol that holds out each test disease together "
            r"with its related diseases and ranks all 7,957 drugs. The model, \sysname, reuses the path-based "
            r"BioPathNet architecture and adds a channel of four leakage-free prior features; no new network layer is "
            r"proposed. The contributions are the protocol, a controlled transfer study, a few-shot evaluation, a "
            r"head-to-head comparison with TxGNN and an external validation of new candidates. On 68 unseen test "
            r"diseases a known indication is ranked in the top ten for \TabModelTestAnyTen\% of diseases "
            r"(\RandExpAnyTenTest\% by chance), and the mean reciprocal rank of \TestMrr{} exceeds that of every "
            r"baseline, including knowledge-graph embeddings (best baseline \KnnTestMrr). When training data are "
            r"scarce, autoimmune source diseases transfer better than label-matched non-autoimmune controls "
            r"(\TrAEMrr{} vs.\ \TrActlEMrr); with the full graph they even lower accuracy (\TrCEEMrr{} vs.\ "
            r"\TrDEMrr). Revealing three known drugs of a disease raises the mean reciprocal rank from "
            r"\FsBppZeroMrr{} to \FsBppThreeMrr{} without retraining, and fusing the model with co-indication "
            r"filtering reaches \FsRrfThreeMrr{} (best baseline \FsKnnCoindThreeMrr). On TxGNN's own split the model "
            r"ranks all drugs significantly better. The ranked lists are hypotheses for expert review, not treatment "
            r"advice.")
KEYWORDS = ("Autoimmune diseases, data leakage, drug repurposing, few-shot learning, graph neural networks, knowledge "
            "graphs, PrimeKG, rare diseases, transfer learning, zero-shot learning")

AUTHORS = ["Venkata Naga Sai Yatin Tatikonda", "Srijan Jha", "Charan Kumar Reddy Arikeri", "Praveen Kumar Yadav",
           "Priyanka H"]
byline = ", ".join(r"\uppercase{%s}\authorrefmark{1}" % a for a in AUTHORS[:-1]) + \
         r", AND \uppercase{%s}\authorrefmark{1}" % AUTHORS[-1]

# Biographies: only facts the authors gave us; each author must check and complete their own before submission.
BIOS = [
    (AUTHORS[0], "is pursuing the B.Tech.\\ degree in computer science and engineering at PES University, Bengaluru, "
                 "India. Research interests include biomedical knowledge graphs, graph neural networks and drug "
                 "repurposing."),
    (AUTHORS[1], "is pursuing the B.Tech.\\ degree in computer science and engineering (artificial intelligence and "
                 "machine learning) at PES University, Bengaluru, India. Research interests include machine learning "
                 "for biomedical data."),
    (AUTHORS[2], "is pursuing the B.Tech.\\ degree in computer science and engineering at PES University, Bengaluru, "
                 "India. Research interests include machine learning for biomedical data."),
    (AUTHORS[3], "is pursuing the B.Tech.\\ degree in computer science and engineering at PES University, Bengaluru, "
                 "India. Research interests include machine learning for biomedical data."),
    (AUTHORS[4], "is an Associate Professor with the Department of Computer Science and Engineering, PES University, "
                 "Bengaluru, India."),
]
bios = "\n\n".join("%% TO COMPLETE BY THE AUTHOR: degree year, other work, research interests\n"
                   "\\begin{IEEEbiographynophoto}{%s} %s\n\\end{IEEEbiographynophoto}" % (n, b) for n, b in BIOS)

TITLE = journal[journal.index(r"\title{") + len(r"\title{"):]
TITLE = TITLE[:TITLE.index("}\n")]

HEAD = r"""%% IEEE Access version -- GENERATED by make_access.py from bpp_v3_journal.tex; edit that file, then rerun.
%% Compile with pdfLaTeX (Overleaf: Menu > Compiler > pdfLaTeX); ieeeaccess.cls needs pdfTeX.
\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{xspace}
\usepackage{xurl}
\usepackage{flafter}   % a float never appears before the text that first mentions it
\graphicspath{{figures/}}
\hyphenation{Table Tables Figure Figures ri-san-ki-zu-mab up-a-da-ci-ti-nib gu-sel-ku-mab ra-vu-li-zu-mab
  ro-za-no-li-xi-zu-mab el-a-fi-bra-nor us-te-ki-nu-mab am-i-fam-pri-dine}
\newcommand{\sysname}{BioPathNet++\xspace}
\input{generated/numbers.tex}
% bold math in the template's Times (from the official template)
\makeatletter
\AtBeginDocument{\DeclareMathVersion{bold}
\SetSymbolFont{operators}{bold}{T1}{times}{b}{n}
\SetSymbolFont{NewLetters}{bold}{T1}{times}{b}{it}
\SetMathAlphabet{\mathrm}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathit}{bold}{T1}{times}{b}{it}
\SetMathAlphabet{\mathbf}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathtt}{bold}{OT1}{pcr}{b}{n}
\SetSymbolFont{symbols}{bold}{OMS}{cmsy}{b}{n}
\renewcommand\boldmath{\@nomath\boldmath\mathversion{bold}}}
\makeatother

\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2024.0429000}

\title{""" + TITLE + r"""}
\author{""" + byline + r"""}
\address[1]{Department of Computer Science and Engineering, PES University, Bengaluru, Karnataka, India}

\markboth
{Tatikonda \headeretal: Leakage-Controlled Drug Repurposing for Rare Autoimmune Diseases}
{Tatikonda \headeretal: Leakage-Controlled Drug Repurposing for Rare Autoimmune Diseases}

\corresp{Corresponding author: Venkata Naga Sai Yatin Tatikonda (e-mail: yatin.tvns@gmail.com).}

\begin{abstract}
""" + ABSTRACT + r"""
\end{abstract}

\begin{keywords}
""" + KEYWORDS + r"""
\end{keywords}

\titlepgskip=-21pt

\maketitle

"""
TAIL = r"""\bibliographystyle{IEEEtran}
\bibliography{references}

""" + bios + r"""

\vfill  % keep the biographies together at the top of the last column instead of spreading them out

\EOD

\end{document}
"""

# ---- assemble the self-contained folder
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
for f in TPL.iterdir():                                   # official template support files, unmodified
    if f.suffix in (".cls", ".sty", ".bst", ".fd", ".map", ".pfb", ".tfm") or f.name in ("logo.png",
                                                                                          "notaglinelogo.png",
                                                                                          "bullet.png"):
        shutil.copy2(f, OUT / f.name)
tex = HEAD + body + TAIL
(OUT / "bpp_v3_access.tex").write_text(tex, encoding="utf-8")
shutil.copy2(HERE / "references.bib", OUT / "references.bib")
(OUT / "generated").mkdir()
for g in ["numbers.tex"] + [m + ".tex" for m in re.findall(r"\\input\{generated/([A-Za-z_]+)\.tex\}", tex)]:
    shutil.copy2(HERE / "generated" / g, OUT / "generated" / g)
(OUT / "figures").mkdir()
for fig in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
    src = HERE / "figures" / fig
    if not src.exists():
        src = HERE / "figures" / (fig + ".pdf")
    shutil.copy2(src, OUT / "figures" / src.name)
words = len(re.sub(r"\\[A-Za-z]+|[{}$]", " ", ABSTRACT).split())
print(f"wrote {OUT / 'bpp_v3_access.tex'} | abstract ~{words} words | keywords {len(KEYWORDS.split(','))}")

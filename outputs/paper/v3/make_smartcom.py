"""Build the SmartCom 2027 (Springer LNNS, svproc class) version of the conference paper.

    python outputs/paper/v3/make_smartcom.py      # writes outputs/paper/v3/bpp_v3_smartcom.tex

The body is taken from bpp_v3_conf.tex (so both versions stay in sync with the same generated numbers) and
converted to Springer's single-column proceedings format, following the SmartCom author guidelines
(smartcomconference.com/author-guidelines-for-paper-submission, read 12 Nov 2026):
  - svproc.cls from the conference's LaTeX package; numbered citations [1] (Springer LaTeX instructions: "by number
    (preferred)"), list sorted alphabetically (spmpsci.bst, MathPhySci style recommended for computer science)
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
# upright, pinned with [H] exactly where it is defined: just before the Method heading
a = body.index("% full-width figure set exactly here")
b = body.index(r"\end{strip}") + len(r"\end{strip}")
strip = body[a:b]
cap = strip[strip.index(r"\captionof{figure}{") + len(r"\captionof{figure}"):strip.index(r"\label{fig:arch}")].strip()
body = body[:a] + (r"""\begin{figure}[H]
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


# single-column page: the compact transfer figure at a readable size; small tables a little larger
rep(r"\includegraphics[width=\columnwidth]{fig_transfer_curve_col.pdf}", r"\includegraphics[width=0.62\textwidth]{fig_transfer_curve_col.pdf}")
body = body.replace(r"\begin{table}[t]", r"\begin{table}[tb]").replace(r"\begin{table}[!h]", r"\begin{table}[tb]")
body = body.replace(r"\begin{figure}[t]", r"\begin{figure}[tb]")
body = re.sub(r"(\\caption\{[^\n]*\}\n\\label\{tab:(?:related|metricdefs|validation)\}\n)\\scriptsize", r"\1\\footnotesize", body)
rep(r"\begin{tabular}{p{1.1cm}p{5.5cm}l}", r"\begin{tabular}{p{1.2cm}p{8.7cm}l}")

# SmartCom charges per page above 10: the same content in fewer words (no result, table or figure removed)
CONDENSE = [
    (r"Drug repurposing finds new uses for approved drugs, whose safety is already known, and is faster and cheaper than de~novo development",
     r"Repurposing approved drugs, whose safety is known, is faster and cheaper than de~novo development"),
    (r"Autoimmune disorders affected 10.2\% of a 22-million-person UK cohort~\citep{conrad2023autoimmune}, yet primary",
     r"Autoimmune disorders affect 10.2\% of a 22-million-person UK cohort~\citep{conrad2023autoimmune}, yet primary"),
    (r"Knowledge graphs (KGs) link drugs, proteins, phenotypes and diseases, and link prediction on them ranks candidate drugs",
     r"Link prediction on knowledge graphs (KGs) of drugs, proteins, phenotypes and diseases ranks candidate drugs"),
    (r"Because autoimmune diseases share immune pathways, the best source of that knowledge might be the data-rich ones: rheumatoid arthritis (RA), lupus (SLE) and multiple sclerosis (MS). We ask three questions.",
     r"As autoimmune diseases share immune pathways, the best source might be the data-rich ones: rheumatoid arthritis (RA), lupus (SLE) and multiple sclerosis (MS)."),
    (r"does a path-based model beat simple repurposing rules on unseen diseases?",
     r"does a path-based model beat simple rules on unseen diseases?"),
    (r"what do path reasoning, the prior channel, the loss and the graph each contribute?",
     r"what do paths, the prior channel, the loss and the graph contribute?"),
    (r"that also hides each test disease's related diseases, motivated by a leakage audit of our own earlier pipeline;",
     r"that also hides each test disease's relatives, motivated by a leakage audit of our earlier pipeline;"),
    (r"(iii)~a full-graph implementation whose memory is linear in the number of nodes, plus a prior channel that stabilizes training;",
     r"(iii)~a full-graph implementation with memory linear in the number of nodes, plus a stabilizing prior channel;"),
    (r"The rare targets have so few that their labels are indications plus off-label uses:",
     r"The rare targets use indications plus off-label uses:"),
    (r"Contraindications are never positives. With so few labels, the targets are case studies; statistical claims rest on larger held-out sets.",
     r"Contraindications are never positives. The targets are case studies; statistics rest on larger held-out sets."),
    (r"PrimeKG's file lists every edge in both directions (8,100,498 rows). We keep one canonical direction per relation, drop duplicates, and add an explicit inverse for every relation (60 relation types in all). Drug--disease pairs are kept in a separate table, and each split's message graph receives only those of its training diseases.",
     r"PrimeKG lists every edge in both directions (8,100,498 rows); we keep one direction per relation, drop duplicates and add explicit inverses (60 relation types). Drug--disease pairs sit in a separate table; a split's message graph receives only those of its training diseases."),
    (r"Without this rule, all 16 Crohn indications stayed reachable through ``Crohn's colitis''. No drug edge of a held-out family enters the message graph, and each training batch also hides its own diseases' drug edges, so training mimics zero-shot inference. An earlier version of our pipeline broke these rules: test pairs were graph edges, and a degree score with no learning reached AUROC 0.972.",
     r"Without this rule, all 16 Crohn indications stayed reachable through ``Crohn's colitis''. No drug edge of a held-out family enters the message graph, and each training batch hides its own diseases' drug edges, mimicking zero-shot inference. Our earlier pipeline broke these rules: test pairs were graph edges, and a degree score with no learning reached AUROC 0.972."),
    (r"A vs.\ A$'$ asks whether autoimmune data helps when data is scarce, C$_E$ vs.\ D whether it still helps at scale.",
     r"A vs.\ A$'$ asks whether autoimmune data helps when data is scarce, C$_E$ vs.\ D whether it helps at scale."),
    (r"The messages use one sparse matrix per relation and a custom autograd function that recomputes products in the backward pass, so memory grows with nodes, not edges.",
     r"One sparse matrix per relation and a custom autograd function that recomputes products in the backward pass make memory grow with nodes, not edges."),
    (r"Every drug is ranked, and ranks are \emph{filtered}: a disease's other known drugs are removed before ranking each one~\citep{bordes2013transe}.",
     r"All drugs are ranked with \emph{filtered} ranks (other known drugs removed)~\citep{bordes2013transe}."),
    (r"The 68 validation diseases select the stopping epoch; 68 further \emph{test} diseases are used neither for training nor for stopping. Of six non-neural baselines and three KG embeddings (TransE, DistMult, RotatE, trained on the same graph), the strongest is disease $k$NN, which transfers the drugs of the 20 most similar training diseases.",
     r"The 68 validation diseases select the stopping epoch; 68 further \emph{test} diseases are used for neither training nor stopping. Of six non-neural baselines and three KG embeddings (TransE, DistMult, RotatE, same graph), the strongest is disease $k$NN, which transfers the drugs of the 20 most similar training diseases."),
    (r"Code, splits, logs and the script that generates every number:", r"Code, splits, logs and number-generating scripts:"),
    (r"which also holds for final instead of early-stopped checkpoints", r"also with final checkpoints"),
    (r"What transfers is a drug-class prior: the A-trained model ranks the same corticosteroids first for every target, a sensible first guess but not disease-specific reasoning.",
     r"What transfers is a drug-class prior: the A model ranks the same corticosteroids first for every target, not disease-specific reasoning."),
    (r"The best source thus depends on data size: related autoimmune diseases when data is scarce, the broad pharmacological graph at full scale.",
     r"The best source thus depends on data size."),
    (r"\sysname therefore outperforms every baseline evaluated on the same leakage-free split. On the six targets it is ahead for",
     r"\sysname thus outperforms every baseline evaluated on the same leakage-free split; on the six targets it is ahead for"),
    (r"These validation values are not comparable with the test-split MRR of \TestMrr{}: that is the full model on 68 \emph{other} diseases, which are easier for every method",
     r"These are not comparable with the test MRR of \TestMrr{}: the test diseases are easier for every method"),
    (r"The prior channel does not raise the mean but reduces seed variance on the targets",
     r"The prior channel does not raise the mean but cuts seed variance on the targets"),
    (r"Ursodeoxycholic acid ranks first for PBC in every seed, and the four labeled cholinesterase inhibitors hold ranks 1--4 for MG. AIH scores near zero only because PrimeKG lists one off-label drug for it, while the model's top choice, prednisolone, is standard AIH therapy (comparator in NCT00838214).",
     r"Ursodeoxycholic acid ranks first for PBC in every seed; the four labeled cholinesterase inhibitors hold ranks 1--4 for MG. AIH scores near zero only because PrimeKG lists one off-label drug, while the top choice, prednisolone, is standard AIH therapy (comparator in NCT00838214)."),
    (r"We searched ClinicalTrials.gov for each candidate and for two reference sets (popularity top 20; 20 random drugs per target). The drug must be a named intervention, matched by whole name",
     r"We searched ClinicalTrials.gov for each candidate and two reference sets (popularity top 20; 20 random drugs per target), requiring a named intervention matched by whole name"),
    (r"A trial \emph{tests} the drug only if its title names it or the drug is in some but not all arms, which excludes background therapy, premedication and transplant conditioning.",
     r"A trial \emph{tests} the drug only if its title names it or the drug is in some but not all arms (excluding background therapy, premedication and conditioning)."),
    (r"PubMed titles were classified with AI assistance and every benefit or harm label was re-checked against the abstract.",
     r"PubMed titles were classified with AI assistance; every benefit or harm label was re-checked against the abstract."),
    (r"Gradient-based path extraction~\citep{zhu2021nbfnet} shows, for example, that amifampridine reaches MG through its indication for Lambert--Eaton syndrome, a related neuromuscular-junction disease, and exposes weak links such as ciprofloxacin reaching Crohn through ``mouth disease''.",
     r"Gradient-based path extraction~\citep{zhu2021nbfnet} shows that amifampridine reaches MG through its Lambert--Eaton indication, a related neuromuscular-junction disease, and exposes weak links such as ciprofloxacin reaching Crohn through ``mouth disease''."),
    (r"Our MRR values lie within the range BioPathNet reports on TxGNN's disease areas (about 0.2--0.8)~\citep{hu2026biopathnet}, but splits differ, so we ran TxGNN's own code on its autoimmune split, with our model trained only on edges that TxGNN also trains on",
     r"Our MRR values lie within BioPathNet's range on TxGNN's disease areas (about 0.2--0.8)~\citep{hu2026biopathnet}, but splits differ, so we ran TxGNN's own code on its autoimmune split, training our model only on edges TxGNN also trains on"),
    (r"Ranking only drugs with indications, as TxGNN does, no difference is significant",
     r"Ranking only drugs with indications, as TxGNN does, nothing is significant"),
    (r"When training data is scarce, related autoimmune diseases are the most useful source; at full scale the broad pharmacological graph is the better source.",
     r"With scarce data, related autoimmune diseases are the most useful source; at full scale, the broad pharmacological graph."),
    (r"Aim 2: ranking of all 7,957 drugs. Any@10: diseases with a known indication in the top 10 (random \RandExpAnyTenTest\%); H@$k$: known drugs in the top $k$; random AUPRC $\approx$0.001. AUROC hard/deg.",
     r"Aim 2: ranking of all 7,957 drugs (metrics: Table~\ref{tab:metricdefs}). AUROC hard/deg."),
    (r"Test diseases were never used for model selection. Test: seed means", r"Test: never used for model selection; seed means"),
    (r"\sysname architecture. Six NBF layers pass DistMult messages from the query disease $u$ over a graph without the drug edges of $u$'s family; a path MLP and a prior MLP (four leakage-free priors) give two scores whose sum ranks all 7,957 drugs.",
     r"\sysname architecture: six NBF layers pass DistMult messages from the query disease $u$ over a graph without the drug edges of $u$'s family; path and prior scores are summed to rank all 7,957 drugs."),
    (r"Their reported gains depend strongly on how the dataset is built and evaluated~\citep{bonner2022kge,ali2021pykeen}, and because they learn one vector per node from its training edges, they degrade for diseases without known drugs~\citep{huang2024txgnn}.",
     r"Their gains depend strongly on dataset construction and evaluation~\citep{bonner2022kge,ali2021pykeen}, and with one vector learned per node they degrade for diseases without known drugs~\citep{huang2024txgnn}."),
    (r"NBFNet scores a (disease, drug) pair by Bellman--Ford message passing that starts at the disease, which makes it inductive and explainable by paths~\citep{zhu2021nbfnet};",
     r"NBFNet scores a (disease, drug) pair by Bellman--Ford message passing from the disease, which is inductive and explainable by paths~\citep{zhu2021nbfnet};"),
    (r" Table~\ref{tab:related} positions \sysname: it reuses BioPathNet's model and adds a stricter hold-out unit and a controlled transfer study.", r""),
    (r"one NVIDIA T4 (16\,GB) per run, 8.5\,s per step, 5.8\,GB peak.", r"one NVIDIA T4 per run (5.8\,GB peak)."),
    (r"The rare targets point the same way", r"The targets point the same way"),
    (r"On the targets, removing all autoimmune diseases leaves MRR unchanged", r"On the targets, removing all autoimmune diseases changes nothing"),
    (r"Path reasoning carries the signal. On the 68 validation diseases, paths alone reach", r"Paths carry the signal: on the 68 validation diseases, paths alone reach"),
    (r"A trial is not proof of benefit: amifampridine helped in MuSK-antibody MG~\citep{bonanno2018amifampridine} but not in acetylcholine-receptor-antibody MG~\citep{remijn2026amifampridine}.",
     r"A trial is no proof of benefit: amifampridine helped in MuSK-antibody MG~\citep{bonanno2018amifampridine} but not in AChR-antibody MG~\citep{remijn2026amifampridine}."),
    (r"three corticosteroid pairs and N-acetylcysteine for SSc, whereas popularity's are mostly antibiotics that can worsen MG or induce hepatitis.",
     r"three corticosteroid pairs and N-acetylcysteine for SSc; popularity's are mostly antibiotics that can worsen MG or induce hepatitis."),
    (r"Limitations: (i) the targets have 1--24 labeled drugs, so their metrics are noisy and can be lower bounds (AIH); (ii) the validation diseases also choose the stopping epoch",
     r"Limitations: (i) the targets have 1--24 labeled drugs, so their metrics are noisy lower bounds (AIH); (ii) validation diseases also choose the stopping epoch"),
    (r"(iii) $E$ has only 12 diseases, and 11-disease models are unstable across seeds; (iv) literature labels have not been reviewed by a clinician;",
     r"(iii) $E$ has 12 diseases, and 11-disease models are seed-unstable; (iv) no clinician has reviewed the literature labels;"),
    (r"\sysname outperforms every baseline evaluated on the same leakage-free split, recovers established therapies, and ranks later-approved drugs high but not significantly above chance. Path reasoning drives accuracy, and the prior channel adds stability.",
     r"\sysname outperforms every baseline evaluated on the same leakage-free split, recovers established therapies, and ranks later-approved drugs high but not significantly above chance. Paths drive accuracy; the prior channel adds stability."),
    (r"Related approaches. KGE: KG embeddings. Induct.: scores diseases with no known drugs. Hold-out: unit removed for zero-shot testing (area: disease area). Transfer: controlled study of the training source.",
     r"Related approaches. Induct.: scores diseases with no known drugs. Hold-out: unit removed for zero-shot testing. Transfer: controlled study of the training source."),
    (r"Metrics, per disease over all 7,957 drugs, averaged over diseases. Random: value for a random ranking.",
     r"Metrics (per disease over all 7,957 drugs, then averaged) and their value for a random ranking."),
    (r"Evidence for unlabeled top-20 candidates (6 targets) and two reference sets (Ours $n$=\ValNovelN; Pop., Random $n$=120; grades as in the text).",
     r"Evidence for unlabeled top-20 candidates (Ours, $n$=\ValNovelN) and reference sets ($n$=120)."),
    (r"knowledge learned from diseases with known drugs must \emph{transfer} to diseases whose drugs the model has never seen.",
     r"knowledge must \emph{transfer} from diseases with known drugs."),
    (r"\emph{controlled transfer study} that compares autoimmune source diseases with label-matched non-autoimmune controls;",
     r"\emph{controlled transfer study} against label-matched non-autoimmune controls;"),
    (r"compares the approaches discussed below.", r"compares the approaches below."),
    (r"Splits: validation seed 0, test seed 1, matched controls seeds 100--102, all stored as JSON.", r"Split seeds: validation 0, test 1, controls 100--102 (JSON)."),
    (r"The path channel uses no node features or text embeddings: nodes differ only by their edges (\texttt{bpp/build\_primekg.py}).",
     r"The path channel uses no node features: nodes differ only by their edges."),
    (r" The best source thus depends on data size.", r""),
    (r"(risankizumab, upadacitinib, guselkumab for Crohn; ravulizumab, rozanolixizumab for MG; elafibranor for PBC)",
     r"(three for Crohn, two for MG, one for PBC)"),
    (r"The targets point the same way (\TrATargetsMrr{} vs.\ \TrActlTargetsMrr; $p$\,\PwilTrAActlTargets, $n=6$), and so does disease $k$NN (\TrAEKnnMrr{} vs.\ \TrActlEKnnMrr).",
     r"The targets (\TrATargetsMrr{} vs.\ \TrActlTargetsMrr; $p$\,\PwilTrAActlTargets, $n=6$) and disease $k$NN (\TrAEKnnMrr{} vs.\ \TrActlEKnnMrr) point the same way."),
    (r"rank at a median of \TempPostMedRank{} of 7,957; \TempPostBetter{} of 6 beat chance, not significant ($p$\,\TempPostP).",
     r"rank at a median of \TempPostMedRank{} of 7,957 (\TempPostBetter{} of 6 above chance; $p$\,\TempPostP)."),
    (r"against \ValTrialTestedPop\% for popularity and \ValTrialTestedRand\% for random drugs;", r"against \ValTrialTestedPop\% (popularity) and \ValTrialTestedRand\% (random);"),
    (r"PubMed shows benefit for \ValLitSupModel\% of the model's candidates", r"PubMed shows benefit for \ValLitSupModel\% of candidates"),
    (r"Evidence for unlabeled top-20 candidates (Ours, $n$=\ValNovelN) and reference sets ($n$=120).",
     r"Evidence for unlabeled top-20 candidates (Ours $n$=\ValNovelN; others $n$=120)."),
    (r"the strongest is disease $k$NN, which transfers the drugs of the 20 most similar training diseases.",
     r"the strongest is disease $k$NN (drugs of the 20 most similar training diseases)."),
    (r"On the 68 validation diseases the gain over $k$NN is", r"On validation the gain over $k$NN is"),
    (r"Paths carry the signal: on the 68 validation diseases, paths alone reach", r"Paths carry the signal: on validation, paths alone reach"),
    (r"ranks the same corticosteroids first for every target, not disease-specific reasoning.", r"ranks the same corticosteroids first for every target."),
    (r"With paths only, A and A$'$ both stay near the floor", r"With paths only, both stay near the floor"),
    (r"lists one off-label drug, while the top choice, prednisolone, is standard AIH therapy", r"lists one off-label drug; the top choice, prednisolone, is standard therapy"),
    (r"requiring a named intervention matched by whole name (A: phase $\geq$3, B: phase 2, C: phase 1/observational)",
     r"requiring a whole-name match to a named intervention (A: phase $\geq$3, B: 2, C: 1/observational)"),
    (r"Path-based zero-shot transfer works for rare autoimmune diseases the model never sees. With scarce data, related autoimmune diseases are the most useful source; at full scale, the broad pharmacological graph.",
     r"Path-based zero-shot transfer works for rare autoimmune diseases the model never sees: related autoimmune diseases are the best source when data is scarce, the broad pharmacological graph at full scale."),
    (r"Aim 1: MRR on held-out autoimmune diseases ($E$, left) and the targets (right) when training on A$'$, A, D or C$_E$ (95\% CIs).",
     r"Aim 1: MRR on $E$ (left) and the targets (right) by training set (95\% CIs)."),
    (r"MRR~\citep{voorhees1999trec} rewards known drugs at the very top: on the test split, MRR \TestMrr{} corresponds to a median known-drug rank of \MedRankTest, with \RankOneShareTest\% ranked first.",
     r"On the test split, MRR~\citep{voorhees1999trec} is \TestMrr{}; the median known-drug rank is \MedRankTest, and \RankOneShareTest\% are ranked first."),
    (r"PubMed titles were classified with AI assistance; every benefit or harm label was re-checked against the abstract.",
     r"AI-assisted PubMed labels were each re-checked against the abstract."),
    (r"(excluding background therapy, premedication and conditioning)", r"(not as background therapy)"),
    (r"Ustekinumab for Crohn counts as unlabeled only because PrimeKG misses its 2016 approval.",
     r"Ustekinumab--Crohn is a PrimeKG label gap (approved 2016)."),
    (r"on the six targets it is ahead for \WinsMainKnnTargets{} of 6, not significant with $n=6$.",
     r"on the six targets it is ahead for \WinsMainKnnTargets{} of 6 (not significant)."),
    (r"examples are amifampridine for MG (NCT03304054), ciprofloxacin for Crohn (NCT00609973) and N-acetylcysteine for SSc (NCT00428883), all checked by hand.",
     r"e.g.\ amifampridine--MG (NCT03304054), ciprofloxacin--Crohn (NCT00609973), both checked by hand."),
    (r"through its Lambert--Eaton indication, a related neuromuscular-junction disease, and exposes", r"through the related Lambert--Eaton syndrome and exposes"),
    (r"Our MRR values lie within BioPathNet's range on TxGNN's disease areas (about 0.2--0.8)~\citep{hu2026biopathnet}, but splits differ, so we ran",
     r"BioPathNet reports MRR of about 0.2--0.8 on TxGNN's disease areas~\citep{hu2026biopathnet}, but on other splits; we therefore ran"),
    (r"(iii) $E$ has 12 diseases, and 11-disease models are seed-unstable; (iv) no clinician has reviewed the literature labels; (v) corticosteroids and antibiotics dominate the lists.",
     r"(iii) $E$ has 12 diseases, and 11-disease models are seed-unstable; (iv) no clinician has reviewed the literature labels."),
    (r"Model: dimension 32, 6 layers, readout MLP $224\!\to\!64\!\to\!1$ (dropout 0.1), prior MLP $4\!\to\!16\!\to\!1$; 68,226 parameters.",
     r"Model: as in Fig.~\ref{fig:arch} (dimension 32, 6 layers, dropout 0.1)."),
    (r"The loss is the mean negative log-softmax probability of each disease's known drugs over all drugs.",
     r"The loss is the mean negative log-softmax of each disease's known drugs over all drugs."),
]
for old, new in CONDENSE:
    rep(old, new)

# Fig. 2 (transfer curve) and Table 2 (metric definitions) are both narrow: set them side by side in one float, placed
# right after the Aim 1 paragraph that cites Fig. 2 (Table 2 is cited earlier, in Evaluation), so with flafter both
# print after their first mention and Fig. 2 stays next to Aim 1. It is a `table` float, so Table 2 keeps its number
# ahead of Table 3. Springer caption positions: figure caption below, table caption above. Table 4 stays on its own,
# right after the paragraph that cites it.
a = body.index(r"\begin{table}[tb]" + "\n" + r"\centering" + "\n" + r"\caption{Metrics")
b = body.index(r"\end{table}", a) + len(r"\end{table}")
tab2 = body[a:b]
body = body[:a] + body[b:].lstrip("\n")
t2cap = tab2[tab2.index(r"\caption{") + len(r"\caption"):tab2.index(r"\label{tab:metricdefs}")].strip()
t2rows = tab2[tab2.index(r"\midrule") + len(r"\midrule"):tab2.index(r"\bottomrule")].strip()
a = body.index(r"\begin{figure}[tb]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=0.62\textwidth]{fig_transfer_curve_col.pdf}")
b = body.index(r"\end{figure}", a) + len(r"\end{figure}")
fig2 = body[a:b]
fcap = fig2[fig2.index(r"\caption{") + len(r"\caption"):fig2.index(r"\label{fig:transfer}")].strip()
# both halves top-aligned (\vspace{0pt} gives each [t] minipage its top edge as reference), so the shorter table
# does not drop to the bottom and leave a gap above its caption
body = body[:a] + r"""\begin{table}[tb]
\begin{minipage}[t]{0.535\textwidth}
\vspace{0pt}
\centering
\captionof{table}""" + t2cap + r"""
\label{tab:metricdefs}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}p{1.1cm}>{\raggedright\arraybackslash}p{3.8cm}r@{}}
\toprule
Metric & Meaning & Random\\
\midrule
""" + t2rows + r"""
\bottomrule
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.44\textwidth}
\vspace{0pt}
\centering
\includegraphics[width=\linewidth]{fig_transfer_curve_half.pdf}
\captionof{figure}""" + fcap + r"""
\label{fig:transfer}
\end{minipage}
\end{table}""" + body[b:]
# Table 4 on its own, as two blocks side by side (trial evidence | literature and target evidence): same 8 rows and
# numbers as the generated table, half the height
_rows = [r for r in (HERE / "generated" / "tab_validation.tex").read_text(encoding="utf-8").splitlines() if r.endswith("\\\\")]
_rows = [r[:-2].strip() for r in _rows if not r.startswith("Evidence")]
assert len(_rows) == 8 and _rows[0].startswith("Trial") and _rows[4].startswith("PubMed"), _rows
_rows = [r.replace("Target among disease genes", "Drug targets a disease gene") for r in _rows]
_tab4 = "\n".join([r"\begin{tabular}{lccc@{\hspace{14pt}}lccc}", r"\toprule",
                   r"Clinical trials & Ours & Pop. & Rand. & Literature and targets & Ours & Pop. & Rand.\\", r"\midrule"]
                  + [_rows[i] + " & " + _rows[i + 4] + r"\\" for i in range(4)] + [r"\bottomrule", r"\end{tabular}"])
# top of a page only: as a bottom float it left a gap on its page (the next section heading did not fit under it)
rep(r"""\begin{table}[tb]
\centering
\caption{Evidence""", r"""\begin{table}[t]
\centering
\caption{Evidence""")
rep(r"""\label{tab:validation}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\input{generated/tab_validation.tex}""", r"""\label{tab:validation}
\scriptsize
\setlength{\tabcolsep}{3pt}
""" + _tab4)

# Springer headings: title case
for old, new in ((r"\subsection{Aim 1: transfer to rare autoimmune diseases}", r"\subsection{Aim 1: Transfer to Rare Autoimmune Diseases}"),
                 (r"\subsection{Aim 2: accuracy on unseen diseases}", r"\subsection{Aim 2: Accuracy on Unseen Diseases}"),
                 (r"\subsection{Aim 3: what each component contributes}", r"\subsection{Aim 3: Contribution of Each Component}"),
                 (r"\subsection{Case studies and validation}", r"\subsection{Case Studies and Validation}")):
    rep(old, new)

# acknowledgment: run-in heading, not a numbered section
ack = body[body.index(r"\section*{Acknowledgment}"):]
body = body.replace(ack, "\\paragraph{Acknowledgments.} Generative AI tools (Anthropic's Claude) assisted with code, data "
                         "analysis, and drafting and editing of the text; the authors designed the study, ran all "
                         "experiments and verified every result against the run logs.\n\n")

ABSTRACT = r"""Rare autoimmune diseases have few approved drugs, so drug repurposing must rank candidates for diseases whose treatments were never seen in training. This zero-shot setting is studied with \sysname, a BioPathNet-style path-based graph neural network with a channel of four leakage-free prior features, on the PrimeKG knowledge graph. A strict protocol holds out each test disease together with its related diseases and ranks all 7,957 drugs. On 68 unseen test diseases, a known indication appears in the top ten for \TabModelTestAnyTen\% of diseases, against \TabKnnTestAnyTen\% for the best baseline and \RandExpAnyTenTest\% by chance, and the mean reciprocal rank is significantly higher than the baseline's (\TestMrr{} vs.\ \KnnTestMrr). When training data are scarce, data-rich autoimmune diseases transfer better than label-matched non-autoimmune diseases; with the full graph they lower accuracy. Ablations show that path reasoning carries the accuracy, while the prior channel mainly stabilizes training across seeds. Registered clinical trials test \ValTrialTestedModel\% of the new candidates, against \ValTrialTestedPop\% of popular drugs. The output is a ranked list of hypotheses for expert review, not treatment advice."""

HEAD = r"""% SmartCom 2027 (Springer LNNS) version -- GENERATED by make_smartcom.py from bpp_v3_conf.tex; edit that file, then rerun.
\documentclass{svproc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xspace}
\usepackage[table]{xcolor}
% numbered citations in square brackets, sorted and compressed ([3, 5-7]); Springer's LaTeX instructions call
% numbered citation "preferred" and the list "ideally sorted in alphabetical order": spmpsci.bst (MathPhySci) does both
\usepackage[numbers,sort&compress]{natbib}
% svproc defines \chapter, so natbib would start the references with \chapter* on a new page in normal size;
% use svproc's own reference layout instead: an unnumbered section in \small, entries labeled "1."
\renewcommand{\bibsection}{\section*{References}}
\renewcommand{\bibfont}{\small}
\renewcommand{\bibnumfmt}[1]{#1.}
\usepackage{float}
\usepackage{capt-of}
\usepackage{flafter}  % a float never appears before the text that defines it (keeps Table 2 below the Method heading)
\usepackage{xurl}
\def\UrlFont{\rmfamily}
\usepackage[hidelinks]{hyperref}  % clickable cross-references and citations (Springer: \label/\ref become links)
\graphicspath{{figures/}}
\hyphenation{Table Tables Figure Figures ri-san-ki-zu-mab up-a-da-ci-ti-nib gu-sel-ku-mab ra-vu-li-zu-mab
  ro-za-no-li-xi-zu-mab el-a-fi-bra-nor us-te-ki-nu-mab am-i-fam-pri-dine}
\newcommand{\sysname}{BioPathNet++\xspace}
\input{generated/numbers.tex}
\emergencystretch=1.5em
\renewcommand{\bibname}{References}
\providecommand{\refname}{References}\renewcommand{\refname}{References}

\begin{document}
\mainmatter
% the Method section opens with a display right below the architecture figure; LaTeX forbids a page break before
% a display by default, which pushed the Method heading to the next page and left a gap under the figure
\predisplaypenalty=0
\title{BioPathNet++ for Autoimmune Diseases: Path-Based Transfer Learning for Rare Disease Drug Repurposing}
\titlerunning{BioPathNet++ for Rare Autoimmune Disease Drug Repurposing}
\author{Venkata Naga Sai Yatin Tatikonda\textsuperscript{*} \and Srijan Jha \and Charan Kumar Reddy Arikeri \and Praveen Kumar Yadav \and Priyanka~H\textsuperscript{*}}
% Springer: names in full, given names first, surname last; running head = initials + surname
\authorrunning{V. N. S. Y. Tatikonda et al.}
\tocauthor{Venkata Naga Sai Yatin Tatikonda, Srijan Jha, Charan Kumar Reddy Arikeri, Praveen Kumar Yadav, Priyanka H}
\institute{Department of Computer Science and Engineering, PES University, Bengaluru, India\\
\textsuperscript{*}Corresponding authors: \email{yatin.tvns@gmail.com}, \email{drpriyankahsachin@gmail.com}}
\maketitle

\begin{abstract}
""" + ABSTRACT + r"""
\keywords{Drug repurposing $\cdot$ Knowledge graphs $\cdot$ Graph neural networks $\cdot$ Zero-shot learning $\cdot$ Transfer learning $\cdot$ Autoimmune diseases}
\end{abstract}

"""
TAIL = r"""\bibliographystyle{spmpsci}
\bibliography{references}

\end{document}
"""
(HERE / "bpp_v3_smartcom.tex").write_text(HEAD + body + TAIL, encoding="utf-8")
words = len(re.sub(r"\\[A-Za-z]+|[{}$]", " ", ABSTRACT).split())
print(f"wrote bpp_v3_smartcom.tex | abstract ~{words} words")

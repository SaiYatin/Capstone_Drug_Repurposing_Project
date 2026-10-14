# CLAUDE.md — BioPathNet++ (handoff, 15 Oct 2026)

Read this whole file before doing anything. It records what was done, what was verified,
what was wrong in earlier work, and exactly where to continue.

---

## 1. Project in one paragraph

**BioPathNet++**: path-based graph neural network (NBFNet / BioPathNet-style) for **zero-shot
drug repurposing in rare autoimmune diseases**. Final-year capstone + paper project of
**TVNS Yatin** (lead dev), PES University Bengaluru, mentor **Dr. Priyanka H.**
Goal: a 4–5 page conference paper (Indian IEEE venue) and a 10–11 page journal article
(Scopus/SCI). Six rare/sparse targets: Primary Biliary Cholangitis (PBC), Systemic Sclerosis
(SSc), Sjögren syndrome, Autoimmune Hepatitis (AIH), Myasthenia Gravis (MG), Crohn disease.
Data-rich autoimmune diseases: RA, SLE, MS.

**Working rule (user's research-engineer skill):** scientific integrity over impressive
numbers. Never fabricate results/citations. Verify every claim against code and logs.
Keep implemented vs planned clearly separate. Run tests before claiming anything works.

---

## 2. Current state at a glance

| Item | Status |
| --- | --- |
| Audit of old v1/v2 work | DONE — old results invalid (Section 3) |
| New code `bpp/` (v3) on PrimeKG | DONE, tested (4/4 pytest incl. gradcheck) |
| Leakage-free splits + baselines | DONE |
| Kaggle session 1 + 2 | DONE — main ×3 seeds, paths-only ×3, prior-only ×1, no-autoimmune ×1 finished |
| Kaggle session 3 | **NEXT** (user is about to run it): finishes noauto_main_s1/s2, noauto_paths_s0, brg_main_s0, bce_main_s0 |
| Paper rewrite (conference + journal) | NOT STARTED — main numbers are final, start here |
| Old paper `outputs/paper/main.tex`, `main_2page.tex` | CONTAIN FALSE CLAIMS — must be rewritten, not patched |

Planning/audit doc (Claude Docs, user can share it):
https://claude.ai/code/artifact/786aaa16-1565-43e1-8fc0-27f6b3926d6c
("BioPathNet++ — Audit & Revised Research Plan": findings F1–F12, literature check, plan,
session-1/2 results). Keep it in sync if you can; otherwise this file is authoritative.

---

## 3. Why the old (v1/v2) results are invalid — verified, do NOT reuse them

Old code: `models/biopathnet*.py`, `models/nbfnet.py`, `training/kaggle_retrain.py`,
`transfer_learning/predict_drugs_v2_retrained.py`, `validation/metrics_v2.py`,
`validation/ablation_study.py`, DRKG data in `datasets/processed/`. Left untouched on purpose.
Reproduce the audit: `python validation/audit_leakage.py` (CPU, numpy/pandas/scipy/sklearn).

- **F1 target leakage**: `scripts/create_pyg_data.py` built the message graph from the full
  enriched DRKG, so all 277 unique test drug–disease pairs were edges of the graph. Every
  "zero-shot success" (Fenofibrate/Prednisolone/Thalidomide→PBC etc.) was a test edge.
- **F2 trivial negatives** (random drug × random disease): a zero-parameter heuristic
  log(drug degree)+log(disease degree) gets test AUROC **0.972** > model's 0.941.
  Under hard negatives the model was ~chance (AUROC 0.47–0.62 per disease).
- **F3** `k_hop_subgraph` default flow followed incoming edges only and there were no inverse
  edges → SSc, MG, AIH, Crohn, Sjögren have out-degree 0 → zero representation → ranking
  depended on the drug only (why Desmopressin was #1 everywhere).
- **F5** "BioBERT/ChemBERTa" features were BioBERT encodings of ID strings
  ("Compound DB00035"); ChemBERTa never used. The "−0.332 without BioBERT" ablation is invalid.
- **F6** 90% of labels were text-mined GNBR edges; Thalidomide–PBC is a GNBR "treats" edge
  but the only RCT was negative (McCormick 1994, J Hepatol, doi:10.1016/s0168-8278(94)80092-8).
- **F7** Orphanet/HPO enrichment added 0 edges to the 9 autoimmune diseases; expansion was
  53% over a 1.25M-edge DRKG extract, not "33%".
- **F8** ablation script loaded v1 not v2; "single NBF layer"/"no BRG" switches did nothing.
- **F9** SOTA table (TransE 0.65, TxGNN 0.92 …) was hard-coded, never run.
- **F10** "BRG" in v2 was just 1-hop disease-neighbour averaging. **F11** HPO parser created
  2,325 spurious relation types (`HPO::has_phenotype_freq_*`). **F12** duplicates, wrong
  counts, claimed RepoDB negatives that don't exist in code.
- Literature: **BioPathNet already exists** (Hu et al., *Nat Biomed Eng* 2026,
  doi:10.1038/s41551-025-01598-z) with BRG, type-aware/SANS negatives and TxGNN zero-shot
  splits → "hard negatives" and "BRG" are NOT novel. TxGNN: Huang et al., *Nat Med* 2024,
  doi:10.1038/s41591-024-03233-x. Leakage benchmark: Brière et al., *Bioinformatics* 2026.

---

## 4. The new system (v3) — `bpp/`

### Data: PrimeKG (Chandak et al., Sci Data 2023, doi:10.1038/s41597-023-01960-3)
- Raw: `datasets/raw/primekg/kg.csv` (940 MB) and `kg.csv.gz` (94 MB). Source:
  Harvard Dataverse doi:10.7910/DVN/IXA7BM (blocked from the sandbox; user downloaded it).
- TxGNN area files (from github mims-harvard/TxGNN `txgnn/data_splits/`):
  `datasets/raw/txgnn_splits/{HumanDO.obo, mondo_references.csv, kg_grouped_diseases_bert_map.csv}`.
- Compact build (9 MB): `datasets/bpp_primekg/` (meta.json, nodes.npz/json, edges.npz,
  therapeutic.npz, splits.json, splits_noauto.json). Rebuild in ~35 s:
  `python -m bpp.build_primekg --kg datasets/raw/primekg/kg.csv.gz --txgnn-dir datasets/raw/txgnn_splits --out datasets/bpp_primekg`
- Stats: 129,375 nodes, 7,957 drugs, 17,080 diseases, 4,007,433 context edges (27 relation
  types, ×2 with inverses + 3 therapeutic ×2 = 60), 42,631 therapeutic pairs
  (indication 9,388; contraindication 30,675; off-label 2,568).

### Splits (leakage rules live in `bpp/kg.py` + `bpp/splits.py`)
- Therapeutic (drug–disease) edges of any held-out disease are NEVER in the message graph.
- Each training batch also removes all drug edges of its query diseases AND their
  disease–disease neighbours (mirrors zero-shot inference).
- **Strict families**: each target is held out together with all disease–disease neighbours.
  Verified necessary: without it all 16 Crohn indications stayed reachable via "Crohn's colitis";
  MG via "adult-onset myasthenia gravis"; SSc via "scleroderma (disease)".
  Validation diseases are held out the same way.
- `splits.json` (protocol **targets**): 6 target families held out; 1,206 train query
  diseases; 68 zero-shot validation diseases.
- `splits_noauto.json` (protocol **no_autoimmune**): additionally holds out the TxGNN
  autoimmune area (DOID:417 → 70 nodes) + all 9 autoimmune strict families = 123 nodes;
  1,194 train, 66 valid. Also gives the `area` eval set (21 area diseases with indications,
  147 positives). Note: TxGNN's DOID:417 area does NOT contain PBC, SSc or Crohn.
- Target labels = PrimeKG indications ∪ off-label of the target node: PBC 1, SSc 5,
  Sjögren 12, AIH 1, MG 7, Crohn 24 (50 total). Labels are inherently sparse → the 6 targets
  are **case studies**; quantitative claims rest on the 68 valid diseases and 21 area diseases.

### Model (`bpp/model.py`)
- NBFNet core: query = (disease, indication); indicator boundary condition on the source
  disease; DistMult messages; mean aggregation + PNA-style log-degree amplification;
  6 layers, dim 32, shortcut + LayerNorm; scores ALL drugs in one pass.
- `RelSpMM`: custom autograd over per-relation sparse CSR matrices, recomputes A_r·X in
  backward → memory O(V·B·d). Measured: graph 0.32 GB, ~0.62 GB activations per query;
  Kaggle T4 peak 5.76 GB at batch 8; ~8.4–8.9 s/step; 151 steps/epoch.
- "++" transfer channel (`--prior`): 4 leakage-free prior features per (disease, drug):
  popularity, disease-similarity kNN (TF-IDF over disease proteins+phenotypes, k=20),
  shared drug-target/disease-gene count, autoimmune frequency — each computed excluding the
  query's own strict family (`bpp/task.py`). Learned MLP added to the path logit.
- Loss: listwise multi-positive softmax over all drugs (default) or self-adversarial BCE.
- ~68k parameters (not 8M as the old paper said).

### Evaluation (`bpp/metrics.py`)
Full ranking over all 7,957 drugs, filtered ranks. MRR, Hits@1/10/50/100, mean rank, AUROC,
AUPRC, AUROC vs hard negatives (drugs indicated for autoimmune diseases), AUROC vs
degree-matched negatives. Macro mean over diseases with bootstrap 95% CI (resampling
diseases); `paired_bootstrap` for model-vs-baseline p-values.

### Commands
```
python -m pytest -q tests/test_bpp.py
python -m bpp.run_baselines --data datasets/bpp_primekg --split splits.json --out outputs/bpp/baselines_targets.json
python -m bpp.train --data datasets/bpp_primekg --split splits.json --out outputs/bpp/main_s0 --prior   # needs GPU
python -m bpp.build_drkg --tsv datasets/processed/enriched_drkg.tsv --out datasets/bpp_drkg            # secondary/dev dataset
```
Local laptop: RTX 3050 4 GB — too small for full training (needs ~6 GB). Use Kaggle.

---

## 5. Kaggle workflow (T4 ×2, 30 GPU-h/week; user has used ~2 sessions)

- Dataset `biopathnet-pp-v3` = `kaggle/biopathnet-pp-v3.zip` (bpp/ + datasets/bpp_primekg/).
  Mounted at `/kaggle/input/datasets/slizzy/biopathnet-pp-v3/` (Kaggle user `slizzy`).
- Notebook = paste `kaggle/run_kaggle.py` into one cell; run with **Save Version → Save & Run
  All** (interactive runs are not saved). It: runs baselines → smoke test → experiment queue
  on both GPUs → `summary.md`. Stops itself at 11 h (`--time-budget-h`).
- Resume between sessions: Kaggle's input picker does not show the notebook's own output, so
  the user zips `runs/` from the downloaded output and uploads it as a new dataset
  (`kaggle/bpp-runs-session1.zip`, `kaggle/bpp-runs-session2.zip` exist). The script restores
  every `runs/*/last.pt` found under /kaggle/input and skips runs that have results.json.
- Settings used: EPOCHS = 20, patience 6 (in the repo file; the user may have set 5 in the
  notebook copy). Runs usually early-stop at 10–18 epochs.
- Session 3 instructions given to the user: upload `bpp-runs-session2.zip` as dataset
  `bpp-runs-session2`, replace the session-1 input, Save & Run All. Expected ~8–10 h.
- Downloaded results land in `C:\Users\Sai Yatin\Downloads\results (N)\`; copies are kept in
  `outputs/bpp/kaggle_session1/` and `outputs/bpp/kaggle_session2/`. When session 3 output
  arrives: copy json/log files to `outputs/bpp/kaggle_session3/` and re-run the analysis.

---

## 6. Results so far (verified from results.json files)

Baselines (CPU, same split). Targets protocol:

| Method | Valid MRR | Valid H@10 | Targets MRR | Targets H@10 |
| --- | --- | --- | --- | --- |
| random | 0.001 | 0.002 | 0.001 | 0.000 |
| drug degree | 0.004 | 0.003 | 0.003 | 0.000 |
| popularity | 0.046 | 0.049 | 0.099 | 0.148 |
| autoimmune frequency | 0.038 | 0.040 | 0.094 | 0.169 |
| shared targets | 0.057 | 0.105 | 0.008 | 0.000 |
| **disease kNN (best baseline)** | 0.281 | 0.350 | 0.296 | 0.424 |

Models (mean ± sd over seeds; session 1+2):

| Model | Seeds | Valid MRR | Valid H@10 | Targets MRR | Targets H@10 | Targets AUROC-hard |
| --- | --- | --- | --- | --- | --- | --- |
| **BioPathNet++ (paths + prior)** | 3 | **0.394 ± 0.003** | 0.456 ± 0.015 | **0.475 ± 0.014** | 0.648 ± 0.027 | 0.892 ± 0.016 |
| Paths only | 3 | 0.405 ± 0.010 | 0.479 ± 0.021 | 0.463 ± 0.071 | 0.657 ± 0.021 | 0.907 ± 0.006 |
| Prior/transfer only | 1 | 0.303 | 0.353 | 0.322 | 0.483 | 0.782 |
| No-autoimmune training (paths+prior) | 1 (s1,s2 unfinished) | 0.454* | 0.557* | 0.473 | 0.628 | 0.899 |
| No-autoimmune, `area` set (21 diseases) | 1 | — | — | area MRR 0.485 (kNN 0.311, p = 0.005) | area H@10 0.608 | 0.816 |

*different validation set (noauto protocol), not comparable to the 0.394 row.

Significance (paired bootstrap over diseases, seed-averaged): main vs kNN valid +0.113,
p < 0.001; targets +0.179, p = 0.043. Main vs paths-only: valid −0.010 (p = 0.75),
targets +0.011 (p = 0.36) → **tied**.

**Interpretation (honest, already corrected once):**
1. BioPathNet++ beats the best baseline significantly; seeds very stable.
2. The NBFNet path reasoning carries the signal. The transfer channel does NOT improve the
   mean; it reduces seed variance on targets (sd 0.014 vs 0.071). A session-1 claim that the
   transfer channel helps rare targets (from one paths-only seed) was WITHDRAWN.
3. **E2 / autoimmune-specific transfer hypothesis NOT supported**: removing all 123
   autoimmune diseases from training barely changes target MRR (0.473 vs 0.493 for seed 0,
   p = 0.30). Transfer comes from the whole pharmacological KG. Paper must say so
   (pending noauto seeds 1–2 from session 3).
4. Case studies (main_s0, leakage-free): PBC → ursodeoxycholic acid #1; MG → ambenonium,
   neostigmine, pyridostigmine, distigmine #1–4, amifampridine #5; SSc → bosentan #2;
   Sjögren → prednisolone #1, cyclosporine/lifitegrast in top 10; Crohn → corticosteroids,
   mesalazine #6, ciprofloxacin #4 (used in perianal Crohn). AIH MRR ≈ 0.001 is a LABEL GAP:
   PrimeKG lists one off-label drug for AIH while the model's #1 is prednisolone (clinical
   first-line). Corticosteroid dominance across diseases = popularity bias → limitation.
   Per-seed target MRR: PBC 1.0 all seeds; SSc 0.42–0.50; Sjögren 0.40–0.61; MG 0.66–0.68;
   Crohn 0.18–0.25; AIH ~0.001.

---

## 7. What to do next (in order)

1. **When session 3 results arrive**: stage json/logs, copy to `outputs/bpp/kaggle_session3/`,
   recompute the tables above (3-seed noauto E2, noauto_paths_s0, BRG ablation, BCE ablation).
   Update this file + the Claude Doc.
2. **Figures** (matplotlib, saved to `outputs/bpp/figures/`): (a) architecture/pipeline
   diagram of v3 (replace old architecture PNGs — they depict the invalid design);
   (b) MRR/Hits bar chart with CIs: baselines vs paths vs full; (c) per-target top-10 table;
   (d) one path explanation per case study — NOT implemented yet: needs an NBFNet
   path-extraction routine (gradient/beam search over edges, as in NBFNet paper) in a new
   `bpp/explain.py`.
3. **External validation of novel candidates**: for each target's top-20 non-label drugs,
   check ClinicalTrials.gov and PubMed; report as evidence grades, not accuracy.
4. **Conference paper (4–5 pp, IEEEtran)**: new file e.g. `outputs/paper/bpp_v3_conf.tex`.
   Framing: *leakage-controlled path-based zero-shot repurposing for rare autoimmune
   diseases built on BioPathNet*; contributions = (i) leakage audit showing degree heuristics
   match naive pipelines, (ii) strict-family zero-shot protocol on PrimeKG, (iii) BioPathNet++
   with memory-efficient full-graph NBFNet on a T4 + transfer channel, (iv) honest negative
   finding on autoimmune-specific transfer, (v) case studies. Cite BioPathNet as the base
   model; do NOT claim to beat TxGNN (not run on our split).
5. **Journal version (10–11 pp)**: adds the full audit (F1–F12 + degree baselines),
   ablations (BRG, BCE, prior-only), sparsity/degree-stratified analysis, external
   validation, limitations (label sparsity, steroid bias, no head-to-head TxGNN).
6. Optional for stronger journal: run TxGNN's own code and BioPathNet's code on our split;
   clinician review of candidates (via Dr. Priyanka).
7. Venues (verify CFPs/indexing before submitting): IEEE CONECCT, IEEE INDICON, ICCCNT,
   InCoB/ISCB-APAC 2027; journals: Computers in Biology and Medicine, J Biomed Informatics,
   BMC Bioinformatics, IEEE Access, Scientific Reports, J Biosciences.
8. Consider renaming (e.g. a distinct name) to avoid implying an official BioPathNet
   successor — ask the user.

---

## 8. File map (new work)

```
bpp/                 v3 package: kg.py, model.py, task.py, metrics.py, splits.py,
                     build_primekg.py, build_drkg.py, run_baselines.py, train.py, README.md
tests/test_bpp.py    4 tests (RelSpMM gradcheck, batch-edge removal, inverse reachability, learning)
kaggle/run_kaggle.py Kaggle runner (queue, resume, summary)
kaggle/*.zip         dataset zip + runs zips for resuming
validation/audit_leakage.py   reproduces the v1/v2 audit numbers
datasets/bpp_primekg/         compact PrimeKG build + splits
datasets/raw/primekg/         kg.csv(.gz)
datasets/raw/txgnn_splits/    TxGNN DOID/MONDO files
outputs/bpp/                  baselines_*.json, cpu_prioronly_*, kaggle_session1/, kaggle_session2/
```
Not yet committed to git: everything above (repo last commit 0582dbd predates v3). Suggest
the user commits with a clear message; do not delete v1/v2 files or checkpoints.

## 9. Gotchas learned
- Windows paths with spaces (`results (1)`) — quote them.
- Don't use `pkill -f <pattern>` matching your own shell command.
- CPU training of the full model needs >7 GB RAM; even dim-16 CPU runs take ~17–27 s/step.
- PyTorch sparse CSR emits "beta" warnings — harmless.
- Kaggle "Debugger warning: frozen modules" lines — harmless.
- PrimeKG node ids for MONDO drop leading zeros; grouped nodes have `_`-joined ids.

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
| Kaggle session 3 | **DONE** 17 Oct → `outputs/bpp/kaggle_session3/`. All 13 planned runs finished |
| Kaggle session 4 | **DONE** 18 Oct → `outputs/bpp/kaggle_session4/`. Transfer curve complete (7 runs) + test_main_s0. test_main_s1/s2 stopped cleanly at epochs 12/8 (time budget) |
| Kaggle session 5 (optional, ~4 h) | Resume test_main_s1/s2: upload `kaggle/bpp-runs-session4.zip` as dataset, add as input next to biopathnet-pp-v3, Save & Run All (notebook code unchanged) |
| TxGNN head-to-head | **DONE 21 Oct** (Kaggle, results (5).zip) → `outputs/bpp/txgnn_h2h/`; papers updated (journal §"Head-to-head with TxGNN" + abstract sentence, conf discussion); no TxGNN RECHECK left |
| TxGNN head-to-head run 2 (fair coverage, 3 seeds each) | **DONE 21 Oct** (Kaggle, results (6).zip) → `outputs/bpp/txgnn_h2h_full/`, verified; results in "RUN 2 RESULTS" below. Papers NOT updated yet (still run 1); `bpp/analyze.py` not yet extended |
| Paper rewrite (conference + journal) | **FINAL DRAFTS 19 Oct**: title/authors set, 3 aims, metrics table, validation section (V1–V8). Not compiled — upload `outputs/paper/v3_latest.zip` to Overleaf |
| Analysis/figures | `python -m bpp.analyze` → summary.json + `outputs/paper/v3/generated/` (all paper numbers/tables as macros); `python -m bpp.figures` → `outputs/paper/v3/figures/` |
| Reviewer/panel explainer | Claude Doc "BioPathNet++ — Results Explained & Reviewer Q&A": https://claude.ai/code/artifact/5c6b4b6b-abb7-4065-b185-22ecfb0ab894 |
| Old paper `outputs/paper/main.tex`, `main_2page.tex` | CONTAIN FALSE CLAIMS — superseded by v3, do not use |

**Session 3 results (17 Oct, verified; Kaggle code identical to repo, earlier runs byte-identical):**
- E2 with 3 seeds: targets MRR without autoimmune training 0.487±0.031 vs 0.475±0.017 with
  (paired with−without −0.013, CI −0.070..+0.055, Wilcoxon p = 0.81) → hypothesis NOT supported, confirmed.
- Area (21 dis., noauto 3 seeds): MRR 0.449±0.042 vs kNN 0.311; diff +0.138, CI +0.010..+0.280,
  Wilcoxon p = 0.147 → "suggestive" (was +0.174/p 0.085 with 1 seed — weakened; papers say so).
- noauto_paths_s0 area 0.393 vs noauto main 0.449: prior channel MAY help on area (+0.056,
  p = 0.073; seed-0 matched +0.092, p = 0.094) → not significant, stated as needing more seeds.
- BCE ablation: valid MRR 0.347 vs 0.394 → listwise better (+0.047, CI +0.004..+0.093, p = 0.005
  vs 3-seed mean; same-seed +0.048, CI −0.001..+0.100, p = 0.023 → "consistent, borderline").
- BRG ablation (drop anatomy_protein_* + drug_drug): 4,007,433 → 1,133,029 edges (−72%),
  valid MRR 0.397 (tie, p = 0.91), 7.49 vs 8.46 s/step (−11%), 5.52 vs 5.76 GB.
- noauto_main_s1 hit the 20-epoch cap (best epoch 19), so it was still improving slightly.
If more runs are ever added: copy to `outputs/bpp/kaggle_sessionN/runs/`, run
`python -m bpp.analyze && python -m bpp.figures`, and re-read sentences that interpret numbers.

**Validation + paper finalisation (19 Oct; user's pasted plan V1–V8; outputs/bpp/validation/):**
- Decisions applied: title "BioPathNet++ for Autoimmune Diseases: Path-Based Transfer Learning for Rare Disease
  Drug Repurposing" (journal adds "---A Leakage-Controlled Study"); authors TVNS Yatin, Srijan Jha, A Charan KR,
  Praveen Kumar Yadav; Dr. Priyanka H. = guide (author-block note / \thanks + acknowledgment), NOT co-author.
  Claim wording: "outperforms every baseline evaluated on the same leakage-free split"; tool for expert review.
- New code: `bpp/score.py` (full 7,957-drug scores; laptop GPU reproduces Kaggle top-10 for 17/18 lists, 1 exact tie
  swap), `bpp/validate.py` (V1–V5, V7, V8, E check; API responses cached in validation/cache/), `bpp/explain.py`
  (grad×input edge attribution + beam search; replica asserted equal to model), tests now 7/7.
- Corrections to the pasted plan: random any-hit@10 = 0.8% (not 1.5%); "MRR 0.49 ≈ first true drug at rank 2" is
  wrong → papers use median known-drug rank (test 3, 40% at rank 1).
- Metrics table (test, 1 seed): Any@10 75% vs kNN 57% vs pop 19% vs random 0.8%; AUPRC 0.554 vs prevalence ~0.001.
- V2 trials (ClinicalTrials.gov API v2, drug must be a NAMED intervention): unlabelled model top-20 (n=86) grade A–C
  19% vs popularity 15% vs random 1%; 13 grade A. Caveats in papers: PBC/AIH grades partly from overlap-syndrome
  trials; ustekinumab–Crohn is a PrimeKG label gap (FDA 2016-09-26, J&J release).
- V3 PubMed: any co-mention 63% (pop 72%, random 18%); titles classified (AI-assisted, 394 titles,
  literature_labels.csv, MUST be verified by authors): supports 24% (pop 16%, random 0%); against 4% (pop 12%).
  Model "against" = corticosteroids in SSc (renal crisis) + 1 AIH case; popularity "against" = antibiotics.
- V4 temporal (dates verified via manufacturer/FDA pages, temporal_sources.json): post-2022 approvals in PrimeKG:
  risankizumab 37, upadacitinib 550, ravulizumab 845, guselkumab 1,363, rozanolixizumab 2,278, elafibranor 5,825
  → median 1,104 (top 14%), 5/6 better than random median, sign test p = 0.11 (NOT significant); kNN better on 3/6.
  Absent from PrimeKG: seladelpar, efgartigimod, zilucoplan, nipocalimab, mirikizumab. Label gaps: tocilizumab-SSc
  rank 58, nintedanib 848, obeticholic acid 2,272.
- V5: ≥1 drug target among disease genes 17% (pop 16%, random 3%); immune-pathway enrichment null for all sets.
- V6 paths (main_s0): MG cholinesterase inhibitors via MUSK → NMJ disease; amifampridine via Lambert–Eaton;
  UDCA via biliary diseases; spurious ciprofloxacin–Crohn via "mouth disease → sinusitis".
- V7: corticosteroids 41%, antibacterials 21% of 120 top-20 slots (name heuristic; DRKG ATC covers only 50% and
  lists steroids under D07/S01/R01, not H02). Popularity-adjusted list (exploratory): amifampridine, guanidine (MG),
  lifitegrast (Sjögren), tildrakizumab (Crohn).
- E check (C_E model, 12 area diseases): 9/12 have a labelled drug in top-10; 21/74 unlabelled have trials.
- V8 `expert_review_sheet.xlsx` ready (empty rating column).
- NOT done / could not verify: no LaTeX compiler (tectonic/MiKTeX need a download → not installed; Overleaf zip
  `outputs/paper/v3_latest.zip` instead) → page counts unmeasured (conf ≈5 pp, journal may exceed 11 pp by word
  count); test split still 1 seed (session 5 optional); literature labels not human-verified.

**TxGNN head-to-head RESULTS (21 Oct, verified; 1 seed each; Kaggle h2h.json reproduced locally byte-for-byte in summaries):**
- Run: DGL OK, TxGNN full pretrain epoch 1.27 h (cap NOT hit), finetune 500 ep 0.14 h (val AUROC still rising at
  ep 480); ours early-stopped (best ep 13, valid MRR 0.186). Test = 13 area diseases with indications, 75 pairs;
  TxGNN candidates 1,798 drugs. TxGNN official eval Recall@100 0.474 = our re-implementation (published 0.338,
  different split draw).
- Coverage handicap (coverage.json): split has 1,299 non-test diseases with indications; ours trained on 637 (+34
  valid) because build_txgnn holds out every disease with ANY TxGNN test edge (+families). Stated in papers.
- TxGNN protocol: R@100 0.475 vs 0.474 (tie, p 0.83); AUROC 0.746 vs 0.909 (TxGNN better, −0.163, CI −0.245..−0.080,
  p 0.003, 10/13); AUPRC 0.101 vs 0.147 (n.s., p 0.46).
- Full ranking (all 7,957): H@100 0.398 vs 0.155 (+0.243, CI +0.089..+0.414, p 0.031 → significant); AUROC 0.910
  vs 0.736 (+0.174, p 0.002 → significant); MRR 0.061 vs 0.026 (11/13 wins, p 0.033 but CI −0.034..+0.105 → NOT
  significant by our rule).
- Why: only 16% of TxGNN's full top-100 are drugs with any indication (ours 99%); TxGNN's protocol filters them out.
  Paper wording: "neither model dominates; first data point (1 seed, 13 diseases)".
- Follow-up PREPARED 21 Oct (second, fairer run; see next block). Paper numbers stay as above until it returns.

**TxGNN head-to-head RUN 2 RESULTS (21 Oct, verified; `outputs/bpp/txgnn_h2h_full/`; 3 seeds each, 13 diseases):**
- Checks: Kaggle code = commit 6ac644c (only CRLF differs); TxGNN split md5 + 75 test pairs + 13 eval diseases + 1,798
  candidates IDENTICAL to run 1 → only our coverage and the seed count changed. Re-ran align + txgnn_eval locally from
  the raw pkl / npz → h2h.json paired/seeds blocks identical. On the downloaded KG: 0 test pairs (eval + 3,248 dropped)
  visible to train/valid diseases or in the message graph; eval diseases not trained; area families (80) held out.
  TxGNN official summary = our protocol re-implementation (R@100/AUROC/AUPRC, 4 d.p.) for all 3 seeds.
  No PRETRAIN CAP (1.23–1.24 h), finetune 0.14 h; ours all early-stopped (best epochs 12, 4, 3), peak 5.70 GB.
- Coverage: ours 1,139 train + 63 valid diseases (run 1: 637 + 34); 411 held out, of which 127 have train
  indications (remaining handicap vs TxGNN); 851 diseases had 3,248 random test edges removed from our table.
- Seed means ± sd, TxGNN protocol: ours R@100 0.614±0.042, AUROC 0.777±0.010, AUPRC 0.343±0.008; TxGNN R@100
  0.536±0.068, AUROC 0.901±0.004, AUPRC 0.161±0.041. Full ranking: ours MRR 0.273±0.009, H@10 0.456, H@100 0.618,
  AUROC 0.936; TxGNN MRR 0.048±0.049 (s1 0.104, others 0.02), H@10 0.091, H@100 0.232, AUROC 0.753.
- Paired (per-disease seed means; significant = CI excludes 0 AND Wilcoxon p < 0.05):
  TxGNN protocol: R@100 +0.078 (CI −0.090..+0.246, p 0.39) n.s.; AUROC −0.124 (CI −0.259..−0.010, p 0.17, 6/13 wins)
  → TxGNN higher but NOT significant by our rule (run 1: −0.163, p 0.003 significant); AUPRC +0.182 (CI −0.003..+0.395,
  p 0.17, 8/13) n.s. Full ranking: MRR +0.226 (CI +0.055..+0.422, p 0.040, 10/13) SIGNIFICANT; H@10 +0.365 (p 0.027)
  SIGNIFICANT; H@100 +0.386 (p 0.002) SIGNIFICANT; AUROC +0.183 (p 0.0002, 13/13) SIGNIFICANT.
- Reading: with matched coverage our model clearly wins full ranking and no longer loses significantly on TxGNN's
  own protocol; TxGNN still has the higher mean AUROC over its filtered candidates. Still only 13 diseases, 1 split.
  Our valid MRR (0.30–0.33) is a different valid set from run 1 (0.186) → not comparable.
- TODO before paper text: extend `bpp/analyze.py::load_txgnn_h2h` for run 2 (seed means, `txgnn_h2h_full`, coverage),
  regenerate macros/table, then rewrite the journal §"Head-to-head with TxGNN", abstract sentence and conf discussion
  (run 1 becomes the coverage ablation). Full download incl. checkpoints + converted KG: `kaggle/txgnn-h2h-run2-results.zip`.

**TxGNN head-to-head RUN 2 — preparation (21 Oct; code + notebook; tests 9/9):**
- `bpp.build_txgnn --coverage {strict,full}` (default strict = run 1, unchanged except new keys `coverage` in
  splits.json and `coverage/stats/dropped_test_pairs` in txgnn_protocol.json). TxGNN's test set = all drug edges of
  the area diseases + 5% random edges from the area's 2-hop neighbourhood (txgnn/data_splits/datasplit.py
  get_edge_group), which is why run 1 held out 873 diseases. `full`: area = diseases whose `rev_` therapeutic rows
  survive process_disease_area_split (TxGNN's area list) → held out WITH their disease-disease families (run 1 held
  out the area without family expansion, but most neighbours were held anyway via random test edges); every other
  disease trains normally and its random test edges are REMOVED from our therapeutic table (= message graph + labels)
  and only listed in txgnn_protocol.json `dropped_test_pairs`. New assert: no test pair of any disease is in the
  table of a train/valid disease (mutation-checked: fires when the removal is disabled). Stats include
  `held_out_with_train_indications` (= remaining coverage handicap vs TxGNN, report it).
- Mini real-ID rehearsal (21 Oct, earlier mini export): strict 64 train dis. / 48 held-out with indications →
  full 100 / 12; 191 random test edges dropped; eval set, candidates, test pairs identical in both modes; tiny
  ours (2 seeds) + align + eval ran. Full-size numbers unknown until Kaggle (expect ~1,260 train diseases).
- `bpp.txgnn_eval`: several npz per model = seeds; `paired`/`paired_full` now use per-disease means over seeds
  (verified: reproduces run-1 h2h.json paired blocks exactly with 1 seed); new `seeds` (mean ± sd over seeds),
  `n_seeds`.
- `kaggle/run_txgnn.py`: COVERAGE = "full", TX_SPLIT_SEED = 1 (one cached TxGNN split for all runs), TX_SEEDS =
  [0,1,2] (training seed set after txgnn's import-time torch.manual_seed(0); run 1 ≈ training seed 0), OUR_SEEDS =
  [0,1,2]. Job queue over both GPUs (order ours0, tx0, tx1, tx2, ours1, ours2; never two TxGNN at once — CPU RAM);
  each job scored right after training; jobs skipped if they no longer fit (TX_RUN_H 1.7 h, OUR_MIN_H 1 h); ours
  stops on the time budget and is scored from best.pt (settings.json `ours_finished` / `jobs` say which). Plan:
  ours ≤ ~4.4 h/seed (≈2× run 1's 6 min/epoch, 20 epochs), TxGNN ~1.6 h/seed → ~9.2 h + setup within SESSION_H 11.
  Per-seed outputs: results/ours_s{N}.npz, ours_s{N}_train_log.jsonl, txgnn_s{N}/ (raw pkl + official summary),
  txgnn_s{N}.npz, logs/ours_s{N}.log, logs/txgnn_s{N}.log. Verified by a stubbed dry run of main() (queue, GPUs,
  no TxGNN overlap, eval gets all 6 npz); TxGNN's training-seed lines were only compile-checked (no DGL env here).
- KAGGLE STEPS (user): (1) Datasets → biopathnet-pp-v3 → New Version → upload the rebuilt `kaggle/biopathnet-pp-v3.zip`
  (21 Oct; run-1 zip kept as `kaggle/biopathnet-pp-v3.txgnn-run1.zip`). (2) Open the TxGNN notebook, replace the
  cell with the new `kaggle/run_txgnn.py`, check the input uses the new dataset version. (3) Settings: GPU T4 ×2,
  Internet ON. (4) Save Version → Save & Run All (~10–11 h). (5) Download output; copy `txgnn_h2h/results/*` + logs to
  `outputs/bpp/txgnn_h2h_full/` (keep run 1 in `outputs/bpp/txgnn_h2h/`).
- AFTER results: `bpp/analyze.py::load_txgnn_h2h` still reads only run 1's layout (first run per model, files
  `ours_s0.npz` / `txgnn_s1.npz`) → extend it for run 2 (seed means, coverage stats) before touching paper text.

**TxGNN head-to-head design (20 Oct; `bpp/build_txgnn.py`, `bpp/txgnn_eval.py`, `kaggle/run_txgnn.py`, test 8/8):**
- TxGNN's own code (git commit 0e0203c, not pip) generates its `autoimmune` disease-area split; its random test edges
  use unseeded np.random → we seed (SPLIT_RANDOM_SEED 0, TX_SEED 1) → different draw from the published runs (say so).
  Its autoimmune area (data/disease_files/autoimmune.csv) INCLUDES Sjögren.
- `bpp.build_txgnn` converts TxGNN's train.csv (forward rows, homogeneous deduped) into a CompactKG → both models see
  the same training graph; test = area diseases' rev_indication rows of test_processed; TxGNN's valid edges dropped;
  ours carves 5% training diseases (+families) for early stopping. Asserts no test pair/disease leaks.
- Ids: TxGNN `convert2str` (numeric → float string); node names "<name> [<type>:<id>]" align predictions.
- TxGNN trained with reproduce/train.py settings (proto, 3 protos, all_nodes_profile, rarity; pretrain 1 ep lr 1e-3
  bs 1024 — capped at 3 h, logged "### PRETRAIN CAP"; finetune 500 ep lr 5e-4) on GPU0; ours (`--prior`) on GPU1.
- Compat shims: pandas DataFrame.append; DGL EdgeDataLoader → DataLoader+as_edge_prediction_sampler.
- 20 Oct first Kaggle attempt FAILED: Kaggle image is now Python 3.13 + torch 2.11/cu128; DGL's last wheels (2.4.0)
  stop at cp312 and torch 2.4.0 has no cp313 → no pip route. Also the git clone failed silently. Fix: TxGNN source
  from the GitHub tarball of the pinned commit (checked locally), and TxGNN runs in its own uv-made Python 3.11 env
  in /tmp/txgnn_h2h/txenv (torch 2.4.0+cu124, dgl 2.4.0+cu124, numpy<2, pandas<2.2) with a clean PYTHONPATH; our
  model stays on Kaggle's Python. Big files (TxGNN data, export CSVs, env) live in /tmp, not the notebook output.
  If DGL still fails: a dgl stub still lets TxGNN generate the split (tested locally), only ours runs.
- Later Kaggle failures, fixed: Jupyter's MPLBACKEND leaked into subprocesses (→ MPLBACKEND=Agg); a `#` comment
  inside the uv package list dropped scikit-learn/matplotlib/seaborn/tqdm/requests (→ fixed; all third-party
  imports of txgnn/ checked statically).
- LOCAL REHEARSAL PASSED (20 Oct, laptop CPU, scratch env Python 3.11 + torch 2.3.0 CPU + DGL 2.2.1 + pandas 1.5.3):
  mini real-ID PrimeKG (481k rows, all 55 autoimmune-area diseases) → the notebook's own split_export.py, TxGNN
  pretrain (cap path hit) + finetune + eval_disease_centric, bpp.build_txgnn, bpp.train/score, align.py,
  bpp.txgnn_eval all ran. On TxGNN's predictions our protocol re-implementation EQUALS TxGNN's official summary
  (AUROC, AUPRC, Recall@10/50/100, MRR@100 to 4 d.p.) → can be stated in the paper. Untested: full-graph GPU
  memory/time of TxGNN on a T4, DGL 2.4 vs 2.2 differences.
- Published (TxGNN reproduce/result_more_metrics.csv, autoimmune, indication, 5 seeds): Recall@100 0.338 ± 0.004.
  Its AUROC/AUPRC (0.875) use balanced sampled negatives → NOT comparable; papers quote only Recall@100.
- Eval: `bpp.txgnn_eval` = TxGNN disease-centric protocol (candidates = drugs with ≥1 indication in kg_directed;
  per-disease AUROC/AUPRC/Recall@k/MRR@k) + our full ranking; paired tests (Recall@100, AUROC, AUPRC).
- Paper macros `Txh*` (pending until results), table `generated/tab_txgnn.tex`.

**Session 4 RESULTS (18 Oct, verified; Kaggle code identical to repo; papers updated, no RECHECK left):**
- Aim 1, scarce data: model MRR on E — A (RA/SLE/MS, 3 seeds) 0.290 vs A′ (3 matched draws) 0.126;
  diff +0.164, CI +0.037..+0.316, Wilcoxon p = 0.016, 9/12 better → SIGNIFICANT. Targets +0.083, p = 0.062.
  Same with FINAL checkpoints (`bpp/evaluate_checkpoint.py`, laptop GPU, `kaggle_session4/last_ckpt_eval/`):
  0.223 vs 0.077, +0.146, CI +0.046..+0.263, p = 0.021. kNN shows same pattern (0.258 vs 0.086).
- CAVEAT (in papers): the A model ranks the SAME corticosteroids first for every target → the scarce-data
  transfer is a drug-class prior, not disease-specific reasoning; model ≈ kNN in this regime (0.290 vs 0.258).
  A/A′ early-stop on general valid (all MRR < 0.05) → tr_A_s1 best epoch 0; tr_A_s2 falls 0.29→0.09 by last epoch.
- Aim 1, full scale: D (non-autoimmune) 0.532 vs C_E (all, 1 seed) 0.452 on E; diff −0.081, p = 0.23 → no gain.
  Answer to Aim 1: transfer from autoimmune diseases matters when data is scarce; at scale the broad graph supplies it.
- Test split (1 seed): MRR 0.494 vs kNN 0.392, +0.102, CI +0.032..+0.179, p = 0.016 → significant; 75% vs 57% any-hit@10.
- `kaggle/bpp-runs-session4.zip` = all session-4 run dirs incl. checkpoints (for resuming s1/s2).

**Session 4 design (17 Oct; `bpp/extra_splits.py`, tests in tests/test_bpp.py, 5/5 pass):**
- Transfer test set E = 12 TxGNN-area diseases with indications that are NOT targets and NOT in
  RA/SLE/MS families; E + targets + valid held out (families) in every condition.
- A = RA/SLE/MS families (11 dis., 161 ind.); A′ = 3 draws of 11 non-autoimmune diseases matched
  disease-by-disease on #indications (160 each); C_E = all 1,185 except E; D = existing noauto runs.
- "B: all autoimmune" is impossible: beyond RA/SLE/MS, PrimeKG's autoimmune diseases ARE the area.
- Test split: 68 random NON-autoimmune diseases (seed 1) + families held out; 1,065 train.
- Graph only contains therapeutic edges of train diseases → each condition also limits drug knowledge.
- Baselines already computed locally (outputs/bpp/baselines_tr_*.json, baselines_test.json):
  kNN on E: A pool 0.258 vs controls 0.086 (better on 12/12, p<0.001) → autoimmune data helps
  the baseline when data is scarce. kNN test-split MRR 0.392 (test diseases easier than valid).
- Runner fix: per-run `--time-budget-h` is now the time LEFT in the session (was 11 h per run →
  a late-starting run could outlive Kaggle's 12 h limit and lose the whole session's output).
- After results: copy to `outputs/bpp/kaggle_session4/runs/`, run analyze + figures, then fix every
  `RECHECK` comment (abstracts, Aim 1 paragraphs, test-split paragraph, journal conclusion).
- Brière et al. 2026 VERIFIED (user opened it): Bioinformatics 42(8) btag608, doi:10.1093/bioinformatics/btag608;
  cited for redundancy/split-realism; NOTE they found degree NOT a driver in KGE models (cited accordingly).

**Statistics correction (15 Oct):** the "targets p = 0.043" quoted earlier is a ONE-sided
paired bootstrap. Two-sided Wilcoxon gives p = 0.188 (n = 6) and the diff CI crosses 0 →
targets gain is NOT significant; papers say so. Rule used in papers: significant only if
bootstrap CI excludes 0 AND Wilcoxon p < 0.05. Area (noauto vs kNN): CI excludes 0 but
Wilcoxon p = 0.085 → "suggestive". Valid (68): significant under both.

**Facts checked for the papers (don't re-derive):** amifampridine is approved for LEMS, not MG,
but has a completed phase-3 MG RCT (NCT03304054); no lifitegrast–Sjögren or limaprost–SSc trial;
ciprofloxacin–Crohn phase-2 RCTs NCT00609973, NCT01783106; prednisone comparator in AIH
NCT00838214. BioPathNet reports indication MRR ≈0.2–0.8 by TxGNN area, Recall@20 0.619 vs
TxGNN 0.539 (cell proliferation); TxGNN AUPRC uses 1:1 sampled negatives. "Brière et al.,
Bioinformatics 2026" was missing from OpenAlex on 15 Oct but IS real (verified 17 Oct, now cited). Commits: user only, no Claude
co-author trailer (user request).

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
| No-autoimmune training (paths+prior) | 3 | 0.445* | 0.554* | 0.487 ± 0.031 | 0.615 ± 0.022 | — |
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
Git: all v3 work is committed on branch **`bpp-v3`** (pushed to origin; commits authored as
Sai Yatin <sumakkiyatin@gmail.com>, which GitHub maps to @SaiYatin). `main` is deliberately
untouched until the v3 paper exists; then open a PR bpp-v3 → main. Not committed on purpose:
`kaggle/*.zip` (gitignored; runs zips hold the best.pt checkpoints, keep them locally),
`checkpoints/best_model_v2.pt` (66 MB, invalid), `Tatikonda_Joshika_CV.tex` + `resume.cls`
(unrelated personal CV). Do not delete v1/v2 files or checkpoints.

## 8b. Evaluation caveat found 15 Oct 2026 (must go in the paper)
The 68 `valid` diseases are ALSO the early-stopping/model-selection set, so "Valid MRR" is
optimistic. Measured from train_log.jsonl (best-epoch MRR minus last-epoch MRR):
main 0.002–0.016, paths-only 0.022–0.031, noauto_main 0.002–0.044 (s1,s2 unfinished). The margin over kNN
(+0.11) is far larger, so the main claim survives, but the paths-only "0.405 > 0.394" is
within selection noise. Untouched by selection: `targets` (6) and noauto `area` (21) — treat
these as the true test sets. Hyperparameters were fixed defaults (not tuned on valid);
say so. A clean fix for the journal: carve a separate test split and retrain (~16 GPU-h).

## 9. Gotchas learned
- Windows paths with spaces (`results (1)`) — quote them.
- Don't use `pkill -f <pattern>` matching your own shell command.
- CPU training of the full model needs >7 GB RAM; even dim-16 CPU runs take ~17–27 s/step.
- PyTorch sparse CSR emits "beta" warnings — harmless.
- Kaggle "Debugger warning: frozen modules" lines — harmless.
- PrimeKG node ids for MONDO drop leading zeros; grouped nodes have `_`-joined ids.

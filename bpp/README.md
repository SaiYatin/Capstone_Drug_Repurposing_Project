# BioPathNet++ v3

Path-based (NBFNet / BioPathNet-style) zero-shot drug repurposing for rare autoimmune diseases,
rebuilt after the October 2026 audit (`validation/audit_leakage.py`). v1/v2 code is untouched.

## What changed vs v2

| v2 problem | v3 fix |
| --- | --- |
| Test drug–disease edges were inside the message graph | Held-out diseases' therapeutic edges never enter the graph; each training batch also loses its own diseases' drug edges (`kg.MessageGraph`) |
| Near-duplicate diseases leaked labels (e.g. "Crohn's colitis") | Targets and validation diseases are held out with all their disease–disease neighbours (strict family) |
| Incoming-only 2-hop subgraph made 5/6 targets invisible | Full-graph propagation with explicit inverse relations |
| Not NBFNet (features everywhere, query added to all nodes) | Indicator boundary condition, query from the disease side, one pass scores every drug |
| Random drug × random disease negatives | Every non-indicated drug is a negative (listwise softmax over all 7,957 drugs) |
| ID-string "BioBERT" features | No node features (NBFNet is feature-free); transfer signal comes from leakage-free priors |
| Hand-weighted hybrid with an unnormalised logit | Learned transfer channel (popularity, disease-similarity kNN, shared targets, autoimmune frequency), each computed excluding the query's own family |
| 90% text-mined labels | PrimeKG curated indications / off-label use |
| Hard-coded SOTA numbers | Baselines run on the identical split (`run_baselines.py`) |

## Data

```
datasets/raw/primekg/kg.csv(.gz)    PrimeKG, Harvard Dataverse doi:10.7910/DVN/IXA7BM
datasets/raw/txgnn_splits/          TxGNN disease-area files (DOID:417 autoimmune area)
datasets/bpp_primekg/               compact build (9 MB) -- this is what Kaggle needs
```

Rebuild (35 s): `python -m bpp.build_primekg --kg datasets/raw/primekg/kg.csv.gz --txgnn-dir datasets/raw/txgnn_splits --out datasets/bpp_primekg`

Protocols:
- `splits.json` (**targets**): the 6 target families held out; 1,206 training diseases; 68 zero-shot validation diseases.
- `splits_noauto.json` (**no_autoimmune**): additionally holds out the TxGNN autoimmune area + all 9 autoimmune families (123 diseases) -> E2 transfer test and an `area` evaluation set.

Target labels = PrimeKG indications + off-label use of the target node (PBC 1, SSc 5, Sjögren 12, AIH 1, MG 7, Crohn 24 drugs).

## Commands

```
python -m pytest -q tests/test_bpp.py                                  # 4 tests incl. gradcheck
python -m bpp.run_baselines --data datasets/bpp_primekg --split splits.json --out outputs/bpp/baselines_targets.json
python -m bpp.train --data datasets/bpp_primekg --split splits.json --out outputs/bpp/main_s0 --prior
```

Kaggle: see the header of `kaggle/run_kaggle.py` (dataset = `bpp/` + `datasets/bpp_primekg/`).

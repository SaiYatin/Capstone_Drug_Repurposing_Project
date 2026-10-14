# v3 papers (IEEE format)

| File | What |
| --- | --- |
| `bpp_v3_conf.tex` | Conference paper, `IEEEtran` conference mode, target 4-5 pages |
| `bpp_v3_journal.tex` | Journal article, `IEEEtran` journal mode, target 10-11 pages |
| `common.tex` | Shared packages, `\sysname` (rename the system here), number macros |
| `references.bib` | Bibliography; every entry checked against OpenAlex/PubMed on 2026-10-15 |
| `generated/` | **Do not edit.** `numbers.tex` + `tab_*.tex`, written by `python -m bpp.analyze` |
| `figures/` | **Do not edit.** PDFs/PNGs written by `python -m bpp.figures` |

## Rebuild after new Kaggle results

```
python -m bpp.analyze      # reads outputs/bpp/kaggle_session*/runs/*/results.json
python -m bpp.figures
```
Then recompile. Search the .tex files for `RECHECK` / `TODO`: numbers update automatically,
but sentences that interpret them (e.g. the no-autoimmune paragraph) must be re-read.

## Compile

No LaTeX is installed on the dev laptop. Either upload this folder to Overleaf
(set the main document, compiler pdfLaTeX), or install TeX Live/MiKTeX and run:
```
pdflatex bpp_v3_conf && bibtex bpp_v3_conf && pdflatex bpp_v3_conf && pdflatex bpp_v3_conf
```
Length was estimated from word counts, not measured: check the page count after compiling.

## Before submission
- Confirm author list/order and affiliations (`TODO(authors)`).
- Edit the AI-use disclosure in the Acknowledgment to describe actual use (IEEE requires it).
- Check the venue's template/page limit; conference CFPs differ.

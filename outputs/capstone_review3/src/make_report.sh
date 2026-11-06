#!/usr/bin/env bash
# Two-pass build of the capstone report: build -> Word merge -> render -> measure pages -> rebuild with pages.
set -e
S="$(cd "$(dirname "$0")" && pwd)"
cd "$S"
TPL="tpl/zip/Report initial Pages(Pg 1-10)"
W() { cygpath -w "$S/$1"; }
build() {
  python build_report.py "$TPL" assets out/body.docx $1 | tail -1
  powershell -NoProfile -ExecutionPolicy Bypass -File merge_report.ps1 "$(W out/body_front.docx)" "$(W out/body.docx)" "$(W out/report.docx)" | tail -1
  powershell -NoProfile -ExecutionPolicy Bypass -File render.ps1 "$(W out/report.docx)" "$(W out/report.pdf)" | tail -1
}
rm -f out/pages.json
build
python find_pages.py out/report.pdf out/pages.json
cp out/pages.json out/pages_pass1.json
build out/pages.json
python find_pages.py out/report.pdf out/pages_pass2.json
python -c "
import json; a=json.load(open('out/pages_pass1.json')); b=json.load(open('out/pages_pass2.json'))
d={k:(a[k],b.get(k)) for k in a if a[k]!=b.get(k)}; print('page numbers stable' if not d else ('CHANGED: %s' % d))"

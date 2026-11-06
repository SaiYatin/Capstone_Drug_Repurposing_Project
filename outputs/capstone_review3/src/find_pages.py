"""Measure printed page numbers of chapters, sections, figures and tables from the rendered report PDF.
Usage: find_pages.py <report.pdf> <out pages.json>"""
import json
import re
import sys

import fitz

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
import report_content as C  # noqa: E402

doc = fitz.open(sys.argv[1])
pages = []                                   # (printed number, lines)
for pg in doc:
    t = pg.get_text()
    m = re.search(r"Dept\. of CSE\s.*?Page (\d+)", t, re.S)       # footer text may come first in the extraction
    if m:
        pages.append((int(m.group(1)), [l.strip() for l in t.splitlines()]))


def find(pred, start=0):
    for i in range(start, len(pages)):
        if any(pred(l) for l in pages[i][1]):
            return i
    return None


out, cursor = {}, 0
for ci, (title, blocks) in enumerate(C.CHAPTERS, 1):
    i = find(lambda l, ci=ci: l == f"CHAPTER - {ci}", cursor)
    out[f"ch{ci}"] = pages[i][0]
    cursor = i
    si = 0
    for b in blocks:
        if b[0] == "h2":
            si += 1
            head = f"{ci}.{si} {b[1]}"
            j = find(lambda l, h=head: l.startswith(h[:40]), cursor)
            out[f"sec{ci}.{si}"] = pages[j][0]
            cursor = j
for key, needle in (("refs", "REFERENCES"), ("appA", "APPENDIX A: DEFINITIONS, ACRONYMS AND"), ("appB", "APPENDIX B: USER MANUAL")):
    i = find(lambda l, n=needle: l.startswith(n), 0)
    out[key] = pages[i][0]
for i, (num, lines) in enumerate(pages):
    for l in lines:
        m = re.match(r"(Figure|Table) (\d+\.\d+):", l)
        if m and f"{m.group(1)} {m.group(2)}" not in out:
            out[f"{m.group(1)} {m.group(2)}"] = num
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(len(out), "entries;", {k: out[k] for k in list(out)[:6]}, "| body pages", len(pages))

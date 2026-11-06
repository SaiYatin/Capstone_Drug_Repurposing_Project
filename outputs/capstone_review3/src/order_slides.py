"""Rewrite <p:sldIdLst> in the given file order. Usage: order_slides.py <unpacked dir> slide1.xml slide2.xml ..."""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
order = sys.argv[2:]
rels = (root / "ppt/_rels/presentation.xml.rels").read_text(encoding="utf-8")
rid_of = {}
for m in re.finditer(r'<Relationship [^>]*?Id="(rId\d+)"[^>]*?Target="slides/(slide\d+\.xml)"', rels):
    rid_of[m.group(2)] = m.group(1)
for m in re.finditer(r'<Relationship [^>]*?Target="slides/(slide\d+\.xml)"[^>]*?Id="(rId\d+)"', rels):
    rid_of[m.group(1)] = m.group(2)
pres_p = root / "ppt/presentation.xml"
pres = pres_p.read_text(encoding="utf-8")
lst = re.search(r"<p:sldIdLst>(.*?)</p:sldIdLst>", pres, re.S)
items = {re.search(r'r:id="(rId\d+)"', e).group(1): e for e in re.findall(r"<p:sldId [^>]*/>", lst.group(1))}
new = "".join(items[rid_of[f]] for f in order)
pres = pres[:lst.start(1)] + new + pres[lst.end(1):]
pres_p.write_text(pres, encoding="utf-8")
print(f"{len(order)} slides in order; dropped:", sorted(set(rid_of) - set(order)))

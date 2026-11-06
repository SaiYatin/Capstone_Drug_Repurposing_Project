"""Build the Phase-3 capstone report from the college's four template files + report_content.py.
Usage: build_report.py <template dir> <assets dir> <out.docx> [pages.json]
pages.json (optional) holds page numbers measured from a rendered PDF (find_pages.py) for the contents lists."""
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "pylib"))
sys.path.insert(0, str(HERE))
import docx  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402
from docxcompose.composer import Composer  # noqa: E402

import report_content as C  # noqa: E402

TPL, ASSETS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PAGES = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 and Path(sys.argv[4]).exists() else {}
FONT = "Times New Roman"
A4_W, A4_H = 11906, 16838
TEXT_W_TW = A4_W - 2 * 1234            # template left/right margins (twips)
TEXT_W_CM = TEXT_W_TW / 567.0

# ------------------------------------------------------------------------------------------ numbering pass
LABELS, CITES = {}, {}
fig_list, tab_list, toc = [], [], []


def number_labels():
    for ci, (title, blocks) in enumerate(C.CHAPTERS, 1):
        n = {"fig": 0, "tab": 0, "eq": 0, "alg": 0}
        for b in blocks:
            kind = b[0]
            if kind == "fig":
                n["fig"] += 1
                LABELS[b[1]] = f"Figure {ci}.{n['fig']}"
            elif kind in ("table", "top10"):
                n["tab"] += 1
                LABELS[b[1]] = f"Table {ci}.{n['tab']}"
            elif kind == "eq":
                n["eq"] += 1
                LABELS[b[1]] = f"({ci}.{n['eq']})"
            elif kind == "code":
                n["alg"] += 1
                LABELS[b[1]] = f"Algorithm {ci}.{n['alg']}"


number_labels()


def cite_numbers(keys):
    out = []
    for k in keys:
        k = k.strip()
        if k not in CITES:
            CITES[k] = len(CITES) + 1
        out.append(CITES[k])
    return "[" + ", ".join(str(x) for x in sorted(out)) + "]"


def expand(text):
    text = re.sub(r"\{([a-z]+:[A-Za-z0-9_]+)\}", lambda m: LABELS[m.group(1)] if not m.group(1).startswith("eq:")
                  else "Equation " + LABELS[m.group(1)], text)
    return re.sub(r"\[@([^\]]+)\]", lambda m: cite_numbers(m.group(1).split(",")), text)


# ------------------------------------------------------------------------------------------ low-level helpers
def set_run_font(run, size=None, bold=None, italic=None, name=FONT, color=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_runs(par, text, size=12, base_bold=False, name=FONT):
    text = expand(text)
    for tok in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            set_run_font(par.add_run(tok[2:-2]), size, True, None, name)
        elif tok.startswith("*") and len(tok) > 2:
            set_run_font(par.add_run(tok[1:-1]), size, base_bold, True, name)
        else:
            set_run_font(par.add_run(tok), size, base_bold, None, name)


def fmt(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.5, before=0, after=6, keep_next=False, left=None, first=None):
    pf = par.paragraph_format
    par.alignment = align
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = keep_next
    if left is not None:
        pf.left_indent = Cm(left)
    if first is not None:
        pf.first_line_indent = Cm(first)


def shade(cell, fill):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def borders(table, color="808080", sz=4):
    tblpr = table._element.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), color)
        b.append(e)
    tblpr.append(b)


def set_widths(table, widths_cm):
    table.autofit = False
    tbl = table._element
    grid = tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(w * 567)))
    for row in table.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)
    tblw = tbl.tblPr.find(qn("w:tblW"))
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tbl.tblPr.append(tblw)
    tblw.set(qn("w:w"), str(int(sum(widths_cm) * 567)))
    tblw.set(qn("w:type"), "dxa")
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tbl.tblPr.append(lay)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader")
    h.set(qn("w:val"), "true")
    trpr.append(h)
    cs = OxmlElement("w:cantSplit")
    trpr.append(cs)


def no_split(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def page_field(par, size=10):
    run = par.add_run()
    set_run_font(run, size)
    for kind, text in (("begin", None), (None, " PAGE "), ("separate", None), (None, "1"), ("end", None)):
        if kind:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            r = OxmlElement("w:r")
            r.append(copy.deepcopy(run._element.rPr))
            r.append(fc)
            par._p.append(r)
        else:
            r = OxmlElement("w:r")
            r.append(copy.deepcopy(run._element.rPr))
            t = OxmlElement("w:instrText" if text == " PAGE " else "w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = text
            r.append(t)
            par._p.append(r)


def bottom_border(par, color="000000", sz=6):
    ppr = par._p.get_or_add_pPr()
    for old in ppr.findall(qn("w:pBdr")):
        ppr.remove(old)
    pbdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(sz))
    b.set(qn("w:space"), "1")
    b.set(qn("w:color"), color)
    pbdr.append(b)
    ppr.insert(0, pbdr) if ppr.find(qn("w:pStyle")) is None else ppr.find(qn("w:pStyle")).addnext(pbdr)


def clear_runs(par):
    for r in list(par._p):
        if r.tag in (qn("w:r"), qn("w:hyperlink"), qn("w:fldSimple")):
            par._p.remove(r)


def replace_in_par(par, old, new):
    full = "".join(r.text for r in par.runs)
    if old not in full:
        return False
    runs = par.runs
    runs[0].text = full.replace(old, new)
    for r in runs[1:]:
        r.text = ""
    return True


def replace_everywhere(doc, mapping, tables_only=False):
    pars = [] if tables_only else list(doc.paragraphs)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                pars.extend(cell.paragraphs)
    for p in pars:
        for old, new in mapping:
            replace_in_par(p, old, new)


def a4(doc):
    for s in doc.sections:
        sp = s._sectPr
        pgsz = sp.find(qn("w:pgSz"))
        pgsz.set(qn("w:w"), str(A4_W))
        pgsz.set(qn("w:h"), str(A4_H))


# ------------------------------------------------------------------------------------------ front matter
def front_matter():
    d12 = docx.Document(TPL / "Page 1-2.docx")
    a4(d12)
    for p in d12.paragraphs:
        replace_in_par(p, "“Title of the project”", f"“{C.TITLE}”")
        replace_in_par(p, "‘Title of the project’", f"‘{C.TITLE}’")
        replace_in_par(p, "UE23CS441A– Capstone Project Phase - 3", "UE23CS441A – Capstone Project Phase - 3")
    # project ID line after the course code (as in the Phase-2 report)
    for p in d12.paragraphs:
        if p.text.strip().startswith("UE23CS441A"):
            new = copy.deepcopy(p._p)
            p._p.addnext(new)
            np_ = docx.text.paragraph.Paragraph(new, p._parent)
            replace_in_par(np_, np_.text, "Project ID: PW26_PH_03")
            break
    names = {f"Name {i + 1}": n for i, (n, _) in enumerate(C.TEAM)}
    srns = {f"<SRN {i + 1}>": s for i, (_, s) in enumerate(C.TEAM)}
    replace_everywhere(d12, list(names.items()) + list(srns.items()) +
                       [("Prof. Guide Name", C.GUIDE), ("<Name of the Guide>", C.GUIDE), ("Designation", C.GUIDE_DESIG)],
                       tables_only=True)

    d38 = docx.Document(TPL / "Page 3-8.docx")
    a4(d38)
    for p in d38.paragraphs:
        replace_in_par(p, "“Title of the project”", f"“{C.TITLE}”")
        replace_in_par(p, "<Prof. Guide Name, Designation>", f"{C.GUIDE}, {C.GUIDE_DESIG}, Department of Computer Science and Engineering, PES University")
        replace_in_par(p, "Prof. <Guide Name>", C.GUIDE)
        replace_in_par(p, "his/her", "her")
    replace_everywhere(d38, [(f"<Name {i + 1}>", n) for i, (n, _) in enumerate(C.TEAM)] +
                       [(f"<SRN {i + 1}>", s) for i, (_, s) in enumerate(C.TEAM)] + [("<Signature>", "")], tables_only=True)
    # abstract
    for p in d38.paragraphs:
        if p.text.startswith("[Write the Abstract"):
            clear_runs(p)
            add_runs(p, C.ABSTRACT[0], 12)
            fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.5, 12, 6)
            prev = p
            for extra in C.ABSTRACT[1:]:
                np_ = copy.deepcopy(p._p)
                prev._p.addnext(np_)
                q = docx.text.paragraph.Paragraph(np_, p._parent)
                clear_runs(q)
                add_runs(q, extra, 12)
                prev = q
            break
    # the template pads pages with empty paragraphs (sized for US Letter): use real page breaks instead
    body_el = d38.element.body
    for ch in list(body_el):
        if ch.tag == qn("w:p") and not "".join(ch.itertext()).strip() and ch.find(".//" + qn("w:sectPr")) is None \
                and ch.find(".//" + qn("w:drawing")) is None:
            body_el.remove(ch)
    for p in d38.paragraphs:
        p.paragraph_format.left_indent = Cm(0)
        p.paragraph_format.first_line_indent = Cm(0)
        if p.text.strip() in ("DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT", "TABLE OF CONTENTS", "LIST OF FIGURES",
                              "LIST OF TABLES"):
            p.paragraph_format.page_break_before = True
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(24)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif p.text.strip():
            p.paragraph_format.space_after = Pt(10)
    for t in d38.tables:                                     # small gap after each table
        gap = OxmlElement("w:p")
        t._tbl.addnext(gap)
    return d12, d38


def fill_contents(d38):
    """Rebuild the contents / figures / tables lists (rows cloned from the template rows)."""
    toc_t, fig_t, tab_t = d38.tables[1], d38.tables[2], d38.tables[3]

    def rebuild(table, rows, bold_rows=()):
        proto = copy.deepcopy(table.rows[1]._tr)
        for el in proto.iter():                              # template rows carry auto-numbering and fixed heights
            for child in list(el):
                if child.tag in (qn("w:numPr"), qn("w:trHeight")):
                    el.remove(child)
        for sp in proto.iter(qn("w:spacing")):
            for a in ("w:line", "w:lineRule", "w:before", "w:after"):
                if sp.get(qn(a)) is not None:
                    del sp.attrib[qn(a)]
        for ind in proto.iter(qn("w:ind")):
            ind.getparent().remove(ind)
        for r in list(table.rows)[1:]:
            table._tbl.remove(r._tr)
        for i, vals in enumerate(rows):
            tr = copy.deepcopy(proto)
            table._tbl.append(tr)
            row = table.rows[-1]
            for cell, v in zip(row.cells, vals):
                for extra in cell.paragraphs[1:]:
                    extra._p.getparent().remove(extra._p)
                p = cell.paragraphs[0]
                clear_runs(p)
                set_run_font(p.add_run(str(v)), 11, i in bold_rows)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
            no_split(row)

    rows, bold = [], []
    for ci, (title, blocks) in enumerate(C.CHAPTERS, 1):
        bold.append(len(rows))
        rows.append([str(ci), title, PAGES.get(f"ch{ci}", "")])
        si = 0
        for b in blocks:
            if b[0] == "h2":
                si += 1
                rows.append(["", f"      {ci}.{si} {b[1]}", PAGES.get(f"sec{ci}.{si}", "")])
    for key, title in (("refs", "REFERENCES"), ("appA", "APPENDIX A: DEFINITIONS, ACRONYMS AND ABBREVIATIONS"),
                       ("appB", "APPENDIX B: USER MANUAL")):
        bold.append(len(rows))
        rows.append(["", title, PAGES.get(key, "")])
    rebuild(toc_t, rows, bold)
    rebuild(fig_t, [[n.split()[1], t, PAGES.get(n, "")] for n, t in fig_list])
    rebuild(tab_t, [[n.split()[1], t, PAGES.get(n, "")] for n, t in tab_list])


# ------------------------------------------------------------------------------------------ body
def prepare_body():
    d = docx.Document(TPL / "Page 10 onwards.docx")
    body = d.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    a4(d)
    s = d.sections[0]
    s.different_first_page_header_footer = True
    # header: short title left, logo right, rule below
    hp = s.header.paragraphs
    runs = hp[0].runs
    runs[0].text = C.SHORT_TITLE
    set_run_font(runs[0], 10, None, True)
    tab = OxmlElement("w:tab")
    runs[1]._element.insert(1 if runs[1]._element.rPr is not None else 0, tab)
    hp[0].paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT)
    clear_runs(hp[1])
    bottom_border(hp[1])
    hp[1].paragraph_format.space_after = Pt(0)
    # footer: rule above, page field, table width = text width
    ft = s.footer
    clear_runs(ft.paragraphs[0])
    bottom_border(ft.paragraphs[0])
    tbl = ft.tables[0]
    set_widths(tbl, [TEXT_W_CM / 3] * 3)
    last = tbl.rows[0].cells[2].paragraphs[0]
    clear_runs(last)
    set_run_font(last.add_run("Page "), 10)
    page_field(last, 10)
    # first page of a chapter: no header, same footer
    fpf = s.first_page_footer._element
    for ch in list(fpf):
        fpf.remove(ch)
    for ch in ft._element:
        fpf.append(copy.deepcopy(ch))
    return d


def style_body(d):
    st = d.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(12)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    for name, size, align in (("Heading 1", 18, WD_ALIGN_PARAGRAPH.CENTER), ("Heading 2", 16, WD_ALIGN_PARAGRAPH.LEFT),
                              ("Heading 3", 14, WD_ALIGN_PARAGRAPH.LEFT)):
        h = d.styles[name]
        h.font.name, h.font.size, h.font.bold = FONT, Pt(size), True
        h.font.italic = False
        h.font.color.rgb = RGBColor(0, 0, 0)
        hr = h.element.get_or_add_rPr()
        rf = hr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            hr.insert(0, rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        pf = h.paragraph_format
        pf.alignment = align
        pf.space_before = Pt(18 if name != "Heading 1" else 6)
        pf.space_after = Pt(10)
        pf.keep_with_next = True
        pf.line_spacing = 1.15


def heading(d, text, level, size):
    p = d.add_paragraph(style=f"Heading {level}")
    set_run_font(p.add_run(text), size, True)
    return p


def chapter_start(d, first, label, title):
    if not first:
        d.add_section(WD_SECTION.NEW_PAGE)
    if label:
        p = d.add_paragraph()
        set_run_font(p.add_run(label), 18, True)
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, 0, 4, keep_next=True)
    h = heading(d, title, 1, 18)
    h.paragraph_format.space_after = Pt(18)


def para(d, text):
    p = d.add_paragraph()
    add_runs(p, text)
    fmt(p)
    return p


def list_items(d, items, numbered=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    for i, it in enumerate(items, 1):
        p = d.add_paragraph()
        mark = f"{i}." if numbered else "•"
        set_run_font(p.add_run(mark + "\t"), 12)
        add_runs(p, it)
        fmt(p, align, 1.5, 0, 3, left=1.0, first=-0.6)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(1.0))


def caption(d, label, text, before=4, after=10, keep_next=False):
    p = d.add_paragraph()
    set_run_font(p.add_run(LABELS[label] + ": "), 11, True)
    add_runs(p, text, 11)
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, before, after, keep_next=keep_next)
    return p


def figure(d, label, key, cap, width):
    p = d.add_paragraph()
    p.add_run().add_picture(str(ASSETS / key), width=Cm(min(width, TEXT_W_CM)))
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 8, 2, keep_next=True)
    caption(d, label, cap)
    fig_list.append((LABELS[label], cap))


def table(d, label, cap, header, rows, widths, size):
    caption(d, label, cap, before=10, after=4, keep_next=True)
    tab_list.append((LABELS[label], cap))
    t = d.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    scale = min(1.0, TEXT_W_CM / sum(widths))
    set_widths(t, [w * scale for w in widths])
    group_rows = set()
    for ri, vals in enumerate([header] + rows):
        row = t.rows[ri]
        if ri > 0 and vals[0] == "" and any(vals[1:]):
            group_rows.add(ri)
        for ci, v in enumerate(vals):
            p = row.cells[ci].paragraphs[0]
            add_runs(p, v, size, base_bold=(ri == 0 or ri in group_rows))
            fmt(p, WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER, 1.0, 1.5, 1.5)
            if ri == 0 or ri in group_rows:
                shade(row.cells[ci], "D9D9D9" if ri == 0 else "F2F2F2")
        no_split(row)
    repeat_header(t.rows[0])
    # merge group-header cells: a label spans the following empty cells
    for ri in sorted(group_rows):
        cells = t.rows[ri].cells
        start = None
        for ci in range(1, len(header) + 1):
            if ci == len(header) or rows[ri - 1][ci]:
                if start is not None and ci - 1 > start:
                    keep = cells[start].text
                    cells[start].merge(cells[ci - 1])
                    m = t.rows[ri].cells[start]
                    for extra in m.paragraphs[1:]:
                        extra._p.getparent().remove(extra._p)
                start = ci
    sp = d.add_paragraph()
    fmt(sp, line=1.0, after=4)


def top10(d, label, cap, start):
    groups = C.TOP10[start:start + 3]
    header = ["Rank"] + [g for g, _ in groups]
    rows = [[str(i + 1)] + [g[1][i] for g in groups] for i in range(10)]
    table(d, label, cap, header, rows, [1.4, 5.0, 5.0, 5.0], 10)


def equation(d, label, text):
    p = d.add_paragraph()
    run = p.add_run("\t" + text + "\t" + LABELS[label])
    set_run_font(run, 12, name="Cambria Math")
    fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.15, 6, 8)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM / 2), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT)


def code(d, label, cap, lines):
    caption(d, label, cap, before=10, after=4, keep_next=True)
    t = d.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t, "A6A6A6")
    set_widths(t, [TEXT_W_CM - 0.4])
    cell = t.rows[0].cells[0]
    shade(cell, "F2F2F2")
    no_split(t.rows[0])
    first = True
    for line in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        set_run_font(p.add_run(line), 8.5, name="Courier New")
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 0)
    sp = d.add_paragraph()
    fmt(sp, line=1.0, after=4)


def body():
    d = prepare_body()
    style_body(d)
    for ci, (title, blocks) in enumerate(C.CHAPTERS, 1):
        chapter_start(d, ci == 1, f"CHAPTER - {ci}", title)
        si = ssi = 0
        for b in blocks:
            k = b[0]
            if k == "h2":
                si += 1
                ssi = 0
                heading(d, f"{ci}.{si} {b[1]}", 2, 16)
            elif k == "h3":
                ssi += 1
                heading(d, f"{ci}.{si}.{ssi} {b[1]}", 3, 14)
            elif k == "p":
                para(d, b[1])
            elif k == "bullets":
                list_items(d, b[1])
            elif k == "numbered":
                list_items(d, b[1], numbered=True)
            elif k == "fig":
                figure(d, b[1], b[2], b[3], b[4])
            elif k == "table":
                table(d, *b[1:])
            elif k == "top10":
                top10(d, b[1], b[2], b[3])
            elif k == "eq":
                equation(d, b[1], b[2])
            elif k == "code":
                code(d, b[1], b[2], b[3])
            else:
                raise ValueError(k)
    # references (IEEE, numbered by first citation)
    chapter_start(d, False, None, "REFERENCES")
    refs = REFS
    for key, n in sorted(CITES.items(), key=lambda kv: kv[1]):
        p = d.add_paragraph()
        set_run_font(p.add_run(f"[{n}]\t"), 11)
        set_run_font(p.add_run(refs[key]), 11)
        fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.15, 0, 6, left=1.0, first=-1.0)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(1.0))
    missing = set(CITES) - set(refs)
    assert not missing, missing
    # appendices
    chapter_start(d, False, None, "APPENDIX A: DEFINITIONS, ACRONYMS AND ABBREVIATIONS")
    t = d.add_table(rows=1 + len(C.APPENDIX_A), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    set_widths(t, [4.6, 11.8])
    for ri, (a, b) in enumerate([("Term", "Meaning")] + C.APPENDIX_A):
        for ci, v in enumerate((a, b)):
            p = t.rows[ri].cells[ci].paragraphs[0]
            add_runs(p, v, 11, base_bold=(ri == 0))
            fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 1.5, 1.5)
            if ri == 0:
                shade(t.rows[ri].cells[ci], "D9D9D9")
        no_split(t.rows[ri])
    repeat_header(t.rows[0])
    chapter_start(d, False, None, "APPENDIX B: USER MANUAL")
    for b in C.APPENDIX_B:
        if b[0] == "p":
            para(d, b[1])
        else:
            list_items(d, b[1], numbered=(b[0] == "numbered"), align=WD_ALIGN_PARAGRAPH.LEFT)
    # sections after the first: inherit header/footer, keep numbering continuous, chapter pages without header
    secs = d.element.body.findall(".//" + qn("w:sectPr"))
    for i, sp in enumerate(secs):
        if sp.find(qn("w:titlePg")) is None:
            sp.append(OxmlElement("w:titlePg"))
        if i > 0:
            for pn in sp.findall(qn("w:pgNumType")):
                sp.remove(pn)
    # the first body section restarts at page 1 (front matter is unnumbered)
    first_sp = secs[0]
    pn = first_sp.find(qn("w:pgNumType"))
    if pn is None:
        pn = OxmlElement("w:pgNumType")
        first_sp.append(pn)
    pn.set(qn("w:start"), "1")
    return d


REFS = {
    "bonanno2018amifampridine": "S. Bonanno et al., “Amifampridine phosphate in the treatment of muscle-specific kinase myasthenia gravis: a phase IIb, randomized, double-blind, placebo-controlled, double crossover study,” SAGE Open Medicine, vol. 6, p. 2050312118819013, 2018, doi: 10.1177/2050312118819013.",
    "remijn2026amifampridine": "L. Remijn-Nelissen et al., “Efficacy and safety of amifampridine in myasthenia gravis: a randomized, double-blind, placebo-controlled crossover trial,” Neurology, vol. 106, no. 8, p. e214715, 2026, doi: 10.1212/WNL.0000000000214715.",
    "pushpakom2019repurposing": "S. Pushpakom et al., “Drug repurposing: progress, challenges and recommendations,” Nature Reviews Drug Discovery, vol. 18, no. 1, pp. 41–58, 2019, doi: 10.1038/nrd.2018.168.",
    "huang2024txgnn": "K. Huang et al., “A foundation model for clinician-centered drug repurposing,” Nature Medicine, vol. 30, no. 12, pp. 3601–3613, 2024, doi: 10.1038/s41591-024-03233-x.",
    "nguengang2020rare": "S. Nguengang Wakap et al., “Estimating cumulative point prevalence of rare diseases: analysis of the Orphanet database,” European Journal of Human Genetics, vol. 28, no. 2, pp. 165–173, 2020, doi: 10.1038/s41431-019-0508-0.",
    "conrad2023autoimmune": "N. Conrad et al., “Incidence, prevalence, and co-occurrence of autoimmune disorders over time and by age, sex, and socioeconomic status: a population-based cohort study of 22 million individuals in the UK,” The Lancet, vol. 401, no. 10391, pp. 1878–1890, 2023, doi: 10.1016/S0140-6736(23)00457-9.",
    "himmelstein2017hetionet": "D. S. Himmelstein et al., “Systematic integration of biomedical knowledge prioritizes drugs for repurposing,” eLife, vol. 6, p. e26726, 2017, doi: 10.7554/eLife.26726.",
    "bonner2022review": "S. Bonner et al., “A review of biomedical datasets relating to drug discovery: a knowledge graph perspective,” Briefings in Bioinformatics, vol. 23, no. 6, p. bbac404, 2022, doi: 10.1093/bib/bbac404.",
    "bonner2022kge": "S. Bonner et al., “Understanding the performance of knowledge graph embeddings in drug discovery,” Artificial Intelligence in the Life Sciences, vol. 2, p. 100036, 2022, doi: 10.1016/j.ailsci.2022.100036.",
    "hu2026biopathnet": "E. Y. Hu et al., “Enhancing link prediction in biomedical knowledge graphs with BioPathNet,” Nature Biomedical Engineering, vol. 10, no. 8, pp. 1747–1769, 2026, doi: 10.1038/s41551-025-01598-z.",
    "zhu2021nbfnet": "Z. Zhu, Z. Zhang, L.-P. Xhonneux and J. Tang, “Neural Bellman-Ford networks: A general graph neural network framework for link prediction,” in Advances in Neural Information Processing Systems (NeurIPS), 2021.",
    "chandak2023primekg": "P. Chandak, K. Huang and M. Zitnik, “Building a knowledge graph to enable precision medicine,” Scientific Data, vol. 10, p. 67, 2023, doi: 10.1038/s41597-023-01960-3.",
    "drkg2020": "V. N. Ioannidis et al., “DRKG – Drug Repurposing Knowledge Graph for COVID-19,” 2020. [Online]. Available: https://github.com/gnn4dr/DRKG",
    "bordes2013transe": "A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston and O. Yakhnenko, “Translating embeddings for modeling multi-relational data,” in Advances in Neural Information Processing Systems (NeurIPS), 2013.",
    "yang2015distmult": "B. Yang, W.-t. Yih, X. He, J. Gao and L. Deng, “Embedding entities and relations for learning and inference in knowledge bases,” in Proc. International Conference on Learning Representations (ICLR), 2015.",
    "sun2019rotate": "Z. Sun, Z.-H. Deng, J.-Y. Nie and J. Tang, “RotatE: Knowledge graph embedding by relational rotation in complex space,” in Proc. International Conference on Learning Representations (ICLR), 2019.",
    "schlichtkrull2018rgcn": "M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov and M. Welling, “Modeling relational data with graph convolutional networks,” in The Semantic Web (ESWC), Lecture Notes in Computer Science, vol. 10843, pp. 593–607, 2018, doi: 10.1007/978-3-319-93417-4_38.",
    "zhang2021covidkg": "R. Zhang, D. Hristovski, D. Schutte, A. Kastrin, M. Fiszman and H. Kilicoglu, “Drug repurposing for COVID-19 via knowledge graph completion,” Journal of Biomedical Informatics, vol. 115, p. 103696, 2021, doi: 10.1016/j.jbi.2021.103696.",
    "morselli2021covid": "D. Morselli Gysi et al., “Network medicine framework for identifying drug-repurposing opportunities for COVID-19,” Proceedings of the National Academy of Sciences, vol. 118, no. 19, p. e2025581118, 2021, doi: 10.1073/pnas.2025581118.",
    "sosa2020rare": "D. N. Sosa, A. Derry, M. Guo, E. Wei, C. Brinton and R. B. Altman, “A literature-based knowledge graph embedding method for identifying drug repurposing opportunities in rare diseases,” in Pacific Symposium on Biocomputing, vol. 25, pp. 463–474, 2020, doi: 10.1142/9789811215636_0041.",
    "zitnik2018decagon": "M. Zitnik, M. Agrawal and J. Leskovec, “Modeling polypharmacy side effects with graph convolutional networks,” Bioinformatics, vol. 34, no. 13, pp. i457–i466, 2018, doi: 10.1093/bioinformatics/bty294.",
    "gao2022kgpredict": "Z. Gao, P. Ding and R. Xu, “KG-Predict: A knowledge graph computational framework for drug repurposing,” Journal of Biomedical Informatics, vol. 132, p. 104133, 2022, doi: 10.1016/j.jbi.2022.104133.",
    "bang2023dreamwalk": "D. Bang, S. Lim, S. Lee and S. Kim, “Biomedical knowledge graph learning for drug repurposing by extending guilt-by-association to multiple layers,” Nature Communications, vol. 14, p. 3570, 2023, doi: 10.1038/s41467-023-39301-y.",
    "ali2021pykeen": "M. Ali et al., “Bringing light into the dark: A large-scale evaluation of knowledge graph embedding models under a unified framework,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 12, pp. 8825–8845, 2022, doi: 10.1109/TPAMI.2021.3124805.",
    "teru2020grail": "K. K. Teru, E. Denis and W. L. Hamilton, “Inductive relation prediction by subgraph reasoning,” in Proc. International Conference on Machine Learning (ICML), 2020.",
    "zhu2023astarnet": "Z. Zhu et al., “A*Net: A scalable path-based reasoning approach for knowledge graphs,” in Advances in Neural Information Processing Systems (NeurIPS), 2023, doi: 10.52202/075280-2591.",
    "galkin2024ultra": "M. Galkin, X. Yuan, H. Mostafa, J. Tang and Z. Zhu, “Towards foundation models for knowledge graph reasoning,” in Proc. International Conference on Learning Representations (ICLR), 2024.",
    "kapoor2023leakage": "S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, p. 100804, 2023, doi: 10.1016/j.patter.2023.100804.",
    "briere2026leakage": "G. Brière, T. Stosskopf, B. Loire and A. Baudot, “Benchmarking the impact of data leakage on the performance of knowledge graph embedding models for biomedical link prediction,” Bioinformatics, vol. 42, no. 8, p. btag608, 2026, doi: 10.1093/bioinformatics/btag608.",
    "saito2015prc": "T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” PLoS ONE, vol. 10, no. 3, p. e0118432, 2015, doi: 10.1371/journal.pone.0118432.",
    "davis2006prroc": "J. Davis and M. Goadrich, “The relationship between precision-recall and ROC curves,” in Proc. 23rd International Conference on Machine Learning (ICML), pp. 233–240, 2006, doi: 10.1145/1143844.1143874.",
    "voorhees1999trec": "E. M. Voorhees, “The TREC-8 question answering track report,” in Proc. Eighth Text REtrieval Conference (TREC-8), 1999.",
    "efron1993bootstrap": "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993.",
    "wilcoxon1945": "F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945, doi: 10.2307/3001968.",
    "mccormick1994thalidomide": "P. A. McCormick, F. Scott, O. Epstein, A. K. Burroughs, P. J. Scheuer and N. McIntyre, “Thalidomide as therapy for primary biliary cirrhosis: a double-blind placebo controlled pilot study,” Journal of Hepatology, vol. 21, no. 4, pp. 496–499, 1994, doi: 10.1016/S0168-8278(94)80092-8.",
    "corso2020pna": "G. Corso, L. Cavalleri, D. Beaini, P. Liò and P. Veličković, “Principal neighbourhood aggregation for graph nets,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.",
}


def main():
    d12, d38 = front_matter()
    b = body()                       # fills fig_list / tab_list / CITES
    fill_contents(d38)
    comp = Composer(d12)
    comp.append(d38)
    front = OUT.with_name(OUT.stem + "_front.docx")
    comp.save(str(front))
    b.save(str(OUT))
    words = sum(len(a.split()) for a in C.ABSTRACT)
    print(f"saved {front.name} + {OUT.name} | abstract {words} words | {len(CITES)} references | {len(fig_list)} figures | {len(tab_list)} tables")


main()

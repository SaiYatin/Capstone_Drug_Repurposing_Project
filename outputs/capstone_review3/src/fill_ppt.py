"""Fill the restructured PES Review-3 template with BioPathNet++ content. Usage: fill_ppt.py <in.pptx> <assets> <out.pptx>
All numbers are the verified project results (same sources as the report)."""
import re
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

SRC, ASSETS, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
FONT = "Trebuchet MS"
DARK, MUTED = RGBColor(0x26, 0x26, 0x26), RGBColor(0x59, 0x59, 0x59)
ORANGE, RUST, GREY = RGBColor(0xE4, 0x83, 0x12), RGBColor(0xBD, 0x58, 0x2C), RGBColor(0x9A, 0x9A, 0x9A)
SLATE, TINT, TINT2 = RGBColor(0x3B, 0x4A, 0x5A), RGBColor(0xFD, 0xF1, 0xE5), RGBColor(0xF2, 0xF2, 0xF2)
YELLOW = RGBColor(0xFF, 0xF2, 0x99)
TEAM = "TVNS Yatin  |  Srijan Jha  |  A Charan KR  |  Praveen Kumar Yadav"
SHORT = "BioPathNet++ for Autoimmune Diseases"

prs = Presentation(SRC)
S = prs.slides


# ------------------------------------------------------------------------------------------------- helpers
def set_text_keep_format(shape, text):
    """Replace a shape's text, keeping the first run's formatting."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    for extra in tf.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    runs = p0.runs
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p0.add_run().text = text


def delete(shape):
    shape._element.getparent().remove(shape._element)


def frame(slide, title):
    """Header label, footer names, slide title; returns the body shape(s) to remove."""
    others, title_shape = [], None
    for sh in list(slide.shapes):
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.strip() == "Title of the Project":
            set_text_keep_format(sh, SHORT)
        elif "name1_name2" in t:
            set_text_keep_format(sh, TEAM)
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(11)
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        elif t.strip():
            others.append(sh)
    # the slide title is the template's red text box nearest the top; every other text box is template body text
    tops = [sh for sh in others if sh.top is not None and sh.top < Inches(1.5) and sh.height < Inches(1.0)]
    title_shape = min(tops, key=lambda sh: sh.top) if tops else None
    for sh in others:
        if sh is not title_shape:
            delete(sh)
    set_text_keep_format(title_shape, title)
    # same title position on every content slide (the template varies), just above its orange rule
    title_shape.left, title_shape.top, title_shape.width, title_shape.height = Inches(2.6), Inches(1.15), Inches(9.1), Inches(0.5)
    title_shape.text_frame.word_wrap = False
    for sh in slide.shapes:                                  # the thin orange rule under the title
        if sh.height is not None and sh.height < Inches(0.08) and sh.width > Inches(5):
            sh.left, sh.top, sh.width = Inches(3.0), Inches(1.73), Inches(8.7)


def runs_from_markup(p, text, size, color, bold=False, font=FONT):
    for tok in re.split(r"(\*\*.+?\*\*)", text):
        if not tok:
            continue
        r = p.add_run()
        r.text = tok[2:-2] if tok.startswith("**") else tok
        r.font.name, r.font.size = font, Pt(size)
        r.font.bold = bold or tok.startswith("**")
        r.font.color.rgb = color


def bulletize(p, level=0, char="•", color=ORANGE):
    ppr = p._p.get_or_add_pPr()
    ppr.set("marL", str(int(Inches(0.28 + 0.3 * level))))
    ppr.set("indent", str(-int(Inches(0.24))))
    buclr = etree.SubElement(ppr, qn("a:buClr"))
    etree.SubElement(buclr, qn("a:srgbClr")).set("val", str(color))
    etree.SubElement(ppr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(ppr, qn("a:buChar")).set("char", char)


def text(slide, x, y, w, h, items, size=16, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=6, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(0.04))
    for i, it in enumerate(items):
        if isinstance(it, str):
            it = {"t": it}
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = it.get("align", align)
        p.space_after = Pt(it.get("space", space))
        runs_from_markup(p, it["t"], it.get("size", size), it.get("color", color), it.get("bold", False),
                         it.get("font", FONT))
        if it.get("bullet"):
            bulletize(p, it.get("level", 0))
    return tb


def card(slide, x, y, w, h, fill=TINT, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    return s


def stat(slide, x, y, w, big, label, sub=None, big_size=36, color=ORANGE):
    card(slide, x, y, w, 1.55 if sub else 1.3)
    text(slide, x + 0.1, y + 0.08, w - 0.2, 0.7, [{"t": big, "size": big_size, "bold": True, "color": color}],
         align=PP_ALIGN.CENTER)
    items = [{"t": label, "size": 13, "color": DARK, "bold": True, "space": 2}]
    if sub:
        items.append({"t": sub, "size": 11, "color": MUTED})
    text(slide, x + 0.12, y + 0.78, w - 0.24, 0.75, items, align=PP_ALIGN.CENTER)


def picture(slide, key, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(ASSETS / key), Inches(x), Inches(y), **kw)


def table(slide, x, y, w, rows, col_w, size=12, header_fill=ORANGE, row_h=0.32, first_bold=False):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    tbl.first_row = True
    for ci, cw in enumerate(col_w):
        tbl.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.paragraphs[0].text = ""
            p = tf.paragraphs[0]
            hdr = ri == 0
            runs_from_markup(p, str(val), size, RGBColor(0xFF, 0xFF, 0xFF) if hdr else DARK,
                             bold=hdr or (first_bold and ci == 0))
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if hdr else (TINT if ri % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
    return tbl


def bar_chart(slide, x, y, w, h, cats, series, colors, title=None, fmt='0.000', maxv=None, legend=False,
              horizontal=False, label_size=12, point_colors=None):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    ch = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name, ch.font.size, ch.font.color.rgb = FONT, Pt(12), DARK
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title
        tp = ch.chart_title.text_frame.paragraphs[0]
        for r in tp.runs:
            r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(14), True, FONT, DARK
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -10 if len(series) > 1 else 0
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.color.rgb, dl.font.name = Pt(label_size), DARK, FONT
    for si, s in enumerate(plot.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = colors[si]
        s.invert_if_negative = False
        if point_colors and si == 0:
            for pi, c in enumerate(point_colors):
                pt = s.points[pi]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = c
    va = ch.value_axis
    va.visible = False
    va.has_major_gridlines = False
    if maxv is not None:
        va.maximum_scale = maxv
    va.minimum_scale = 0
    ca = ch.category_axis
    ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(12), DARK
    ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    ca.has_major_gridlines = False
    return ch


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ------------------------------------------------------------------------------------------------- slides
# 1. Title
s = S[0]
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.strip() == "Title of the Project":
        set_text_keep_format(sh, SHORT)
    elif sh.has_text_frame and "name1_name2" in sh.text_frame.text:
        set_text_keep_format(sh, TEAM)
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.color.rgb = Pt(11), RGBColor(0xFF, 0xFF, 0xFF)
    elif sh.has_text_frame and "Project Title" in sh.text_frame.text:
        sh.top, sh.height, sh.left, sh.width = Inches(3.05), Inches(3.6), Inches(1.0), Inches(11.4)
        vals = [("Project Title", "BioPathNet++ for Autoimmune Diseases: Path-Based Transfer Learning for Rare Disease Drug Repurposing"),
                ("Project ID", "PW26_PH_03"), ("Project Guide", "Dr. Priyanka H, Associate Professor"),
                ("Project Team", "TVNS Yatin (PES1UG23CS643), Srijan Jha (PES1UG23AM316),\nA Charan KR (PES1UG23CS101), Praveen Kumar Yadav (PES1UG23CS449)")]
        tf = sh.text_frame
        for extra in tf.paragraphs[1:]:
            extra._p.getparent().remove(extra._p)
        tf.paragraphs[0].text = ""
        for i, (k, v) in enumerate(vals):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(8)
            r = p.add_run()
            r.text = f"{k}:  "
            r.font.name, r.font.size, r.font.color.rgb, r.font.bold = FONT, Pt(18), RGBColor(0x00, 0x33, 0xCC), True
            lines = v.split("\n")
            for li, line in enumerate(lines):
                if li:
                    p.add_line_break()
                r = p.add_run()
                r.text = line
                r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(18), DARK
notes(s, "Introduce the team and the project. One sentence: we rank existing drugs for rare autoimmune diseases whose "
         "treatments a model has never seen, and we made sure the evaluation is leakage-free.")

# 2. Outline
s = S[1]
frame(s, "Outline")
items = ["Abstract", "Suggestions from Review 2 and key changes", "Implementation details", "Project demonstration",
         "Test plan and strategy", "Results and discussion", "Applicability to other domains",
         "Timeline and report status", "Conclusion and future work", "IEEE drafts and target venues", "References"]
for i, it in enumerate(items):
    col, row = divmod(i, 6)
    x, y = 1.2 + col * 5.8, 2.05 + row * 0.72
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.48), Inches(0.48))
    c.fill.solid()
    c.fill.fore_color.rgb = ORANGE
    c.line.fill.background()
    c.text_frame.text = str(i + 1)
    c.text_frame.word_wrap = False
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(c.text_frame, m, 0)
    for p in c.text_frame.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(13), True, FONT, RGBColor(0xFF, 0xFF, 0xFF)
    text(s, x + 0.65, y + 0.02, 4.9, 0.48, [{"t": it, "size": 18}], anchor=MSO_ANCHOR.MIDDLE)

# 3. Abstract
s = S[2]
frame(s, "Abstract")
text(s, 0.7, 2.0, 7.3, 4.7, [
    {"t": "**Problem.** Rare autoimmune diseases (PBC, systemic sclerosis, Sjögren, autoimmune hepatitis, myasthenia "
          "gravis, Crohn) have few approved drugs. Repurposing must work **zero-shot**: rank existing drugs for a disease "
          "whose treatments the model has never seen.", "size": 15, "space": 10},
    {"t": "**Approach.** BioPathNet++: a path-based graph neural network (Neural Bellman-Ford) on PrimeKG "
          "(129,375 nodes, 4 million edges) plus a leakage-free transfer channel, trained on free Kaggle T4 GPUs.",
     "size": 15, "space": 10},
    {"t": "**Rigour.** An audit showed our earlier near-perfect scores came from leakage. We rebuilt everything: "
          "strict-family hold-out, full ranking of all 7,957 drugs, six baselines, 3 seeds, paired tests, TxGNN "
          "head-to-head, five validation layers.", "size": 15, "space": 10},
    {"t": "**Scope.** Ranked hypotheses for expert review, not treatment advice.", "size": 15}])
stat(s, 8.35, 2.0, 4.3, "76%", "unseen test diseases with an approved drug in the top 10",
     "best baseline 57%, random 0.8%")
stat(s, 8.35, 3.7, 4.3, "0.501 vs 0.392", "MRR vs best baseline (68 test diseases)", "p = 0.003, 3 seeds", big_size=28)
stat(s, 8.35, 5.4, 4.3, "7,957", "drugs ranked for every disease", big_size=30)
notes(s, "Key message: a leakage-controlled, path-based model that beats every baseline on diseases it has never seen.")

# 4. Suggestions from Review 2
s = S[3]
frame(s, "Suggestions from Review - 2")
card(s, 0.7, 2.0, 5.6, 4.6, fill=TINT2)
text(s, 0.95, 2.15, 5.1, 0.5, [{"t": "Panel remarks (Review 2)", "size": 18, "bold": True, "color": RUST}])
ph = text(s, 0.95, 2.8, 5.1, 3.6, [{"t": "[Add the panel's remarks from Review 2 and our response to each.]",
                                    "size": 15, "color": DARK}], name="TODO_review2_remarks")
ph.fill.solid()
ph.fill.fore_color.rgb = YELLOW
card(s, 6.6, 2.0, 6.1, 4.6)
text(s, 6.85, 2.15, 5.6, 0.5, [{"t": "Key changes made in Phase 3", "size": 18, "bold": True, "color": RUST}])
text(s, 6.85, 2.75, 5.7, 3.8, [
    {"t": "Audited the earlier pipeline: its AUPRC 0.935-0.999 came from leakage (277/277 test pairs were graph "
          "edges; a no-learning degree rule scored AUROC 0.972). Those numbers are withdrawn.", "bullet": True, "size": 14},
    {"t": "Rebuilt on PrimeKG with a strict-family zero-shot protocol and full ranking.", "bullet": True, "size": 14},
    {"t": "Added six baselines, 3 seeds, paired tests and a selection-free test split.", "bullet": True, "size": 14},
    {"t": "Head-to-head with TxGNN on TxGNN's own split; five validation layers.", "bullet": True, "size": 14},
    {"t": "IEEE conference and journal drafts; college Turnitin similarity 0%.", "bullet": True, "size": 14}])
notes(s, "Fill in the panel's actual Review-2 remarks in the yellow box before presenting.")

# 5. Implementation 1: architecture
s = S[4]
frame(s, "Implementation Details: System Architecture")
picture(s, "diag_architecture.png", 0.55, 1.95, h=4.75)
text(s, 8.45, 1.95, 4.3, 4.8, [
    {"t": "Technology", "size": 17, "bold": True, "color": RUST, "space": 8},
    {"t": "Python 3.13, PyTorch sparse CSR", "bullet": True, "size": 14},
    {"t": "PrimeKG knowledge graph (Harvard)", "bullet": True, "size": 14},
    {"t": "Kaggle notebooks, 2 x NVIDIA T4", "bullet": True, "size": 14},
    {"t": "pytest, Git / GitHub", "bullet": True, "size": 14},
    {"t": "DGL 2.4 in a separate env (TxGNN)", "bullet": True, "size": 14},
    {"t": "ClinicalTrials.gov and PubMed APIs", "bullet": True, "size": 14, "space": 12},
    {"t": "Package **bpp**: 19 modules, about 3,900 lines; every paper number is generated from run logs.",
     "size": 13, "color": MUTED}])
notes(s, "Four layers: data, protocol (what is held out), learning, evaluation and reporting.")

# 6. Implementation 2: protocol + audit
s = S[5]
frame(s, "Implementation Details: Leakage-Free Protocol")
picture(s, "diag_strict_family.png", 0.6, 1.95, w=7.6)
text(s, 0.6, 5.55, 7.6, 1.2, [
    {"t": "A disease is held out together with its whole family (disease-disease neighbours). Without this, all 16 Crohn "
          "drugs stayed reachable via \"Crohn's colitis\".", "size": 13, "color": MUTED}])
card(s, 8.55, 1.95, 4.15, 4.75, fill=TINT2)
text(s, 8.75, 2.05, 3.8, 0.5, [{"t": "Why: audit of the old pipeline", "size": 16, "bold": True, "color": RUST}])
for i, (big, lab) in enumerate([("277 / 277", "test pairs were edges of the graph"),
                                ("0.972", "test AUROC of a no-learning degree rule (model 0.941)"),
                                ("5 / 6", "targets unreachable: same drug ranked first everywhere")]):
    y = 2.6 + i * 1.33
    text(s, 8.75, y, 3.8, 0.55, [{"t": big, "size": 26, "bold": True, "color": ORANGE}])
    text(s, 8.75, y + 0.55, 3.8, 0.7, [{"t": lab, "size": 13}])
notes(s, "This is the most important design decision: the evaluation, not the model, is what makes results trustworthy.")

# 7. Implementation 3: model
s = S[6]
frame(s, "Implementation Details: Model")
picture(s, "fig_pipeline.png", 2.3, 1.85, w=8.7)
cards = [("Path channel (NBFNet)", ["Starts at the query disease, 6 layers", "Relation-specific DistMult messages",
                                   "Scores all 7,957 drugs in one pass", "Needs no embedding of the disease"]),
         ("Memory trick (RelSpMM)", ["60 sparse relation matrices", "Recompute A_r H in backward pass",
                                     "Memory O(V x B x d), not O(E x B x d)", "Full 4M-edge graph: peak 5.8 GB on a T4"]),
         ("Transfer channel + loss", ["4 leakage-free priors (popularity, kNN,", "shared targets, autoimmune frequency)",
                                      "Listwise softmax over all drugs", "68,226 parameters in total"])]
for i, (h, lines) in enumerate(cards):
    x = 0.7 + i * 4.1
    card(s, x, 5.12, 3.85, 1.68)
    text(s, x + 0.15, 5.15, 3.6, 0.42, [{"t": h, "size": 15, "bold": True, "color": RUST}])
    text(s, x + 0.15, 5.55, 3.6, 1.2, [{"t": l, "bullet": True, "size": 12, "space": 1} for l in lines])
notes(s, "Path reasoning is inductive: a disease never seen in training is handled exactly like any other node.")

# 8. Implementation 4: algorithm + modules
s = S[7]
frame(s, "Implementation Details: Algorithm and Modules")
algo = ["for epoch in 1..20:",
        "  for batch U of 8 training diseases:",
        "    G_b <- graph without drug edges of families(U)",
        "    h <- indicator on each u in U",
        "    for t in 1..6:  h <- NBFLayer_t(h, G_b)",
        "    s <- MLP(h at all drugs) + g(prior(U))",
        "    loss <- listwise softmax(s, known drugs)",
        "    backward; clip; Adam step",
        "  validation MRR -> best.pt; last.pt",
        "  stop after 6 epochs without gain"]
card(s, 0.6, 1.95, 5.9, 4.8, fill=TINT2)
text(s, 0.8, 2.05, 5.5, 0.45, [{"t": "Training loop (pseudocode)", "size": 15, "bold": True, "color": RUST}])
text(s, 0.8, 2.55, 5.6, 4.1, [{"t": l, "size": 12.5, "font": "Courier New", "space": 3} for l in algo])
table(s, 6.8, 1.95, 5.9, [["Module", "Role"],
                          ["build_primekg.py", "graph + splits (35 s)"], ["splits.py", "strict-family hold-out"],
                          ["task.py", "labels, priors, baselines"], ["model.py", "RelSpMM, NBF layers"],
                          ["train.py", "training, resume, early stop"], ["metrics.py", "full ranking, bootstrap"],
                          ["analyze.py", "paired tests, paper numbers"], ["validate.py", "trials, PubMed, temporal"],
                          ["explain.py", "path explanations"], ["build_txgnn.py", "TxGNN split converter"]],
      [2.3, 3.6], size=12, row_h=0.43)
notes(s, "Each batch removes its own diseases' drug edges, so training mirrors the zero-shot test condition.")

# 9. Demonstration
s = S[8]
frame(s, "Project Demonstration")
text(s, 0.6, 1.95, 6.0, 0.45, [{"t": "Example output: Myasthenia gravis (never seen in training)", "size": 15,
                                "bold": True, "color": RUST}])
table(s, 0.6, 2.45, 6.0, [["Rank", "Drug", "Evidence"],
                          ["1", "Ambenonium", "★ known (PrimeKG)"], ["2", "Neostigmine", "★ known"],
                          ["3", "Pyridostigmine", "★ known, phase 3/4 trials"], ["4", "Distigmine", "★ known"],
                          ["5", "Prednisolone", "★ known"], ["7", "**Amifampridine**", "**new: phase-3 trial NCT03304054**"]],
      [0.8, 2.2, 3.0], size=12, row_h=0.4)
text(s, 0.6, 5.35, 6.0, 1.3, [{"t": "★ = drug PrimeKG lists for MG; ranks averaged over 3 seeds. The model saw none of "
                                    "MG's drug links.", "size": 12, "color": MUTED}])
card(s, 6.95, 1.95, 5.75, 2.25, fill=TINT2)
text(s, 7.15, 2.05, 5.4, 0.45, [{"t": "Why pyridostigmine? (path explanation)", "size": 15, "bold": True, "color": RUST}])
text(s, 7.15, 2.55, 5.4, 1.6, [{"t": "myasthenia gravis  →  MUSK gene  →  neuromuscular junction disease  →  "
                                     "pyridostigmine", "size": 14},
                                {"t": "Gradient x input on every edge, then beam search for the strongest paths.",
                                 "size": 12, "color": MUTED}])
card(s, 6.95, 4.4, 5.75, 2.35, fill=TINT2)
text(s, 7.15, 4.5, 5.4, 0.45, [{"t": "Live demo (laptop GPU)", "size": 15, "bold": True, "color": RUST}])
text(s, 7.15, 4.95, 5.5, 1.8, [{"t": "python -m bpp.score --run <run> --out scores.npz", "size": 11.5, "font": "Courier New"},
                               {"t": "python -m bpp.explain --run <run> --out paths.json", "size": 11.5, "font": "Courier New"},
                               {"t": "Scoring reproduces the Kaggle top-10 lists on the laptop (17 of 18; one exact tie).",
                                "size": 12, "color": MUTED}])
notes(s, "Show the ranked list and one explanation path. Training itself runs on Kaggle (2 to 4 hours per model), "
         "so the live part is scoring and explanation with a saved checkpoint.")

# 10. Test plan
s = S[9]
frame(s, "Test Plan and Strategy")
table(s, 0.6, 1.95, 7.6, [["Test activity", "What it checks"],
                          ["Unit tests (pytest, 9)", "gradcheck, edge removal, reachability"],
                          ["Leakage assertions", "no test pair reaches training"],
                          ["Reproducibility checks", "Kaggle code = repo; resumed runs"],
                          ["Statistical testing", "3 seeds, bootstrap CI + Wilcoxon"],
                          ["Selection-free test split", "68 diseases never used to tune"],
                          ["Local rehearsal", "TxGNN pipeline on mini PrimeKG"],
                          ["External validation", "trials, PubMed, later approvals"]],
      [3.0, 4.6], size=12.5, row_h=0.5)
text(s, 8.5, 1.95, 4.25, 4.8, [
    {"t": "Why this approach", "size": 16, "bold": True, "color": RUST, "space": 4},
    {"t": "A research model fails silently: leakage gives great numbers, so tests target leakage and statistics.",
     "size": 13, "space": 10},
    {"t": "Environment", "size": 16, "bold": True, "color": RUST, "space": 4},
    {"t": "Laptop (RTX 3050) for build, tests, analysis; Kaggle 2 x T4 for training.", "size": 13, "space": 10},
    {"t": "Team roles", "size": 16, "bold": True, "color": RUST, "space": 4},
    {"t": "[Fill in each member's testing role]", "size": 13},
    {"t": "Drawback", "size": 16, "bold": True, "color": RUST, "space": 4},
    {"t": "Small rare-disease sets give low statistical power.", "size": 13}])
for sh in s.shapes:
    if sh.has_text_frame and "[Fill in each member" in sh.text_frame.text:
        sh.name = "TODO_team_roles"
notes(s, "Every downloaded Kaggle result was verified against the repository code and logs before use.")

# 11. Results 1: accuracy
s = S[10]
frame(s, "Results and Discussion: Accuracy on Unseen Diseases")
bar_chart(s, 0.5, 1.9, 6.1, 4.0, ["Random", "Popularity", "Disease kNN", "BioPathNet++"],
          [("Any@10", (0.8, 19, 57, 76))], [GREY], title="Approved drug in top 10 (% of 68 test diseases)",
          fmt='0"%"', maxv=95, point_colors=[GREY, GREY, GREY, ORANGE])
bar_chart(s, 6.7, 1.9, 6.1, 4.0, ["Random", "Popularity", "Disease kNN", "BioPathNet++"],
          [("MRR", (0.001, 0.062, 0.392, 0.501))], [GREY], title="Mean reciprocal rank (68 test diseases)",
          maxv=0.62, point_colors=[GREY, GREY, GREY, ORANGE])
text(s, 0.6, 6.0, 12.1, 0.8, [{"t": "Gain over the best baseline: **+0.109 MRR** (95% CI +0.044 to +0.180, p = 0.003, "
                                    "3 seeds). Validation diseases agree: 0.394 vs 0.281 (p < 0.001). Median rank of a "
                                    "known drug: 2 of 7,957.", "size": 14}])
notes(s, "These 68 test diseases were never used for training or for choosing the stopping epoch.")

# 12. Results 2: transfer
s = S[11]
frame(s, "Results and Discussion: Source of Transfer (Aim 1)")
picture(s, "fig_transfer_curve.png", 0.55, 1.9, w=8.0)
text(s, 0.6, 5.1, 8.0, 1.6, [{"t": "MRR on 12 held-out autoimmune diseases (left) and the 6 rare targets (right) by "
                                   "training data. A' = 11 non-autoimmune diseases matched on number of labels.",
                              "size": 12, "color": MUTED}])
card(s, 8.85, 1.9, 3.9, 2.35)
text(s, 9.0, 1.98, 3.6, 2.2, [{"t": "Scarce data", "size": 16, "bold": True, "color": RUST, "space": 4},
                              {"t": "RA/SLE/MS: **0.290** vs matched controls 0.126 (p = 0.016).", "size": 13},
                              {"t": "But paths alone: 0.007 vs 0.010, so the gain is a drug-class prior.", "size": 13}])
card(s, 8.85, 4.4, 3.9, 2.35)
text(s, 9.0, 4.48, 3.6, 2.2, [{"t": "Full scale", "size": 16, "bold": True, "color": RUST, "space": 4},
                              {"t": "Adding autoimmune data: no gain (0.452 vs 0.532, p = 0.23).", "size": 13},
                              {"t": "Rare targets unchanged without them (0.487 vs 0.475).", "size": 13}])
notes(s, "Answer to Aim 1: related autoimmune diseases help only when data is scarce; at scale the broad graph supplies it.")

# 13. Results 3: components
s = S[12]
frame(s, "Results and Discussion: Component Analysis (Aim 3)")
bar_chart(s, 0.5, 1.9, 7.4, 4.3, ["Disease kNN", "Prior only", "BCE loss", "Full model", "BRG subset", "Paths only"],
          [("Validation MRR", (0.281, 0.303, 0.347, 0.394, 0.397, 0.405))], [GREY],
          title="Validation MRR (68 diseases)", maxv=0.48,
          point_colors=[GREY, GREY, GREY, ORANGE, ORANGE, ORANGE], label_size=12)
text(s, 8.2, 1.95, 4.55, 4.8, [
    {"t": "**Path reasoning carries the accuracy**: without it MRR drops to 0.303 (p < 0.001).", "bullet": True, "size": 14,
     "space": 9},
    {"t": "**Transfer channel = stability**: mean unchanged (p = 0.15), target sd 0.087 → 0.017 across seeds.",
     "bullet": True, "size": 14, "space": 9},
    {"t": "**Listwise loss beats BCE**: 0.394 vs 0.347 (p = 0.005).", "bullet": True, "size": 14, "space": 9},
    {"t": "**72% of edges are redundant**: 4.0M → 1.1M edges, same MRR, 11% faster.", "bullet": True, "size": 14}])
notes(s, "Honest reading: the ++ adds stability, not accuracy. Paths-only versus full model is a statistical tie.")

# 14. Results 4: TxGNN
s = S[13]
frame(s, "Results and Discussion: Head-to-Head with TxGNN")
bar_chart(s, 0.5, 1.9, 6.1, 3.7, ["MRR", "Hits@100", "AUROC"],
          [("TxGNN", (0.048, 0.232, 0.753)), ("BioPathNet++", (0.273, 0.618, 0.936))], [SLATE, ORANGE],
          title="Ranking all 7,957 drugs: significant", maxv=1.15, legend=True)
bar_chart(s, 6.7, 1.9, 6.1, 3.7, ["Recall@100", "AUPRC", "AUROC"],
          [("TxGNN", (0.536, 0.161, 0.901)), ("BioPathNet++", (0.614, 0.343, 0.777))], [SLATE, ORANGE],
          title="TxGNN's protocol (1,798 drugs): not significant", maxv=1.15, legend=True)
text(s, 0.6, 5.7, 12.1, 1.1, [
    {"t": "TxGNN's own split and code, 13 test diseases, 3 seeds each; our model trained only on edges TxGNN also "
          "trains on. Why the protocols disagree: only **23%** of TxGNN's top-100 drugs have any indication (ours "
          "**99%**); TxGNN's protocol filters those drugs out.", "size": 13}])
notes(s, "Our re-implementation of TxGNN's evaluation reproduces TxGNN's own numbers exactly. One split, 13 diseases: "
         "a first data point, not a final ranking.")

# 15. Results 5: case studies
s = S[14]
frame(s, "Results and Discussion: Case Studies (Top 5 per Target)")
top5 = [["Rank", "PBC", "SSc", "Sjögren", "AIH", "MG", "Crohn"],
        ["1", "Ursodeoxycholic acid★", "Benzoic acid★", "Propylene glycol★", "Prednisolone", "Ambenonium★", "Hydrocortisone★"],
        ["2", "Cholic acid", "Aminobenzoic acid★", "Lifitegrast", "Triamcinolone", "Neostigmine★", "Cortisone acetate★"],
        ["3", "Doripenem", "Prednisolone [B]", "Povidone★", "Hydrocortisone acetate", "Pyridostigmine★", "Triamcinolone★"],
        ["4", "Homatropine", "Triamcinolone", "Povidone K30★", "Sulfacetamide", "Distigmine★", "Prednisolone★"],
        ["5", "Ampicillin", "Betamethasone", "Prednisolone★", "Hydrocortisone", "Prednisolone★", "Ofloxacin"]]
table(s, 0.5, 1.95, 12.3, top5, [0.65, 2.15, 1.9, 1.9, 2.0, 1.75, 1.95], size=11.5, row_h=0.45)
text(s, 0.55, 4.8, 12.2, 0.4, [{"t": "★ known in PrimeKG;  [A]/[B] phase 3-4 / phase 2 trial that tests the drug in the disease;  rank averaged over 3 seeds",
                                "size": 12, "color": MUTED}])
for i, (h, b) in enumerate([("PBC", "First-line drug ursodeoxycholic acid ranked #1 in every seed."),
                            ("MG", "4 cholinesterase inhibitors at #1-4; amifampridine (#7) has a phase-3 MG trial."),
                            ("AIH", "Label gap: model's #1 prednisolone is standard therapy, but PrimeKG lists one other drug.")]):
    x = 0.55 + i * 4.1
    card(s, x, 5.3, 3.9, 1.45)
    text(s, x + 0.12, 5.35, 3.65, 1.35, [{"t": h, "size": 14, "bold": True, "color": RUST, "space": 2},
                                         {"t": b, "size": 12}])
notes(s, "Full top-10 lists with trial grades are in the report (Tables 8.7 and 8.8). Corticosteroids dominate: "
         "a known popularity bias we report.")

# 16. Results 6: validation
s = S[15]
frame(s, "Results and Discussion: Validation of New Candidates")
bar_chart(s, 0.5, 1.9, 7.6, 4.2, ["Trial tests the drug", "PubMed: benefit", "PubMed: harm / no benefit"],
          [("BioPathNet++", (13, 15, 5)), ("Popularity", (8, 10, 12)), ("Random", (0, 0, 0))],
          [ORANGE, GREY, RGBColor(0xD9, 0xD9, 0xD9)], title="Unlabelled top-20 candidates, % (6 targets)",
          fmt='0"%"', maxv=30, legend=True)
text(s, 8.4, 1.95, 4.35, 4.8, [
    {"t": "Temporal check", "size": 16, "bold": True, "color": RUST, "space": 4},
    {"t": "6 drugs approved after the data snapshot rank at a median of **1,104 / 7,957** (top 14%); "
          "risankizumab for Crohn ranks **#37**. Not significant (p = 0.11).", "size": 13, "space": 12},
    {"t": "Honest reading", "size": 16, "bold": True, "color": RUST, "space": 4},
    {"t": "Candidates are plausible (13% have a trial testing the drug vs 0% for random drugs) but only modestly "
          "ahead of popular drugs. Clinician review is the next step.", "size": 13}])
notes(s, "Trials count only if they test the drug (title or some-but-not-all arms), so background steroids are "
         "excluded. Literature labels were AI-assisted and checked against abstracts; a clinician has not reviewed them.")

# 17. Applicability
s = S[16]
frame(s, "Applicability to Other Domains")
table(s, 0.5, 1.95, 12.3, [["Domain", "Common problem", "What changes", "Data source"],
                           ["Other rare disease areas", "few known drugs per disease", "only the held-out set", "PrimeKG"],
                           ["Drug side effects", "predict unseen drug-effect links", "query relation", "SIDER / PrimeKG"],
                           ["Drug-target discovery", "new targets for a drug", "source = drug node", "DrugBank / PrimeKG"],
                           ["Clinical decision support", "rank options with reasons", "add path explanations UI", "hospital KG"],
                           ["Auditing other KG models", "leakage in evaluation", "reuse protocol + baselines", "any KG"]],
      [2.9, 3.4, 3.1, 2.9], size=13, row_h=0.5)
text(s, 0.55, 5.1, 12.2, 1.6, [
    {"t": "**Benefits:** inductive (works for unseen nodes), explainable, fits on free GPUs.  **Feasibility:** only the "
          "query relation and split change; code is relation-agnostic.  **Limits:** needs a curated knowledge graph; "
          "incomplete labels lower measured accuracy.", "size": 14}])
notes(s, "The method is general link prediction over a knowledge graph; the strict-family protocol is reusable anywhere.")

# 18. Timeline
s = S[17]
frame(s, "Timeline: Update on Tasks")
picture(s, "diag_timeline.png", 0.5, 1.95, w=8.2)
card(s, 8.95, 1.95, 3.8, 3.7, fill=TINT2)
text(s, 9.1, 2.05, 3.5, 3.5, [
    {"t": "Status", "size": 16, "bold": True, "color": RUST, "space": 8},
    {"t": "All planned experiments complete (7 Kaggle sessions)", "bullet": True, "size": 13},
    {"t": "Project report complete (56 pages, Phase-3 format)", "bullet": True, "size": 13},
    {"t": "IEEE conference paper: final draft", "bullet": True, "size": 13},
    {"t": "Journal manuscript: complete draft", "bullet": True, "size": 13},
    {"t": "Pending: clinician review of candidates", "bullet": True, "size": 13}])
notes(s, "Show the report to the panel here.")

# 19. Conclusion
s = S[18]
frame(s, "Conclusion and Future Work")
for i, (h, b) in enumerate([("Aim 2: accuracy", "Beats every baseline on unseen diseases: 76% vs 57% approved drug in top 10."),
                            ("Aim 1: transfer", "Autoimmune data helps only when data is scarce; at scale the broad graph suffices."),
                            ("Aim 3: components", "Paths carry accuracy; transfer channel adds stability; 72% of edges redundant.")]):
    x = 0.55 + i * 4.1
    card(s, x, 1.95, 3.9, 1.75)
    text(s, x + 0.15, 2.02, 3.6, 1.65, [{"t": h, "size": 16, "bold": True, "color": RUST, "space": 6}, {"t": b, "size": 13}])
text(s, 0.6, 4.0, 12.1, 2.4, [
    {"t": "Future work", "size": 16, "bold": True, "color": RUST, "space": 6},
    {"t": "Clinician review of top candidates with the prepared review sheet", "bullet": True, "size": 14},
    {"t": "KG-embedding baselines and BioPathNet's official code on the same split", "bullet": True, "size": 14},
    {"t": "TxGNN comparison on more disease areas; reduce the corticosteroid bias", "bullet": True, "size": 14}])
notes(s, "One line to close: rigorous evaluation turned an impressive but invalid result into a modest, trustworthy one.")

# 20. IEEE draft + venues
s = S[19]
frame(s, "IEEE Draft and Target Venues")
card(s, 0.6, 1.95, 5.9, 3.6, fill=TINT2)
text(s, 0.8, 2.05, 5.5, 3.4, [
    {"t": "Status of the drafts", "size": 16, "bold": True, "color": RUST, "space": 6},
    {"t": "Conference paper (IEEE, 5 pages): final draft", "bullet": True, "size": 14},
    {"t": "Journal manuscript (IEEE, about 11 pages): complete draft", "bullet": True, "size": 14},
    {"t": "College Turnitin check: 0% similarity; AI score below the reporting threshold", "bullet": True, "size": 14},
    {"t": "Every number generated from run logs", "bullet": True, "size": 14},
    {"t": "Title: BioPathNet++ for Autoimmune Diseases: Path-Based Transfer Learning for Rare Disease Drug Repurposing",
     "size": 12, "color": MUTED}])
card(s, 6.8, 1.95, 5.9, 3.6)
text(s, 7.0, 2.05, 5.5, 3.4, [
    {"t": "Targeted venues (CFPs to be confirmed)", "size": 16, "bold": True, "color": RUST, "space": 6},
    {"t": "Conferences: IEEE CONECCT, IEEE INDICON, ICCCNT, InCoB 2027", "bullet": True, "size": 14},
    {"t": "Journals: Computers in Biology and Medicine, Journal of Biomedical Informatics, BMC Bioinformatics, "
          "IEEE Access, Scientific Reports", "bullet": True, "size": 14}])
notes(s, "Venue deadlines and indexing must be checked before submission.")

# 21. References
s = S[20]
frame(s, "References")
refs = [
    "[1] E. Y. Hu et al., \"Enhancing link prediction in biomedical knowledge graphs with BioPathNet,\" Nature Biomedical Engineering, vol. 10, pp. 1747-1769, 2026.",
    "[2] K. Huang et al., \"A foundation model for clinician-centered drug repurposing,\" Nature Medicine, vol. 30, pp. 3601-3613, 2024.",
    "[3] P. Chandak, K. Huang and M. Zitnik, \"Building a knowledge graph to enable precision medicine,\" Scientific Data, vol. 10, p. 67, 2023.",
    "[4] Z. Zhu, Z. Zhang, L.-P. Xhonneux and J. Tang, \"Neural Bellman-Ford networks: A general graph neural network framework for link prediction,\" in NeurIPS, 2021.",
    "[5] S. Kapoor and A. Narayanan, \"Leakage and the reproducibility crisis in machine-learning-based science,\" Patterns, vol. 4, p. 100804, 2023.",
    "[6] G. Brière, T. Stosskopf, B. Loire and A. Baudot, \"Benchmarking the impact of data leakage on the performance of knowledge graph embedding models for biomedical link prediction,\" Bioinformatics, vol. 42, p. btag608, 2026.",
    "[7] D. S. Himmelstein et al., \"Systematic integration of biomedical knowledge prioritizes drugs for repurposing,\" eLife, vol. 6, p. e26726, 2017.",
    "[8] S. Pushpakom et al., \"Drug repurposing: progress, challenges and recommendations,\" Nature Reviews Drug Discovery, vol. 18, pp. 41-58, 2019."]
text(s, 0.6, 1.95, 12.1, 4.8, [{"t": r, "size": 13, "space": 7} for r in refs])
notes(s, "The full list of 34 references is in the report.")

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")

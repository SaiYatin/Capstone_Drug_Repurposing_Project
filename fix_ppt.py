"""Fix slide order and update remaining slides."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

SRC = r'C:\D_Games\capstone\ppts\sem 6 esa\UE23CS320B_Phase2_ESA_FILLED.pptx'
DST = r'C:\D_Games\capstone\ppts\sem 6 esa\UE23CS320B_Phase2_ESA_FINAL.pptx'

PROJ_TITLE = 'BioPathNet++ for Rare Autoimmune Drug Repurposing'
NAMES = 'TVNS Yatin, A Charan, Srijan, Praveen'

prs = Presentation(SRC)

def set_text(shape, lines, font_size=10, font_name='Trebuchet MS', bold=None):
    tf = shape.text_frame
    tf.word_wrap = True
    for i in range(len(tf.paragraphs) - 1, 0, -1):
        p = tf.paragraphs[i]._p
        p.getparent().remove(p)
    first = True
    for line in lines:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        for r in list(para.runs):
            r._r.getparent().remove(r._r)
        run = para.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.name = font_name
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        if bold is not None:
            run.font.bold = bold

# Print current state
print("CURRENT SLIDE ORDER:")
for i, sl in enumerate(prs.slides):
    texts = []
    for s in sl.shapes:
        if s.has_text_frame and s.text.strip():
            t = s.text.strip()[:60]
            if len(t) > 5:
                texts.append(t)
    print(f"  {i+1}: {texts[:2]}")

# Fix slides 2, 3, 4 content by checking what's actually there
for i, sl in enumerate(prs.slides):
    for shape in sl.shapes:
        if not shape.has_text_frame:
            continue
        t = shape.text.strip()
        
        # Fix ALL "Title of the Project" placeholders
        if t == 'Title of the Project':
            set_text(shape, [PROJ_TITLE], font_size=12, font_name='Arial', bold=False)
        
        # Fix ALL "NAME1_NAME2..." placeholders  
        if 'NAME1_NAME2' in t:
            set_text(shape, [NAMES], font_size=11, font_name='Arial')
        
        # Slide 3 (Abstract): "Well defined problem statement..."
        if 'Well defined problem statement' in t:
            set_text(shape, [
                "Problem: Over 95% of rare autoimmune patients lack approved therapies. "
                "Drug repurposing via knowledge graphs fails due to extreme data sparsity.",
                "",
                "Scope: BioPathNet++ addresses this by combining:",
                "  • Knowledge Graph enrichment (DRKG + Orphanet + HPO)",
                "  • Neural Bellman-Ford path-based reasoning with immune metapaths",
                "  • Multi-disease transfer learning (4 train → 2 test diseases, zero-shot)",
                "  • Five-layer validation (benchmark, zero-shot, few-shot, literature, pathway)",
                "",
                "Phase 2 Deliverables:",
                "  • Trained model achieving 0.935 AUPRC (beating TxGNN's 0.920)",
                "  • Validated drug predictions for PBC & Systemic Sclerosis",
                "  • Top-4 PBC drugs are all clinically validated treatments",
                "  • 78.6% literature support, 75% immune pathway match",
            ], font_size=10)
        
        # Slide 4 (Suggestions): Just leave it empty
        if 'Provide the suggestions and remarks' in t:
            set_text(shape, [
                "No specific suggestions were provided by the panel during Phase 2 review."
            ], font_size=14)
        
        # Slide 5 (Architecture): Fix the content text
        if 'Architecture of the overall project' in t or 'Explain the architecture' in t:
            set_text(shape, [
                "Four-phase pipeline:",
                "Phase 1: KG Construction — DRKG + Orphanet + HPO → 84,712 nodes, 1.9M edges",
                "Phase 2: Joint Training — NBFNet on RA, SLE, MS, Crohn's with 7 immune metapaths",
                "Phase 3: Transfer Learning — Zero-shot to PBC & Systemic Sclerosis (no retraining)",  
                "Phase 4: 5-Layer Validation — Benchmark, zero-shot, few-shot, literature, pathway",
            ], font_size=10)
        
        # Fix Slide 7 contributions if still template text
        if 'Contribution of individual member of the team for Phase-2 ESA' in t:
            set_text(shape, [
                "TVNS Yatin:",
                "  • Designed & implemented complete BioPathNet++ architecture (NBFNet + metapath attention)",
                "  • Built disease-centric 3-hop inference pipeline and KG enrichment module",
                "  • Engineered hard negative sampling with RepoDB clinical trial data",
                "  • Developed automated 5-layer validation framework",
                "",
                "Srijan:",
                "  • Authored Phase 2 research report and IEEE conference paper",
                "  • Implemented literature validation module (PubMed API integration)",
                "  • Data exploration and disease coverage analysis",
                "  • Designed publication-ready visualization pipeline",
                "",
                "A Charan:",
                "  • Built knowledge graph construction and entity mapping module",
                "  • Developed Entrez Gene ID ↔ Symbol mapping for pathway analysis",
                "  • Implemented drug name resolution with PubChem API fallback",
                "  • Conducted immune pathway analysis across target diseases",
                "",
                "Praveen:",
                "  • Curated immune gene lists for 7 pathway categories (91 genes)",  
                "  • Implemented few-shot fine-tuning evaluation with balanced sampling",
                "  • Managed dataset preprocessing and train/val/test split strategy",
                "  • RepoDB negative sampling and data cleaning pipeline",
            ], font_size=9)

# --- Reorder slides: move 12,13 to after slide 6 ---
# python-pptx: manipulate the sldIdLst XML
from lxml import etree

# Access the presentation XML element
pres_elem = prs.part._element
ns = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
sldIdLst = pres_elem.find(f'{ns}sldIdLst')
sldIds = list(sldIdLst)  # all <sldId> elements

print(f"\nTotal slide elements: {len(sldIds)}")

# Current order: 1,2,3,4,5,6,7,8,9,10,11,12,13
# Want:          1,2,3,4,5,6,12,13,7,8,9,10,11
# That means: move element at index 11 (slide 12) to after index 5 (slide 6)
#             move element at index 12 (slide 13) to after the moved slide 12

if len(sldIds) == 13:
    # Remove slides 12 and 13 (indices 11, 12 in 0-indexed)
    slide12 = sldIds[11]  # Drug Rankings
    slide13 = sldIds[12]  # Validation Summary
    
    sldIdLst.remove(slide12)
    sldIdLst.remove(slide13)
    
    # Re-get the list after removal
    sldIds = list(sldIdLst)
    # Now we have 11 slides, insert after index 5 (slide 6 = Results SOTA)
    # sldIds[5] is slide 6 (Results)
    ref_element = sldIds[5]  # slide 6
    
    # Insert slide12 (drug rankings) after slide 6
    ref_element.addnext(slide12)
    # Insert slide13 (validation) after slide12
    slide12.addnext(slide13)
    
    print("Slides reordered: Drug Rankings and Validation moved after Results")
else:
    print(f"WARNING: Expected 13 slides, got {len(sldIds)}. Skipping reorder.")

# Verify final order
print("\nFINAL SLIDE ORDER:")
for i, sl in enumerate(prs.slides):
    texts = []
    for s in sl.shapes:
        if s.has_text_frame:
            t = s.text.strip()[:50]
            if len(t) > 5:
                texts.append(t)
    print(f"  {i+1}: {texts[:2]}")

prs.save(DST)
print(f"\nSaved to: {DST}")

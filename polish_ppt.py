"""Final polish - fix remaining template placeholders."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

SRC = r'C:\D_Games\capstone\ppts\sem 6 esa\UE23CS320B_Phase2_ESA_FINAL.pptx'
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

# Detailed check of every shape on every slide
for i, sl in enumerate(prs.slides):
    print(f"\n=== SLIDE {i+1} ===")
    for j, shape in enumerate(sl.shapes):
        if shape.has_text_frame:
            t = shape.text.strip()
            if not t:
                continue
            
            # --- SLIDE 1: Title page fields ---
            if 'Project Title' in t and ':' in t and 'Project ID' in t:
                print(f"  Fixing Slide {i+1} project info")
                set_text(shape, [
                    "Project Title   :  BioPathNet++ for Rare Autoimmune Drug Repurposing",
                    "Project ID       :      UE23CS320B",
                    "Project Guide :      Dr. Priyanka H",
                ], font_size=14, font_name='Arial')
                continue
            
            # --- Outline slide (slide 2) ---
            if 'Abstract and scope' in t and 'Suggestions' in t and 'Architecture' in t:
                print(f"  Fixing Slide {i+1} outline")
                set_text(shape, [
                    "1. Abstract and Scope",
                    "2. Suggestions from Phase 2 Review",
                    "3. Implementation Details",
                    "4. Results: Benchmark Performance",
                    "5. Results: Drug Predictions",
                    "6. Results: Validation Framework",
                    "7. Individual Contribution",
                    "8. Phase 3 & 4 Progress Plan",
                    "9. References",
                ], font_size=16, font_name='Trebuchet MS')
                continue
            
            # Fix drug ranking slides - the body text still says old content
            if 'Test AUPRC: 0.935' in t and i == 6:  # Slide 7 (Drug Rankings)
                print(f"  Fixing Slide {i+1} drug rankings body")
                set_text(shape, [
                    "PBC: Top-4 predictions (Thalidomide, UDCA, Fenofibrate, Prednisolone)",
                    "    are ALL clinically validated treatments — identified via zero-shot transfer.",
                    "Systemic Sclerosis: Drug predictions span TNF, JAK-STAT, and cytokine pathways.",
                ], font_size=11, font_name='Trebuchet MS')
                continue
            
            if 'Test AUPRC: 0.935' in t and i == 7:  # Slide 8 (Validation)
                print(f"  Fixing Slide {i+1} validation body")
                set_text(shape, [
                    "Layer 1: Benchmark AUPRC 0.935 beats TxGNN (0.920) ✓",
                    "Layer 2: Zero-shot PBC 0.949 / SS 0.871 — no retraining needed ✓",
                    "Layer 3: Few-shot fine-tuning shows minimal change (Δ < 0.006) ✓",
                    "Layer 4: Literature validation 78.6% support (≥70% target) ✓",
                    "Layer 5: Immune pathway match 75% (PBC 95%, SS 55%)",
                ], font_size=11, font_name='Trebuchet MS')
                continue
            
            short = t[:60]
            print(f"  Shape {j}: '{short}'")

# Check which slides have images
print("\n\n=== IMAGES ON SLIDES ===")
for i, sl in enumerate(prs.slides):
    imgs = [s for s in sl.shapes if s.shape_type == 13]  # PICTURE type
    if imgs:
        for img in imgs:
            print(f"  Slide {i+1}: Image at ({img.left}, {img.top}), size={img.width}x{img.height}")

prs.save(DST)
print(f"\nFinal PPT saved to: {DST}")

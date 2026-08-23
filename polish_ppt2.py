"""Final cleanup - fix outline, remove wrong architecture image from Suggestions, 
   remove duplicate body images from slides 7/8, fix slide numbers."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor

SRC = r'C:\D_Games\capstone\ppts\sem 6 esa\UE23CS320B_Phase2_ESA_FINAL.pptx'
DST = r'C:\D_Games\capstone\ppts\sem 6 esa\UE23CS320B_Phase2_ESA_FINAL.pptx'

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

# --- Fix Slide 2: Outline ---
slide = prs.slides[1]
for shape in slide.shapes:
    if shape.has_text_frame:
        t = shape.text.strip()
        if 'Abstract and scope' in t and 'Suggestions' in t:
            print("Fixing Outline slide text")
            set_text(shape, [
                "1. Abstract and Scope",
                "2. Suggestions from Phase 2 Review",
                "3. Implementation Details", 
                "4. Results: Benchmark Performance",
                "5. Results: Drug Predictions (Zero-Shot)",
                "6. Results: Five-Layer Validation",
                "7. Individual Contribution",
                "8. Phase 3 & 4 Progress Plan",
                "9. References",
            ], font_size=16, font_name='Trebuchet MS')

# --- Fix Slide 4: Remove the architecture image that was incorrectly placed ---
slide = prs.slides[3]  # Suggestions slide
shapes_to_remove = []
for shape in slide.shapes:
    # Remove any large images (the architecture image ~8.2M EMU wide)
    if shape.shape_type == 13 and shape.width > Inches(5):
        shapes_to_remove.append(shape)
        print(f"Removing architecture image from Suggestions slide (size: {shape.width})")

for shape in shapes_to_remove:
    sp = shape._element
    sp.getparent().remove(sp)

# --- Fix Slides 7 and 8: Remove the duplicated images from template copy ---
# Slide 7 (Drug Rankings) has 5 images - should only have 3 (logo + 2 drug charts)
# Slide 8 (Validation) has 4 images - should only have 2 (logo + validation chart)

for slide_idx in [6, 7]:  # slides 7 and 8 (0-indexed)
    slide = prs.slides[slide_idx]
    images = [(j, shape) for j, shape in enumerate(slide.shapes) if shape.shape_type == 13]
    
    if slide_idx == 6:  # Drug rankings: keep logo + 2 drug ranking images (last 2)
        # Images from template duplication: logo, sota chart, per-disease chart (from results slide)
        # Plus the 2 new drug ranking images
        # Remove the inherited SOTA and per-disease charts
        to_remove = []
        for j, img in images:
            # Logo images are < 3 inches wide, keep them
            if img.width < Inches(3):
                continue
            # The drug ranking images were placed at (274320, 2286000) — keep these
            # The inherited charts were at (274320, 2377440) and (5029200, 2377440)
            if img.top < Emu(2300000):  # if top is less than ~2.3M EMU, it's inherited
                continue
            if img.top == Emu(2377440):  # inherited from results slide
                to_remove.append(img)
                print(f"Removing inherited chart from Slide {slide_idx+1}")
        for img in to_remove:
            sp = img._element
            sp.getparent().remove(sp)
    
    elif slide_idx == 7:  # Validation: keep logo + validation chart (last 1)
        to_remove = []
        for j, img in images:
            if img.width < Inches(3):
                continue
            # The validation chart was placed at (914400, 2926080)
            # The inherited ones are at (274320, 2377440) and (5029200, 2377440)
            if img.top == Emu(2377440):
                to_remove.append(img)
                print(f"Removing inherited chart from Slide {slide_idx+1}")
        for img in to_remove:
            sp = img._element
            sp.getparent().remove(sp)

# --- Fix page numbers for slides 7-13 ---
page_numbers = {6: '7', 7: '8', 8: '9', 9: '10', 10: '11', 11: '12', 12: '13'}
for idx, num in page_numbers.items():
    slide = prs.slides[idx]
    for shape in slide.shapes:
        if shape.has_text_frame:
            t = shape.text.strip()
            if t in ['6', '7', '8', '9', '10', '11']:
                # Only update if it's a small shape (page number area)
                if shape.width < Inches(2):
                    set_text(shape, [num], font_size=11, font_name='Arial')

prs.save(DST)
print(f"\nFinal PPT saved: {DST}")

# Final verification
print("\nFINAL SLIDE SUMMARY:")
for i, sl in enumerate(prs.slides):
    imgs = len([s for s in sl.shapes if s.shape_type == 13])
    titles = []
    for s in sl.shapes:
        if s.has_text_frame:
            t = s.text.strip()[:40]
            if len(t) > 3 and t not in ['TVNS Yatin, A Charan, Srijan, Praveen',
                                         'BioPathNet++ for Rare Autoimmune Drug R']:
                titles.append(t)
    print(f"  Slide {i+1}: {titles[:2]} ({imgs} images)")

"""Add architecture image to Implementation slide and final page number fixes."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pptx import Presentation
from pptx.util import Inches, Pt

SRC = r'C:\D_Games\capstone\ppts\sem 6 esa\UE23CS320B_Phase2_ESA_FINAL.pptx'
DST = r'C:\D_Games\capstone\ppts\sem 6 esa\UE23CS320B_Phase2_ESA_FINAL.pptx'
VIZ = r'C:\D_Games\capstone\implementations\outputs\visualizations'

prs = Presentation(SRC)

# Add architecture image to Slide 5 (Implementation Details)
slide5 = prs.slides[4]
# Shrink the text box first
for shape in slide5.shapes:
    if shape.has_text_frame and 'Module 1' in shape.text:
        # Make text box shorter to fit architecture below
        shape.height = Inches(2.5)  # was ~2.8 inches
        print(f"Shrunk implementation text box to {shape.height}")

# Add architecture image below the text
slide5.shapes.add_picture(
    f'{VIZ}/biopathnet_architecture_v3.png',
    left=Inches(0.8), top=Inches(4.5),
    width=Inches(8.5)
)
print("Added architecture image to Implementation slide")

prs.save(DST)
print(f"\nSaved: {DST}")

# Final count
print("\n=== FINAL PPT STRUCTURE ===")
for i, sl in enumerate(prs.slides):
    imgs = len([s for s in sl.shapes if s.shape_type == 13])
    heading = ""
    for s in sl.shapes:
        if s.has_text_frame:
            t = s.text.strip()
            if len(t) > 3 and 'TVNS' not in t and 'BioPathNet' not in t and len(t) < 60:
                heading = t[:50]
                break
    print(f"  Slide {i+1:2d}: {heading:<50s} | {imgs} img(s)")

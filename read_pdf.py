import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import fitz  # pymupdf

doc = fitz.open(r'C:\D_Games\capstone\Report\final paper\CAPSTONE-11.pdf')
print(f"Total pages: {doc.page_count}\n")

for page_num in range(doc.page_count):
    page = doc[page_num]
    text = page.get_text()
    print(f"\n{'='*80}")
    print(f"PAGE {page_num + 1}")
    print(f"{'='*80}")
    print(text[:3000])
    if len(text) > 3000:
        print(f"\n... [truncated, {len(text)} chars total]")

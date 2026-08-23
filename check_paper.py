import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import fitz

doc = fitz.open(r'C:\D_Games\capstone\Report\final paper\CAPSTONE-11.pdf')
text = ''.join([p.get_text() for p in doc])

print("=== OTHER DISEASE MENTIONS ===")
for d in ['Sjogren', 'Myasthenia', 'Autoimmune Hepatitis', "Crohn"]:
    count = text.count(d)
    print(f"  {d}: {count} mentions")
    if count > 0:
        # find context
        idx = 0
        while True:
            idx = text.find(d, idx)
            if idx == -1:
                break
            snippet = text[max(0,idx-50):idx+len(d)+50].replace('\n',' ')
            print(f"    -> ...{snippet}...")
            idx += len(d)

print("\n=== KEY PHRASES ===")
for phrase in ['five rare', 'five target', 'six rare', 'six target', 'two target', 'two rare', 'all diseases', 'remaining diseases']:
    count = text.lower().count(phrase)
    print(f"  '{phrase}': {count} mentions")
    if count > 0:
        idx = 0
        while True:
            idx = text.lower().find(phrase, idx)
            if idx == -1:
                break
            snippet = text[max(0,idx-60):idx+len(phrase)+60].replace('\n',' ')
            print(f"    -> ...{snippet}...")
            idx += len(phrase)

print("\n=== TRAIN DISEASE MENTIONS ===")
for d in ['Crohn', 'RA,', 'SLE,', 'MS,']:
    count = text.count(d)
    print(f"  {d}: {count} mentions")

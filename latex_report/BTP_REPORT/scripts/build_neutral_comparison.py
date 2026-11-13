"""Create black comparison labels and row frames without changing saved pixels.

Requires PyMuPDF. The original, immutable weather_examples.pdf is preserved.
Only vector drawing/text colours change; embedded image streams are checked.
"""
from pathlib import Path
import hashlib,json,re
import pymupdf as pdf
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'figures/weather_examples.pdf'
out=ROOT/'figures/weather_examples_mono.pdf'
doc=pdf.open(source);page=doc[0]
pattern=rb'(?<![\w.])([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(rg|RG)\b'
def monochrome(m):
    rgb=list(map(float,m.group(1,2,3)))
    return (b'0 g' if m[4]==b'rg' else b'0 G') if max(rgb)-min(rgb)>1e-6 else m[0]
page.clean_contents()
for xref in page.get_contents():
    doc.update_stream(xref,re.sub(pattern,monochrome,doc.xref_stream(xref)))
for y in [20.88,75.96,131.04,186.12]:
    page.draw_rect(pdf.Rect(48.8,y-1.2,492.6,y+52.32),color=(.30,.30,.30),width=.35)
doc.set_metadata({**doc.metadata,'title':'Matched weather comparison: black headings','modDate':''})
doc.save(out,garbage=4,deflate=True)
modified=pdf.open(out)
def image_hashes(d):
    return sorted(hashlib.sha256(d.xref_stream(i[0])).hexdigest() for i in d[0].get_images())
assert image_hashes(doc)==image_hashes(modified)
spans=[s for b in modified[0].get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans']]
assert all(s['color']==0 for s in spans)
record={'source':'figures/weather_examples.pdf','output':'figures/weather_examples_mono.pdf','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'all_headers_black':True,'embedded_image_streams_unchanged':True,'image_count':len(doc[0].get_images()),'row_frames':'0.35 pt neutral rule; one group per weather','generator':'scripts/build_neutral_comparison.py'}
(ROOT/'evidence/neutral_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

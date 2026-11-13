"""Presentation-only derivatives of preserved measured charts (PyMuPDF).

Separate the detection axis title from its legend; correct a stale appendix
letter in the export chart. No plotted values or marks are changed.
"""
from pathlib import Path
import hashlib,json,re
import pymupdf as pdf
ROOT=Path(__file__).resolve().parents[1]
records=[]
def record(source,out,change):
    records.append({'source':str(source.relative_to(ROOT)),'output':str(out.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'change':change})
source=ROOT/'figures/dawn_operating_points.pdf';out=ROOT/'figures/dawn_operating_points_spaced.pdf'
doc=pdf.open(source);pg=doc[0];stream=pg.read_contents()
# The first four markers are the shared legend. Its text has a unique baseline.
stream,n=re.subn(rb'q 0 0 495 147\.6 re W n.*?\nB\nQ',lambda m:b'q 1 0 0 1 0 -10 cm\n'+m[0]+b'\nQ',stream,count=4,flags=re.S)
assert n==4 and stream.count(b'4.1903125 cm')==4
stream=stream.replace(b'4.1903125 cm',b'-5.8096875 cm')
pg.set_mediabox(pdf.Rect(0,0,495,159.6))
doc.update_stream(pg.get_contents()[0],b'q 1 0 0 1 0 12 cm\n'+stream+b'\nQ')
doc.save(out,garbage=4,deflate=True)
with pdf.open(source) as original,pdf.open(out) as revised:
    area=pdf.Rect(0,0,495,130)
    assert original[0].get_pixmap(clip=area,matrix=pdf.Matrix(2,2)).samples==revised[0].get_pixmap(clip=area,matrix=pdf.Matrix(2,2)).samples
    assert original[0].get_text().split()==revised[0].get_text().split()
record(source,out,'Add 12 pt below the chart and shift only the legend by 10 pt; plotted region pixel-equal at 144 dpi.')
source=ROOT/'figures/edge_acceptance.pdf';out=ROOT/'figures/edge_acceptance_current.pdf'
doc=pdf.open(source);pg=doc[0]
old='For the measured Nano engine, see Appendix D.'
boxes=pg.search_for(old);assert len(boxes)==1
box=boxes[0]
pg.add_redact_annot(box,fill=None)
pg.apply_redactions(images=0,graphics=0)
pg.insert_text((box.x0,box.y1-1.5),'For the measured Nano engine, see Appendix C.',fontname='helv',fontsize=7.1,color=(1/3,.3607843137,.4078431373))
doc.save(out,garbage=4,deflate=True)
with pdf.open(source) as original,pdf.open(out) as revised:
    assert 'Appendix C' in revised[0].get_text()
    area=pdf.Rect(0,0,495,130)
    assert original[0].get_pixmap(clip=area,matrix=pdf.Matrix(2,2)).samples==revised[0].get_pixmap(clip=area,matrix=pdf.Matrix(2,2)).samples
record(source,out,'Correct stale appendix letter D to C (Detection, Resolution Controls, and Nano Execution).')
(ROOT/'evidence/chart_layout_repairs.json').write_text(json.dumps({'generator':'scripts/build_chart_layout_repairs.py','figures':records},indent=2)+'\n')
print(json.dumps(records,indent=2))

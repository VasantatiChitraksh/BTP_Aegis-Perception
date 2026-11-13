#!/usr/bin/env python3
"""Restyle vector flowcharts; preserve topology, labels and quantitative panels.

Dependencies: matplotlib, PyMuPDF, fonttools, brotli, Pillow.
The bundled Virgil font is licensed under the SIL OFL (fonts/LICENSE.md).
"""
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import pymupdf as pdf

ROOT=Path(__file__).resolve().parents[1]
FONT=ROOT/'fonts/Virgil.ttf'
font_manager.fontManager.addfont(FONT)
FAMILY=font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({'font.family':[FAMILY,'DejaVu Sans'], 'pdf.fonttype':42,
                     'text.color':'#212529','savefig.facecolor':'white'})
INK='#343a40'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pastel(rgb):
    if rgb is None:return 'none'
    r,g,b=rgb
    if min(rgb)<.7:return INK
    if max(rgb)-min(rgb)<.025:return '#d0ebff'
    if g>r and g>b:return '#d3f9d8'
    if r>g and b>g:return '#e5dbff'
    if r>b and g>b:return '#ffec99'
    return '#d0ebff'

def draw_path(ax,d):
    verts=[];codes=[];last=None
    def move(p):
        nonlocal last
        if last!=tuple(p):verts.append(tuple(p));codes.append(MPath.MOVETO)
        last=tuple(p)
    for item in d['items']:
        kind=item[0]
        if kind=='re':
            r=item[1];ps=[r.tl,r.tr,r.br,r.bl,r.tl]
            verts.extend(map(tuple,ps));codes.extend([MPath.MOVETO]+[MPath.LINETO]*3+[MPath.CLOSEPOLY]);last=None
        elif kind=='l':
            move(item[1]);verts.append(tuple(item[2]));codes.append(MPath.LINETO);last=tuple(item[2])
        elif kind=='c':
            move(item[1]);verts.extend(map(tuple,item[2:]));codes.extend([MPath.CURVE4]*3);last=tuple(item[-1])
        elif kind=='qu':
            q=item[1];ps=[q.ul,q.ur,q.lr,q.ll,q.ul]
            verts.extend(map(tuple,ps));codes.extend([MPath.MOVETO]+[MPath.LINETO]*3+[MPath.CLOSEPOLY]);last=None
        else:raise ValueError(kind)
    if not verts:return
    if d.get('closePath') and codes[-1]!=MPath.CLOSEPOLY:
        verts.append(verts[0]);codes.append(MPath.CLOSEPOLY)
    is_box=any(i[0]=='re' for i in d['items']) and d['rect'].width>20 and d['rect'].height>10
    fill=pastel(d.get('fill'))
    # Filled arrowheads retain their dark ink; soft colour belongs to boxes only.
    if not is_box and d.get('fill') is not None:fill=INK
    patch=PathPatch(MPath(verts,codes),facecolor=fill,edgecolor=INK,
                    lw=.85 if is_box else .72,capstyle='round',joinstyle='round')
    dash=re.search(r'\[([^]]+)\]',d.get('dashes') or '')
    if dash:patch.set_linestyle((0,[float(x) for x in dash.group(1).split()]))
    patch.set_sketch_params(scale=.42 if is_box else .20,length=95,randomness=1.4)
    ax.add_patch(patch)

def restyle(source,target,region=None):
    original=pdf.open(source);page=original[0];whole=page.rect
    region=pdf.Rect(region) if region else whole
    fig=plt.figure(figsize=(whole.width/72,whole.height/72))
    ax=fig.add_axes((0,0,1,1));ax.set(xlim=(0,whole.width),ylim=(whole.height,0));ax.axis('off')
    drawings=page.get_drawings()
    for d in drawings:
        if d['rect'].width>=whole.width-.1 and d['rect'].height>=whole.height-.1:continue
        if region.contains(d['rect']):draw_path(ax,d)
    boxes=[d['rect'] for d in drawings if d['rect'].width>20 and d['rect'].height>10
           and any(i[0]=='re' for i in d['items']) and d['rect']!=whole]
    text_items=[]
    for block in page.get_text('dict')['blocks']:
        if block['type']!=0:continue
        for line in block['lines']:
            box=pdf.Rect(line['bbox'])
            if not region.contains(box):continue
            value=''.join(s['text'] for s in line['spans'])
            # Standard mathematical letters retain the same meaning in Virgil.
            value=unicodedata.normalize('NFKC',value)
            math_formula='sqin the' in value
            if math_formula:value=r'$\tilde{w}=sq$ in the'
            size=max(s['size'] for s in line['spans'])*1.07
            x=(box.x0+box.x1)/2;y=(box.y0+box.y1)/2
            parent=next((r for r in boxes if r.contains(box)),None)
            allowed=(parent.width-7) if parent else min(box.width*1.08,region.x1-box.x0-1)
            item=ax.text(x,y,value,fontsize=size,ha='center',va='center',parse_math=math_formula)
            text_items.append((item,allowed,box.height*1.05))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for item,width,height in text_items:
        bbox=item.get_window_extent(renderer)
        factor=min(1,width/(bbox.width*72/fig.dpi),height/(bbox.height*72/fig.dpi))
        item.set_fontsize(item.get_fontsize()*factor)
    target.parent.mkdir(parents=True,exist_ok=True)
    drawing=target.with_name(target.stem+'_drawing.pdf')
    fig.savefig(drawing,metadata={'CreationDate':None,'ModDate':None,'Title':'Aegis flowchart — sketch style'})
    plt.close(fig)
    styled=pdf.open(drawing)
    if region!=whole:
        # Remove only the flow region; preserve chart/table streams outside it.
        page.add_redact_annot(region,fill=(1,1,1))
        page.apply_redactions(images=0,graphics=2,text=0)
        page.show_pdf_page(region,styled,0,clip=region)
        original.save(target,garbage=4,deflate=True)
    else:styled.save(target,garbage=4,deflate=True)
    styled.close();original.close();drawing.unlink()
    return {'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),
            'output':str(target.relative_to(ROOT)),'output_sha256':sha(target),
            'styled_region_points':list(region),'dimensions_points':list(whole)}

def main():
    figures=ROOT/'figures'
    specs=[('project_evidence_flow',None),('benchmark_protocol',(0,128.5,234.72,148.32)),
           ('restorer_architectures',None),('controlled_resolution',(0,0,248,151.92))]
    records=[]
    for name,region in specs:
        records.append(restyle(figures/(name+'.pdf'),figures/(name+'_excalidraw.pdf'),region))
    records.append(restyle(figures/'style_sources/weight_grid_workflow.pdf',
                           figures/'weight_grid_workflow_excalidraw.pdf'))
    record={'scope':'Flowchart styling only; source topology and labels retained. Quantitative panels are preserved.',
            'font':{'file':'fonts/Virgil.ttf','sha256':sha(FONT),'source':'https://github.com/excalidraw/virgil','license':'fonts/LICENSE.md'},
            'figures':records,'generator':'scripts/build_sketch_flowcharts.py'}
    (ROOT/'evidence/sketch_flowcharts.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Built five vector flowcharts with original canvas dimensions.')

if __name__=='__main__':main()

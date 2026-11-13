#!/usr/bin/env python3
"""Compose the cover from supplied hardware photos and saved weather pairs.

This is a document diagram, not a new inference result. Hardware photographs are
embedded as supplied. The network and arrows are vector drawing primitives.
"""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
NAVY='#18394b';TEAL='#217c85'

def build(root,output):
    raw=root/'figures/cover_inputs'
    photo_paths=[raw/'rover_camera.png',raw/'jetson_connections.png']
    for p in photo_paths:
        if not p.is_file():raise FileNotFoundError('Original hardware photograph required: '+str(p))
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42})
    fig=plt.figure(figsize=(6.85,4.35))
    ax=fig.add_axes((0,0,1,1));ax.set(xlim=(0,6.85),ylim=(0,4.35),aspect='equal');ax.axis('off')
    fig.patch.set_facecolor('white')
    def photo(p,x,y,w,h):
        im=Image.open(p).convert('RGB')
        actual_h=w*im.height/im.width
        if actual_h>h:w=h*im.width/im.height;actual_h=h
        shown=ax.imshow(im,extent=(x,x+w,y,y+actual_h),zorder=2,interpolation='lanczos')
        ax.add_patch(Rectangle((x,y),w,actual_h,facecolor='none',edgecolor='#777777',lw=.45,zorder=3))
        return (x,y,w,actual_h)
    def weather(p,x,y,w,h):
        im=Image.open(p).convert('RGB')
        if p.name.startswith('smoke_'):im=im.crop((0,13,im.width,im.height))
        im=ImageOps.fit(im,(650,385),method=Image.Resampling.LANCZOS)
        ax.imshow(im,extent=(x,x+w,y,y+h),zorder=2)
        ax.add_patch(Rectangle((x,y),w,h,facecolor='none',edgecolor='#777777',lw=.45,zorder=3))
    def arrow(start,end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=12,
            lw=1.1,color='black',shrinkA=0,shrinkB=0,zorder=6))
    # Input examples lead to the upper camera on the supplied front-view photo.
    for i,w in enumerate(['fog','snow','rain','smoke']):
        weather(root/'figures'/f'{w}_input.png',.06+(i%2)*.94,3.00+(.60 if i<2 else 0),.88,.52)
    # Leave a visible white gutter above the output gallery; preserve aspect ratio.
    front=photo(photo_paths[0],2.36,2.00,1.59,2.12)
    x,y,w,h=front
    camera=(x+.36*w,y+.865*h)
    arrow((1.96,camera[1]),camera)
    # One outlined hardware-and-network module, with no internal flow arrow.
    module=FancyBboxPatch((4.63,.04),2.14,4.23,
        boxstyle='round,pad=0,rounding_size=.06',facecolor='white',
        edgecolor='#454545',lw=.95,zorder=1)
    module.set_sketch_params(scale=.30,length=120,randomness=1.3)
    ax.add_patch(module)
    board=photo(photo_paths[1],5.00,1.62,1.40,2.49)
    arrow((4.07,2.93),(4.60,2.93))
    # The network and shared caption belong to the same module as the board photo.
    xs=[4.95,5.70,6.45]
    ys=[[.58,.94,1.30],[.46,.78,1.10,1.42],[.58,.94,1.30]]
    for c in [0,1]:
        for y1 in ys[c]:
            for y2 in ys[c+1]:ax.plot([xs[c],xs[c+1]],[y1,y2],color='#9aaeb4',lw=.55,zorder=2)
    for c in range(3):
        for yy in ys[c]:ax.add_patch(Circle((xs[c],yy),.075,fc=TEAL if c==1 else NAVY,ec='white',lw=.5,zorder=4))
    ax.text(5.70,.13,'Aegis Perception module',ha='center',va='bottom',fontsize=9.4,color='black')
    # Four corresponding restored examples complete the flow; no chip symbol.
    for i,w in enumerate(['fog','snow','rain','smoke']):
        weather(root/'figures'/f'{w}_restormer.png',.75+(i%2)*1.40,.12+(.87 if i<2 else 0),1.30,.77)
    arrow((4.60,.94),(3.60,.94))
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,metadata={'Title':'Aegis Perception hardware flow','CreationDate':None,'ModDate':None})
    fig.savefig(output.with_suffix('.png'),dpi=210)
    plt.close(fig)
    record={'generator':'scripts/build_cover.py','label':'Aegis Perception module','flow':['four saved inputs','camera on the supplied rover photo','combined Jetson USB/board and neural-network module','four saved restored outputs'],'hardware_photos':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in photo_paths},'photo_edits':'None: original hardware files are embedded with proportional scaling.','scope':'Cover illustration only; saved weather pairs are representative examples, not new device experiments.'}
    (root/'evidence/cover_hardware.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path)
    args=p.parse_args();build(args.root,args.output or args.root/'figures/cover_pipeline.pdf')

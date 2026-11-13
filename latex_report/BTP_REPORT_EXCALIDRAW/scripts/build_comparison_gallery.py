"""Render saved matched-input comparisons; does not run restoration or detection.

Uses only the compact immutable copies in figures/comparison_inputs. Every image
and per-panel count is checked against its recorded SHA-256 before rendering.
Vehicle images share an identical display crop, recorded in provenance.json;
people images retain the full field of view. Detector overlays are not redrawn.
Run: python scripts/build_comparison_gallery.py
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
INPUTS=ROOT/'figures/comparison_inputs'
OUT=ROOT/'figures'
MANIFEST=json.loads((INPUTS/'provenance.json').read_text())
INK='#000000'; GRAY='#586976'; TEAL='#246967'; BLUE='#3B6599'; AMBER='#994E3A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'text.color':INK,'pdf.fonttype':42,'savefig.facecolor':'white'})
WEATHERS=['fog','rain','snow','smoke']
MODES=['direct','resize_128','restored_128']
HEADS=['YOLOv8n only\nSource resolution','YOLOv8n only\n128 × 128 resize control','Restormer + YOLOv8n\n128 × 128 restoration']
COLORS=[BLUE,GRAY,TEAL]

def check_inputs():
    for item in MANIFEST['inputs']:
        p=INPUTS/item['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
    for source in MANIFEST['sources']:
        for weather in WEATHERS:
            results=[json.loads((INPUTS/source/weather/m/'result.json').read_text()) for m in MODES]
            assert len({r['weather_source_sha256'] for r in results})==1
            common=INPUTS/source/weather/'weather_source.png'
            assert hashlib.sha256(common.read_bytes()).hexdigest()==results[0]['weather_source_sha256']
            assert all(r['input_shape']==[1,3,128,128] for r in results[1:])
            n=len(MANIFEST['sources'][source]['gt'])
            assert all(r['matches']['tp']+r['matches']['fn']==n for r in results)

def build(source,stem):
    crop=MANIFEST['rendering'][source]['crop_xyxy']
    width=6.85
    col_x=[.40,2.55,4.70]
    col_w=2.05
    dims=MANIFEST['sources'][source]['dimensions']
    image_ratio=(crop[3]-crop[1])/(crop[2]-crop[0]) if crop else dims[1]/dims[0]
    image_h=col_w*image_ratio
    row_h=image_h+.24
    height=.53+len(WEATHERS)*row_h+.08
    fig=plt.figure(figsize=(width,height))
    frame=fig.add_axes((0,0,1,1)); frame.set(xlim=(0,width),ylim=(0,height));frame.axis('off')
    for i,(head,color) in enumerate(zip(HEADS,COLORS)):
        frame.add_patch(Rectangle((col_x[i],height-.47),col_w,.44,facecolor='white',edgecolor='#777777',linewidth=.5))
        frame.text(col_x[i]+col_w/2,height-.245,head,ha='center',va='center',fontsize=8.3,weight='bold',color='black',linespacing=1.35)
    for j,weather in enumerate(WEATHERS):
        row_top=height-.54-j*row_h
        image_bottom=row_top-image_h
        frame.text(.17,image_bottom+image_h/2,weather.upper(),ha='center',va='center',fontsize=8.2,weight='bold',rotation=90)
        for i,mode in enumerate(MODES):
            folder=INPUTS/source/weather/mode
            result=json.loads((folder/'result.json').read_text())
            im=Image.open(folder/'detections.png').convert('RGB')
            if crop:im=im.crop(crop)
            ax=fig.add_axes((col_x[i]/width,image_bottom/height,col_w/width,image_h/height))
            ax.imshow(im,interpolation='nearest',aspect='equal');ax.axis('off')
            counts=result['matches']
            label=f"TP {counts['tp']}   FP {counts['fp']}   FN {counts['fn']}"
            frame.text(col_x[i]+col_w/2,image_bottom-.095,label,ha='center',va='center',fontsize=8.2,color=INK)
        frame.add_patch(Rectangle((.37,image_bottom-.19),6.41,image_h+.22,facecolor='none',edgecolor='#777777',linewidth=.5))
    fig.savefig(OUT/(stem+'.pdf'),metadata={'CreationDate':None,'ModDate':None,'Creator':'Aegis saved-overlay gallery builder'})
    fig.savefig(OUT/(stem+'.jpg'),dpi=190)
    plt.close(fig)
    return {'figure':stem+'.pdf','height_inches':height,'width_inches':width,'source':source,'weather_conditions':WEATHERS,'modes':MODES,'panels':12,'crop_xyxy':crop}

if __name__=='__main__':
    check_inputs()
    figures=[build('truck_car','matched_128_vehicles'),build('small_people','matched_128_people')]
    record={'generator':'scripts/build_comparison_gallery.py','input_manifest':'figures/comparison_inputs/provenance.json','figures':figures,'scope':MANIFEST['scope'],'runtime':MANIFEST['runtime'],'no_new_inference':True}
    (ROOT/'evidence/comparison_galleries.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

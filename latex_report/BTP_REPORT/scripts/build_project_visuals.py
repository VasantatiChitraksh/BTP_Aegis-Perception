"""Project narrative diagrams and displays, derived from recorded evidence.
Run from any directory: python scripts/build_project_visuals.py
"""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
E=ROOT/'evidence'; BLUE='#3B6599'; TEAL='#246967'; AMBER='#994E3A'; INK='#203343'; GRAY='#596776'; LIGHT='#F0F4F6'; PALE='#E6EFEB'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':GRAY,'savefig.facecolor':'white'})
def rows(n):return list(csv.DictReader((E/n).open()))
def canvas(h):
 f=plt.figure(figsize=(6.85,h));a=f.add_axes((0,0,1,1));a.set(xlim=(0,6.85),ylim=(0,h));a.axis('off');return f,a
def text(a,x,y,t,size=9,color=INK,weight='normal',ha='left',va='center'):
 a.text(x,y,t,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.35)
def arrow(a,p,q,color=TEAL,dashed=False):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=10,lw=1,color=color,linestyle='--' if dashed else '-'))
def save(f,name):
 f.savefig(OUT/(name+'.pdf'),bbox_inches=None,metadata={'CreationDate':None,'ModDate':None,'Creator':'Aegis evidence-based report visual builder'});plt.close(f)

def project_flow():
 f,a=canvas(1.70)
 color='#374F60'
 xs=[.03,2.36,4.69];w=2.12
 top=[('Paired weather images','Frozen split and background groups'),('Six restoration models','Training and validation selection'),('Selected Restormer','26.13M parameters; epoch 51')]
 bottom=[('Matched continuation','Eight controlled training arms'),('Detection controls','Raw, resize-only and restored views'),('Export and Nano execution','Numerics, AP, latency and energy')]
 for yy,nodes in [(1.03,top),(.07,bottom)]:
  for x,(head,detail) in zip(xs,nodes):
   a.add_patch(Rectangle((x,yy),w,.55,fc='white',ec=color,lw=.65))
   a.text(x+w/2,yy+.355,head,fontsize=9.1,fontfamily='DejaVu Serif',fontweight='bold',ha='center',va='center',color='#222222')
   a.text(x+w/2,yy+.145,detail,fontsize=7.5,fontfamily='DejaVu Serif',ha='center',va='center',color='#333333')
 for i in range(2):arrow(a,(xs[i]+w,1.305),(xs[i+1],1.305),color)
 centers=[x+w/2 for x in xs]
 a.plot([centers[2],centers[2]],[1.03,.82],color=color,lw=.7)
 a.plot([centers[0],centers[2]],[.82,.82],color=color,lw=.7)
 for x in centers:arrow(a,(x,.82),(x,.62),color)
 save(f,'project_evidence_flow')

def split():
 rs=rows('dataset_split.csv');lookup={(r['weather'],r['split']):r for r in rs};ws=['rain','fog','snow','smoke'];f,axs=plt.subplots(1,2,figsize=(6.85,2.65),gridspec_kw={'width_ratios':[1.1,1]})
 left=[0]*4
 for split,c in [('train',BLUE),('val','#8EB8BB'),('test',TEAL)]:
  vals=[int(lookup[w,split]['pairs']) for w in ws];axs[0].barh(ws,vals,left=left,color=c,label={'train':'Train','val':'Validation','test':'Test'}[split],height=.65);left=[a+b for a,b in zip(left,vals)]
 axs[0].invert_yaxis();axs[0].set_xlabel('Paired observations');axs[0].set_title('(a) Frozen split',loc='left',fontweight='bold',fontsize=10,pad=28);axs[0].legend(fontsize=7,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.0),frameon=False);axs[0].tick_params(labelsize=8)
 y=list(range(4));p=[int(lookup[w,'test']['pairs']) for w in ws];g=[int(lookup[w,'test']['backgrounds']) for w in ws]
 axs[1].barh([v-.18 for v in y],p,height=.32,color=BLUE,label='Pairs');axs[1].barh([v+.18 for v in y],g,height=.32,color=TEAL,label='Backgrounds');axs[1].set_yticks(y,ws);axs[1].invert_yaxis();axs[1].set_xlim(0,1260)
 for j in y:axs[1].text(p[j]+20,j-.18,f'{p[j]:,}',va='center',fontsize=8);axs[1].text(g[j]+20,j+.18,f'{g[j]:,}',va='center',fontsize=8)
 axs[1].set_title('(b) Held-out diversity',loc='left',fontweight='bold',fontsize=10,pad=28);axs[1].set_xlabel('Count');axs[1].legend(fontsize=7,ncol=2,loc='lower center',bbox_to_anchor=(.5,1),frameon=False);axs[1].tick_params(labelsize=8)
 f.subplots_adjust(left=.075,right=.98,top=.71,bottom=.18,wspace=.47)
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for ax in axs:assert not ax._left_title.get_window_extent(renderer).overlaps(ax.get_legend().get_window_extent(renderer)), 'Title/legend overlap'
 save(f,'dataset_coverage')

def camera():
 rs=rows('S3_camera_replay.csv');d={(r['mode'],r['case'].split('_')[0]):r for r in rs};ws=['fog','rain','snow','smoke'];f,axs=plt.subplots(1,2,figsize=(6.85,2.45))
 for ax,key,mul,title in zip(axs,['mean_psnr_db','mean_ssim'],[1,100],['PSNR change (dB)','SSIM change (×100)']):
  vals=[(float(d['restored',w][key])-float(d['raw',w][key]))*mul for w in ws];ax.barh(ws,vals,color=[TEAL if v>=0 else AMBER for v in vals],height=.6);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=.8);ax.set_title(title,loc='left',fontsize=10,fontweight='bold');ax.set_xlim(min(vals)*1.35,max(max(vals)*1.5,1))
  for j,v in enumerate(vals):ax.text(v+(.3 if v>=0 else -.3),j,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
  ax.tick_params(labelsize=8)
 f.subplots_adjust(left=.10,right=.97,top=.83,bottom=.17,wspace=.38);save(f,'camera_fidelity_scaled')

for fn in (project_flow,split,camera):fn()
records={'phase_status_source':'User-supplied project status, 13 November 2026. Phases 1–6 are completed experiments; phase 7 RAMiT + YOLOv12 edge deployment is ongoing exploration. No new performance claimed.','data_sources':['evidence/six_models.csv','evidence/S1_accuracy.csv','evidence/S2_dawn_performance.csv','evidence/detection_pilot72.csv','evidence/S3_camera_replay.csv','evidence/dataset_split.csv','figures/{weather}_{input,restormer}.png','sections/11_contributions.tex'],'figures':[n+'.pdf' for n in ['project_evidence_flow','dataset_coverage','camera_fidelity_scaled']],'display_policy':'At most two decimal places; original evidence retained at full precision.'}
(ROOT/'evidence/project_visuals.json').write_text(json.dumps(records,indent=2)+'\n')
print('Built project visuals')

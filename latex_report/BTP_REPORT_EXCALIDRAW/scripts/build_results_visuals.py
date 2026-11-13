"""Table-to-figure conversions. Values are read at full precision, labels use <=2 decimals."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
R=Path(__file__).resolve().parents[1]; E=R/'evidence'; O=R/'figures';O.mkdir(exist_ok=True)
TEAL='#246967';BLUE='#3B6599';AMBER='#994E3A';GRAY='#596776'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def rows(n):return list(csv.DictReader((E/n).open()))
def save(f,n):f.savefig(O/(n+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(f)
models=['Restormer','PromptIR','TransWeather','RAMiT','LiteWeather v2','Attention Pix2Pix'];weathers=['rain','fog','snow','smoke'];d={(v['model'],v['weather']):v for v in rows('six_models.csv')}
cmap=LinearSegmentedColormap.from_list('teal',['#F0F5F4','#A9CCC5',TEAL])
for key,mul,title,name in [('psnr_db',1,'PSNR (dB)','weather_psnr'),('ssim',100,'SSIM × 100','weather_ssim')]:
 m=np.array([[float(d[model,w][key])*mul for w in weathers] for model in models]);f,a=plt.subplots(figsize=(6.85,2.65));f.subplots_adjust(left=.24,right=.97,top=.76,bottom=.13)
 a.imshow(m,cmap=cmap,aspect='auto');a.set_xticks(range(4),[w.title() for w in weathers]);a.xaxis.tick_top();a.set_yticks(range(6),models);a.tick_params(length=0);a.set_title(title+'  •  Higher is better',fontsize=11,fontweight='bold',pad=24)
 for i in range(6):
  for j in range(4):
   a.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',color='white' if (m[i,j]-m.min())/(m.max()-m.min())>.58 else '#203343',fontsize=10)
 for j in range(4):a.add_patch(Rectangle((j-.48,np.argmax(m[:,j])-.48),.96,.96,fill=False,edgecolor=AMBER,lw=2))
 a.set_xticks(np.arange(-.5,4,1),minor=True);a.set_yticks(np.arange(-.5,6,1),minor=True);a.grid(which='minor',color='white',lw=2);a.tick_params(which='minor',length=0)
 for spine in a.spines.values():spine.set_visible(False)
 f.text(.24,.025,'Outline: best score for that weather. Same six-model test protocol.',fontsize=8,color=GRAY);save(f,name)

j={v['model']:v for v in rows('joint_test.csv')};print('joint keys',list(j))
labels={'spatial':'Spatial','detail':'Detail','combined':'Both adapters','recipe_only':'Loss recipe only','combined_ssim':'Both + SSIM','combined_ffl':'Both + FFL','combined_recipe':'Both + recipe'}
keys=[k for k in j if k not in ['original','control']]; vals=[1000*(float(j[k]['test_psnr_db'])-float(j['control']['test_psnr_db'])) for k in keys]
f,a=plt.subplots(figsize=(6.85,2.75));f.subplots_adjust(left=.23,right=.97,top=.85,bottom=.23)
a.barh([labels.get(k,k) for k in keys],vals,color=[BLUE if k in ['spatial','detail','combined'] else TEAL for k in keys],height=.60);a.invert_yaxis();a.set_xlim(0,max(vals)*1.2);a.set_xlabel('Test PSNR gain over matched continuation control (mdB)');a.set_title('Continued training explains most of the gain',loc='left',fontsize=11,fontweight='bold');a.grid(axis='x',alpha=.2);a.set_axisbelow(True)
for y,v in enumerate(vals):a.text(v+.35,y,f'+{v:.2f}',va='center',fontsize=9)
f.text(.23,.025,'Control vs original: +118.97 mdB.  1 mdB = 0.001 dB.  No replacement passed validation.',fontsize=8,color=GRAY);save(f,'continuation_millidb')

history=json.loads((E/'rain_baseline_historical.json').read_text());fine=history['metrics_test']['summary']['all'];pre=history['metrics_test_pretrained']['summary']['all']
f,axs=plt.subplots(1,2,figsize=(6.85,2.45));f.subplots_adjust(left=.085,right=.98,top=.82,bottom=.23,wspace=.35)
for a,vals,title,c in [(axs[0],[fine['raw_psnr_db'],pre['psnr_db'],fine['psnr_db']],'PSNR (dB)',TEAL),(axs[1],[100*fine['raw_ssim'],100*pre['ssim'],100*fine['ssim']],'SSIM × 100',BLUE)]:
 a.plot(range(3),vals,marker='o',color=c,lw=2);a.set_xticks(range(3),['Raw','Pretrained','Fine-tuned']);a.tick_params(labelsize=8);a.set_title(title,loc='left',fontsize=10,fontweight='bold');a.set_ylim(min(vals)-3,max(vals)+5);a.set_xlim(-.18,2.18);a.grid(axis='y',alpha=.2)
 for x,v in enumerate(vals):a.annotate(f'{v:.2f}',(x,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=c,fontweight='bold')
f.text(.085,.03,'Historical rain-only protocol: 224 test pairs; fine-tuned checkpoint selected at epoch 15.',fontsize=8,color=GRAY);save(f,'historical_rain_progress')

nr=rows('S2_dawn_performance.csv');d={(v['mode'],v['case']):v for v in nr};f,axs=plt.subplots(1,2,figsize=(6.85,2.9));f.subplots_adjust(left=.1,right=.96,top=.80,bottom=.22,wspace=.35)
ys=np.arange(3)
for a,key,title,unit in [(axs[0],'ram_peak_mb','Board RAM peak','Logged MB'),(axs[1],'gpu_mean_percent','Mean GPU utilization','Percent')]:
 for offset,mode,c,name in [(-.18,'raw',BLUE,'YOLO only'),(.18,'restored',TEAL,'Restormer + YOLO')]:
  vals=[float(d[mode,w][key]) for w in ['fog','rain','snow']];a.barh(ys+offset,vals,height=.32,color=c,label=name)
  for yy,v in zip(ys+offset,vals):a.text(v+max(vals)*.015,yy,f'{v:.0f}' if key=='ram_peak_mb' else f'{v:.2f}',va='center',fontsize=8)
 a.set_yticks(ys,['Fog','Rain','Snow']);a.invert_yaxis();a.set_xlim(0,3700 if key=='ram_peak_mb' else 117);a.set_title(title,loc='left',fontsize=10,fontweight='bold');a.set_xlabel(unit);a.tick_params(labelsize=8);a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
handles,labels=axs[0].get_legend_handles_labels();f.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=8);f.text(.1,.025,'Board-wide readings, not model allocations. Every run also records nonzero swap.',fontsize=8,color=GRAY);save(f,'nano_resources')

manifest={'figures':{'weather_psnr.pdf':{'source':'evidence/six_models.csv','measure':'native-resolution balanced per-weather PSNR'},'weather_ssim.pdf':{'source':'evidence/six_models.csv','measure':'SSIM multiplied by 100 for display'},'continuation_millidb.pdf':{'source':'evidence/joint_test.csv','measure':'1000 times arm test PSNR minus continuation-control test PSNR; not original checkpoint'},'historical_rain_progress.pdf':{'source':'evidence/rain_baseline_historical.json','scope':'historical rain-only protocol, not four-weather benchmark'},'nano_resources.pdf':{'source':'evidence/S2_dawn_performance.csv','scope':'per-weather logged board-wide RAM peak and mean GPU utilization, no new experiment'}},'display_precision':'two decimals maximum; raw evidence retained unchanged'}
(E/'new_visuals.json').write_text(json.dumps(manifest,indent=2)+'\n')

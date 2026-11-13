"""Build the project Gantt from a frozen provenance record.

Run: python scripts/build_work_timeline.py
Requires matplotlib. No repository/network access is needed for regeneration.
Date spans connect recorded milestones; they are not estimates of person-hours.
"""
from pathlib import Path
import datetime as dt
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'evidence/work_timeline_provenance.json').read_text())
OUT = ROOT / 'figures'
INK, GREY = '#203343', '#596776'
TEAL, BLUE, AMBER = '#246967', '#3B6599', '#994E3A'
PALE, RULE = '#EFF4F4', '#CCD7DD'
COLORS = [TEAL, BLUE, '#816A98']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'pdf.fonttype': 42, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': GREY, 'ytick.color': INK, 'savefig.facecolor': 'white'})

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), metadata={'CreationDate': None,
        'ModDate': None, 'Creator': 'Aegis dated-record visual builder'})
    plt.close(fig)

def day(value):
    return dt.date.fromisoformat(value)

# As in the reference MTP, an explicitly disclosed condensed early interval
# preserves room for the dense sequence of late-October/November milestones.
START, CHANGE = day('2026-08-26'), day('2026-10-07')
def xpos(value):
    d = day(value)
    return ((d - START).days / (CHANGE - START).days * 12
            if d <= CHANGE else 12 + (d - CHANGE).days)

def timeline():
    fig = plt.figure(figsize=(6.85, 2.65))
    ax = fig.add_axes((.315, .17, .665, .80))
    color='#3C617A'
    xend = xpos(DATA['as_of'])
    ticks = ['2026-08-26','2026-09-20','2026-10-07','2026-10-20','2026-10-30','2026-11-13']
    ticklabels = ['26 Aug','20 Sep','7 Oct','20 Oct','30 Oct','13 Nov']
    for x in map(xpos,ticks): ax.axvline(x,color='#E7EAEC',lw=.6,zorder=0)
    for i,p in enumerate(DATA['phases']):
        y=6-i;xs=[xpos(d) for d in p['events']]
        if p['status']=='Completed':
            ax.plot([min(xs),max(xs)],[y,y],color=color,lw=7,solid_capstyle='butt',alpha=.27,zorder=1)
            ax.scatter(xs,[y]*len(xs),color=color,s=14,zorder=3)
        else:
            ax.scatter(xs,[y],facecolors='white',edgecolors=color,s=25,lw=1,zorder=4)
            ax.add_patch(FancyArrowPatch((xs[0]+.4,y),(xs[0]+2.8,y),arrowstyle='-|>',mutation_scale=9,linewidth=1,color=color,linestyle='--'))
            ax.text(xs[0]-1.2,y+.20,'Ongoing',ha='right',va='bottom',fontsize=8,color='#333333')
        ax.text(-1.6,y,p['number']+'  '+p['label'],ha='right',va='center',fontsize=8.1,color='#222222')
    ax.axvline(xpos('2026-10-07'),color=GREY,alpha=.45,lw=.6,linestyle=':')
    ax.set(xlim=(-.5,xend+3),ylim=(-.5,6.5),yticks=[],xticks=list(map(xpos,ticks)))
    ax.set_xticklabels(ticklabels,fontsize=8)
    ax.tick_params(axis='x',length=3,color=RULE,pad=4)
    for side in ('left','right','top'):ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(RULE)
    save(fig,'work_timeline')

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    timeline()
    print('Built work_timeline.pdf; contributions are an editable LaTeX table')

#!/usr/bin/env python3
"""Build a label-free cover pipeline from the project's saved image pairs.

Use from BTP_REPORT/scripts with paired PNGs in BTP_REPORT/figures.
All illustrated inputs and outputs are existing saved qualitative examples;
the network and device are conceptual vector symbols, not measured results.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
NAVY = '#18394b'
TEAL = '#217c85'
PALE = '#eef5f6'
EDGE = '#ccdce0'


def build(root: Path, output: Path):
    fig, ax = plt.subplots(figsize=(6.85, 4.2))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 6.85), ylim=(0, 4.2), aspect='equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.add_patch(FancyBboxPatch((.04, .06), 6.77, 4.08,
        boxstyle='round,pad=0,rounding_size=.18', fc='#f5f8f9', ec='none'))

    def tile(path, x, y, w=1.30, h=.77):
        # Common display crop within each input/output pair, never new inference.
        im = Image.open(path).convert('RGB')
        if path.name.startswith('smoke_'):
            im = im.crop((0, 13, im.width, im.height))
        im = ImageOps.fit(im, (650, 385), method=Image.Resampling.LANCZOS)
        clip = FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0,rounding_size=.045', fc='none', ec=EDGE, lw=.55)
        ax.add_patch(clip)
        shown = ax.imshow(im, extent=(x, x+w, y, y+h), zorder=2)
        shown.set_clip_path(clip)
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0,rounding_size=.045', fc='none', ec=EDGE, lw=.55, zorder=3))

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end,
            arrowstyle='-|>', mutation_scale=15, lw=1.5, color=TEAL,
            shrinkA=0, shrinkB=0, zorder=5))

    # A clockwise pipeline: degraded scenes, network, restored scenes, device.
    # Larger 2 x 2 panels keep the actual photographs prominent on the cover.
    for start, bottom, suffix in [(.24, 2.38, 'input'), (3.91, .24, 'restormer')]:
        for i, weather in enumerate(['fog', 'snow', 'rain', 'smoke']):
            tile(root / 'figures' / f'{weather}_{suffix}.png',
                 start + (i % 2)*1.40, bottom + (.87 if i < 2 else 0))
    arrow((3.10, 3.20), (4.05, 3.20))
    arrow((5.26, 2.28), (5.26, 2.04))
    arrow((3.72, 1.05), (2.75, 1.05))

    # Conceptual learned representation; no labels or phase-status markings.
    ax.add_patch(FancyBboxPatch((4.32, 2.44), 1.86, 1.51,
        boxstyle='round,pad=0,rounding_size=.12', fc='white', ec=EDGE, lw=.85))
    xs = [4.57, 5.25, 5.93]
    ys = [[2.78, 3.20, 3.62], [2.66, 3.02, 3.38, 3.74], [2.78, 3.20, 3.62]]
    for c in [0, 1]:
        for y1 in ys[c]:
            for y2 in ys[c+1]:
                ax.plot([xs[c], xs[c+1]], [y1, y2], color='#80aeb4', lw=.85, alpha=.65, zorder=2)
    for c in range(3):
        for yy in ys[c]:
            ax.add_patch(Circle((xs[c], yy), .09 if c != 1 else .095,
                fc=TEAL if c == 1 else NAVY, ec='white', lw=.8, zorder=4))

    # Conceptual edge processor scaled to balance the prominent image panels.
    cx, cy = 1.63, 1.07
    for pos in [-.40, -.20, 0, .20, .40]:
        ax.plot([cx-.75, cx-.59], [cy+pos, cy+pos], color=NAVY, lw=2)
        ax.plot([cx+.59, cx+.75], [cy+pos, cy+pos], color=NAVY, lw=2)
        ax.plot([cx+pos, cx+pos], [cy-.75, cy-.59], color=NAVY, lw=2)
        ax.plot([cx+pos, cx+pos], [cy+.59, cy+.75], color=NAVY, lw=2)
    ax.add_patch(FancyBboxPatch((cx-.60, cy-.60), 1.20, 1.20,
        boxstyle='round,pad=0,rounding_size=.09', fc=NAVY, ec=NAVY, lw=.9, zorder=3))
    ax.add_patch(FancyBboxPatch((cx-.405, cy-.405), .81, .81,
        boxstyle='round,pad=0,rounding_size=.05', fc='#e8f2f3', ec='none', zorder=4))
    for dx in [-.215, 0, .215]:
        for dy in [-.215, 0, .215]:
            ax.add_patch(Rectangle((cx+dx-.065, cy+dy-.065), .13, .13,
                fc=TEAL, ec='none', zorder=5))
    ax.add_patch(Circle((cx-.49, cy+.49), .035, fc='#91c7cb', ec='none', zorder=5))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor='white', metadata={'Title': 'Aegis Perception cover pipeline',
        'Author': 'Aegis Perception project',
        'Subject': 'Saved weather image pairs connected by conceptual restoration and edge-compute symbols'})
    fig.savefig(output.with_suffix('.png'), dpi=210, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    build(args.root, args.output or args.root / 'figures' / 'cover_pipeline.pdf')

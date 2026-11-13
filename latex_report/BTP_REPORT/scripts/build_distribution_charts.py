"""Build genuine observation histograms from compact, audited count snapshots.

Default rebuild needs only the two JSON snapshots in ../evidence. To refresh them
from full-precision saved project observations, run --refresh-from-repo at the
original repository location. No inference, training or synthetic observations.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EVIDENCE = ROOT / 'evidence'
FIGURES = ROOT / 'figures'
TEAL, AMBER, BLUE, GRAY = '#246967', '#994E3A', '#3B6599', '#596776'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white'})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path):
    return {'path': str(path.relative_to(REPO)), 'sha256': digest(path)}


def summarize(values, bins):
    a = np.asarray(values, dtype=float)
    counts, edges = np.histogram(a, bins=bins)
    assert int(sum(counts)) == len(a), 'Histogram must include every observation'
    return {'n': len(a), 'bin_edges': edges.tolist(), 'bin_counts': counts.tolist(),
        'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
        'min': float(min(a)), 'max': float(max(a))}


def write_json(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2) + '\n')


def refresh():
    source = REPO / 'docs/manuscript/evidence/per_image_restoration.csv.gz'
    with gzip.open(source, 'rt') as handle:
        rows = list(csv.DictReader(handle))
    restoration = {'measure': 'per-image restored PSNR minus raw PSNR in dB',
        'bin_width_db': 2, 'aggregation': 'each test image has equal histogram weight',
        'scope': 'Native-resolution held-out paired test split; descriptive image distribution, not group/weather-balanced benchmark mean.',
        'source': source_record(source), 'series': []}
    for weather in ['rain', 'fog', 'snow', 'smoke']:
        selected = {}
        for model in ['restormer', 'ramit']:
            subset = [r for r in rows if r['weather'] == weather and r['model'] == model]
            selected[model] = {r['sample_id'] for r in subset}
            assert len(selected[model]) == len(subset)
            gains = [float(r['psnr_gain_db']) for r in subset]
            # Catch a stale or differently defined gain column.
            assert all(abs(float(r['psnr_db']) - float(r['raw_psnr_db']) - float(r['psnr_gain_db'])) < 1e-8 for r in subset)
            series = {'weather': weather, 'model': model, **summarize(gains, np.arange(-12, 32, 2))}
            series['positive_gain_percent'] = 100 * sum(v > 0 for v in gains) / len(gains)
            restoration['series'].append(series)
        assert selected['restormer'] == selected['ramit']
    write_json('restoration_gain_histogram.json', restoration)

    base = REPO / 'artifacts/edge_v1/nano_20261111/replay_accuracy_1920/dawn_replay'
    identity = json.loads((base / 'identity.json').read_text())
    latency = {'measure': 'synchronized per-frame pipeline_ms', 'bin_width_ms': 20,
        'scope': identity['scope'], 'warmup_excluded': identity['warmup'],
        'minimum_seconds_per_case': identity['minimum_seconds_per_case'],
        'restorer_shape_hw': identity['restorer_shape_hw'],
        'repeat': 1, 'identity_source': source_record(base / 'identity.json'),
        'warning': 'Repeated replay observations are not independent images; raw and restored measured-frame counts differ. Each panel normalizes within its own run.',
        'series': []}
    for mode in ['raw', 'restored']:
        for weather in ['fog', 'rain', 'snow']:
            path = base / 'repeat_01' / f'{mode}_{weather}_timing.json'
            saved = json.loads(path.read_text())
            values = [r['pipeline_ms'] for r in saved['frames']]
            bins = np.arange(100, 282, 20) if mode == 'raw' else np.arange(1500, 1802, 20)
            series = {'mode': mode, 'weather': weather, 'source': source_record(path),
                'unique_images': saved['summary']['unique_frames'], **summarize(values, bins)}
            assert series['n'] == saved['summary']['measured_frames']
            assert abs(series['median'] - saved['summary']['pipeline_median_ms']) < 1e-8
            assert abs(series['p95'] - saved['summary']['pipeline_p95_ms']) < 1e-8
            latency['series'].append(series)
    write_json('nano_latency_histogram.json', latency)


def prettify(ax):
    ax.grid(axis='y', alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, length=3)
    for spine in ax.spines.values():
        spine.set_color('#ABB5BC')


def save(fig, name):
    fig.savefig(FIGURES / name, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def draw_restoration(data):
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.90), sharex=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.14, top=.87, hspace=.49, wspace=.22)
    index = {(s['weather'], s['model']): s for s in data['series']}
    for ax, weather in zip(axes.flat, ['rain', 'fog', 'snow', 'smoke']):
        for model, color in [('ramit', AMBER), ('restormer', TEAL)]:
            s = index[weather, model]
            counts = np.asarray(s['bin_counts']) * 100 / s['n']
            ax.stairs(counts, s['bin_edges'], color=color, linewidth=1.5, fill=True, alpha=.16)
            ax.stairs(counts, s['bin_edges'], color=color, linewidth=1.3)
        ax.axvline(0, color='#5A626A', linestyle=':', linewidth=.9)
        ax.set_title(f"{weather.title()}  ·  {index[weather, 'restormer']['n']:,} images", fontsize=9,
                     fontweight='bold', loc='left', pad=5)
        medians = (f"Median gain  Restormer {index[weather, 'restormer']['median']:.2f} dB"
                   f"\nRAMiT {index[weather, 'ramit']['median']:.2f} dB")
        ax.text(.02, .95, medians, transform=ax.transAxes, va='top', ha='left', fontsize=6.8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1.5})
        ax.set_xlim(-12, 30)
        ax.set_ylim(0, max(max(np.asarray(index[weather, m]['bin_counts']) * 100 / index[weather, m]['n']) for m in ['ramit', 'restormer']) * 1.37)
        ax.set_xticks([-10, 0, 10, 20, 30])
        prettify(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel('Images in bin (%)', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('PSNR gain over raw input (dB)', fontsize=8)
    fig.legend([Line2D([0],[0], color=TEAL, lw=2), Line2D([0],[0], color=AMBER, lw=2)],
               ['Restormer', 'RAMiT'], loc='upper center', bbox_to_anchor=(.5, 1.005),
               ncol=2, frameon=False, fontsize=9)
    fig.text(.09, .015, '2 dB bins; same test images for both models. Dashed line: zero improvement.', fontsize=7.5, color=GRAY)
    save(fig, 'restoration_gain_histograms.pdf')


def draw_latency(data):
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 3.85), sharey=True)
    fig.subplots_adjust(left=.105, right=.965, bottom=.15, top=.90, hspace=.57, wspace=.18)
    index = {(s['mode'], s['weather']): s for s in data['series']}
    for row, mode in enumerate(['raw', 'restored']):
        for col, weather in enumerate(['fog', 'rain', 'snow']):
            ax = axes[row, col]
            s = index[mode, weather]
            heights = np.asarray(s['bin_counts']) * 100 / s['n']
            color = BLUE if mode == 'raw' else TEAL
            ax.stairs(heights, s['bin_edges'], fill=True, color=color, alpha=.78, linewidth=.6)
            ax.axvline(s['median'], color='#273746', linestyle='--', linewidth=.8)
            ax.set_title(f'{weather.title()}  ·  n = {s["n"]}', fontsize=9, fontweight='bold', loc='left', pad=5)
            ax.text(.97, .96, f'Median {s["median"]:.2f} ms\nP95 {s["p95"]:.2f} ms',
                    transform=ax.transAxes, va='top', ha='right', fontsize=6.8,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':1.5})
            ax.set_xlim(100, 280) if mode == 'raw' else ax.set_xlim(1500, 1800)
            ax.set_xticks([100, 160, 220, 280] if mode == 'raw' else [1500, 1600, 1700, 1800])
            ax.set_ylim(0, 68)
            ax.set_yticks([0, 20, 40, 60])
            ax.set_xlabel('Pipeline latency (ms)', fontsize=7.5, labelpad=2)
            prettify(ax)
    axes[0,0].set_ylabel('YOLOv8n only\nMeasured frames (%)', fontsize=8)
    axes[1,0].set_ylabel('Restormer + YOLOv8n\nMeasured frames (%)', fontsize=8)
    fig.text(.105,.025, '20 ms bins; 24 unique images per weather. Different x-axis ranges expose variation in each path.', fontsize=7.3, color=GRAY)
    save(fig, 'nano_pipeline_latency_histograms.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-from-repo', action='store_true')
    args = parser.parse_args()
    if args.refresh_from_repo:
        refresh()
    restoration_path = EVIDENCE / 'restoration_gain_histogram.json'
    latency_path = EVIDENCE / 'nano_latency_histogram.json'
    draw_restoration(json.loads(restoration_path.read_text()))
    draw_latency(json.loads(latency_path.read_text()))
    manifest = {'generator': 'scripts/build_distribution_charts.py',
        'purpose': 'Saved observation histograms; no new experiments.',
        'generator_sha256': digest(Path(__file__)),
        'display_precision': 'At most two decimals; numeric snapshot summaries retain full precision.',
        'figures': {'restoration_gain_histograms.pdf': {'snapshot': 'evidence/restoration_gain_histogram.json',
             'sha256': digest(restoration_path), 'figure_sha256': digest(FIGURES / 'restoration_gain_histograms.pdf'), 'scope': '2,334 paired test images per model; equally weighted observations, not benchmark aggregate metric.'},
          'nano_pipeline_latency_histograms.pdf': {'snapshot': 'evidence/nano_latency_histogram.json',
             'sha256': digest(latency_path), 'figure_sha256': digest(FIGURES / 'nano_pipeline_latency_histograms.pdf'), 'scope': 'Nano labelled replay; one recorded run per weather/path; 672 raw and 72 restored timing observations across 72 unique images.'}}}
    write_json('distribution_charts_provenance.json', manifest)
    print('Wrote 2 histogram figures and their audited snapshots/provenance.')


if __name__ == '__main__':
    main()

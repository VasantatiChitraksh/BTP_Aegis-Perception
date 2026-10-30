#!/usr/bin/env python3
"""Prepare deterministic synthetic-weather frames and evaluate a frozen deployment kit.

These screen-space corruptions are stress-test proxies, not calibrated weather physics.
No model is trained or modified. The original clear frames are the paired reference.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WEATHERS = ('fog', 'rain', 'snow', 'smoke')
ROAD_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                5: 'bus', 7: 'truck'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def weather(rgb, kind, level, time_s, seed):
    """Deterministic RGB uint8 transform; same scene seed gives coherent moving fields."""
    if kind not in WEATHERS or level not in (1, 2, 3):
        raise ValueError('Expected fog/rain/snow/smoke and severity 1, 2, or 3')
    h, w = rgb.shape[:2]
    rng = np.random.default_rng(seed)
    base = rgb.astype(np.float32) / 255
    if kind in ('fog', 'smoke'):
        grid = rng.random((18, 32), dtype=np.float32)
        # A smooth advected field; NO inferred depth or physical visibility claim.
        field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        field = np.roll(field, (round(time_s * h * .008), round(time_s * w * .025)),
                        axis=(0, 1))
        field = cv2.GaussianBlur(field, (0, 0), max(1, min(w, h) * .018))
        if kind == 'fog':
            alpha = (.12, .28, .46)[level - 1] * (.75 + .5 * field)
            color = np.array([.88, .90, .92])
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            plume = .5 + .5 * np.sin(xx / w * 8 + yy / h * 3 - time_s * .7)
            alpha = (.20, .40, .65)[level - 1] * np.clip(field * 1.2 * plume, 0, 1)
            color = np.array([.54, .52, .49])
        out = base * (1 - alpha[..., None]) + color * alpha[..., None]
    else:
        layer = np.zeros((h, w), np.float32)
        # Fixed pool ensures variants use nested particles, not unrelated randomness.
        count = (180, 380, 650)[level - 1] if kind == 'rain' else (90, 180, 300)[level - 1]
        particles = rng.random((650, 5))
        for px, py, speed, size, phase in particles[:count]:
            if kind == 'rain':
                x = int(((px + time_s * .10) % 1) * w)
                y = int(((py + time_s * (.7 + speed * .5)) % 1) * h)
                length = max(3, round(h * (.015 + .025 * size)))
                cv2.line(layer, (x, y), (x + max(1, length // 4), y + length),
                         .3 + .5 * phase, 1, cv2.LINE_AA)
            else:
                x = int(((px + .025 * math.sin(time_s + phase * 6.28)
                          + time_s * .025) % 1) * w)
                y = int(((py + time_s * (.06 + speed * .12)) % 1) * h)
                radius = max(1, round(min(h, w) * (.002 + .005 * size)))
                cv2.circle(layer, (x, y), radius, .4 + .5 * phase, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (3, 3), .7)
        alpha = np.clip(layer * (.50, .70, .90)[level - 1], 0, 1)[..., None]
        out = base * (1 - alpha) + alpha * np.array([.86, .89, .94])
    return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Failed to write {path}')


def record(args):
    """Capture an unannotated camera clip; preserve actual frame timestamps."""
    sidecar = args.output.with_suffix('.capture.json')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('Choose a fresh video filename')
    if args.frames < 1 or args.fps <= 0:
        raise ValueError('Invalid frame count or FPS')
    cap = cv2.VideoCapture(args.camera)
    writer = None
    timestamps = []
    try:
        if not cap.isOpened():
            raise ValueError('No usable camera at this index; a Jetson is not itself a camera')
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = args.fps
        started = time.perf_counter()
        for _ in range(args.frames):
            ok, frame = cap.read()
            stamp = time.perf_counter() - started
            if not ok:
                raise ValueError('Camera capture failed; partial clip is not a complete recording')
            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('MP4 encoder unavailable')
            writer.write(frame)
            timestamps.append(stamp)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    write_json(sidecar, {'source_sha256': digest(args.output), 'timestamps': timestamps,
                        'nominal_fps': fps,
                        'scope': 'Raw camera capture, no restoration/detection'})
    print(f'Recorded {len(timestamps)} frames: {args.output}; keep {sidecar.name} beside it')


def prepare(args):
    if args.output.exists():
        raise FileExistsError('Use a fresh output folder; previous experiments are preserved')
    if args.stride < 1 or args.frames < 1 or args.max_width < 16 or args.start < 0:
        raise ValueError('Invalid frame sampling settings')
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {args.video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('Video needs a valid frame rate for temporal sampling')
    capture_info = args.video.with_suffix('.capture.json')
    timestamps = None
    if capture_info.exists():
        info = json.loads(capture_info.read_text())
        if info['source_sha256'] != digest(args.video):
            capture.release()
            raise ValueError('Camera timestamp sidecar does not match video')
        timestamps = info['timestamps']
    args.output.mkdir(parents=True)
    cases = ['clear'] + [f'{w}_{s}' for w in WEATHERS for s in (1, 2, 3)]
    records, images = [], []
    index = 0
    first = round(args.start * fps)
    try:
        while len(images) < args.frames:
            ok, bgr = capture.read()
            if not ok:
                break
            frame_index = index
            index += 1
            if frame_index < first or (frame_index - first) % args.stride:
                continue
            h, w = bgr.shape[:2]
            if w > args.max_width:
                bgr = cv2.resize(bgr, (args.max_width, round(h * args.max_width / w)),
                                 interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image_id = len(images) + 1
            time_s = timestamps[frame_index] if timestamps else frame_index / fps
            filename = f'{image_id:06d}.png'
            reference = f'frames/clear/{filename}'
            images.append({'id': image_id, 'file_name': reference,
                           'width': rgb.shape[1], 'height': rgb.shape[0]})
            for case in cases:
                output = rgb
                if case != 'clear':
                    kind, severity = case.rsplit('_', 1)
                    output = weather(rgb, kind, int(severity), time_s, args.seed)
                relative = f'frames/{case}/{filename}'
                save_rgb(args.output / relative, output)
                records.append({'image_id': image_id, 'case': case, 'path': relative,
                                'reference': reference, 'source_frame': frame_index,
                                'time_seconds': time_s,
                                'sha256': digest(args.output / relative)})
    finally:
        capture.release()
    if not images:
        raise ValueError('No frames sampled; choose an earlier --start')
    manifest = {'schema': 1, 'scene_id': args.scene_id, 'split': args.split,
                'source_video': str(args.video.resolve()), 'source_sha256': digest(args.video),
                'source_fps': fps, 'seed': args.seed, 'stride': args.stride,
                'capture_sidecar_sha256': digest(capture_info) if timestamps else None,
                'max_width': args.max_width, 'generator_sha256': digest(__file__),
                'scope': 'Synthetic screen-space stress tests; not physical weather calibration',
                'cases': cases, 'records': records}
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'annotations.template.json', {
        'info': {'description': 'UNLABELLED: annotate clean PNGs in COCO format; xywh pixels'},
        'images': images, 'annotations': [], 'reviewed_image_ids': [],
        'categories': [{'id': k, 'name': v} for k, v in ROAD_CLASSES.items()]})
    print(f'Prepared {len(images)} frames x {len(cases)} conditions in {args.output}')


def read_manifest(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    root = dataset.resolve()
    for row in manifest['records']:
        for field in ('path', 'reference'):
            path = (root / row[field]).resolve()
            if root not in path.parents:
                raise ValueError('Unsafe dataset path')
        if digest(root / row['path']) != row['sha256']:
            raise ValueError(f"Changed input: {row['path']}")
    return manifest


def load_runtime(kit):
    # Import ONLY the requested standalone kit. Never rewrite its checksummed files.
    subprocess_args = [sys.executable, str(kit / 'scripts/deployment/edge_verify_package.py')]
    import subprocess
    subprocess.run(subprocess_args, check=True)
    sys.path.insert(0, str(kit / 'src'))
    spec = importlib.util.spec_from_file_location('weather_edge_runtime',
                                                 kit / 'scripts/deployment/edge_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a new experiment output directory')
    if args.warmup < 0 or not 0 < args.conf < 1:
        raise ValueError('Invalid warmup/confidence')
    manifest = read_manifest(args.dataset)
    candidates = {'raw': None, 'resize_control': None}
    for setting in args.restorer:
        label, filename = setting.split('=', 1)
        if label in candidates or not label.replace('_', '').isalnum():
            raise ValueError('Candidate labels must be unique alphanumeric/underscore names')
        candidates[label] = Path(filename).resolve()
    if len(candidates) < 3:
        raise ValueError('Supply at least one --restorer label=/path/to/approved.engine')
    args.output.mkdir(parents=True)
    plan = {'scene_id': manifest['scene_id'], 'split': manifest['split'],
            'cases': manifest['cases'], 'candidates': {k: str(v) if v else None
                                                     for k, v in candidates.items()},
            'dataset_sha256': digest(args.dataset / 'manifest.json'),
            'runner_sha256': digest(__file__), 'confidence': args.conf,
            'warmup': args.warmup,
            'scope': ('Pipeline timing excludes file decode, scoring and result writing; '
                      'no live FPS')}
    write_json(args.output / 'plan.json', plan)
    if args.plan_only:
        print('Plan only: no inference or measured results produced')
        return
    runtime = load_runtime(args.kit.resolve())
    from aegis_perception.edge.quality import quality
    cv2.setNumThreads(1)
    detector = runtime.Backend(args.detector.resolve())
    summary = []
    try:
        for label, engine in candidates.items():
            restorer = runtime.Backend(engine) if engine else None
            try:
                mode = 'restored' if restorer else label
                identity = {'detector_sha256': digest(args.detector),
                            'restorer_sha256': digest(engine) if engine else None,
                            'restorer_bytes': engine.stat().st_size if engine else None,
                            'requested_precision': (restorer.requested_precision
                                                    if restorer else None),
                            'environment': runtime.environment()}
                write_json(args.output / f'{label}_identity.json', identity)
                with (args.output / f'{label}_frames.jsonl').open('w') as stream:
                    for case in manifest['cases']:
                        rows = [r for r in manifest['records'] if r['case'] == case]
                        scores, latencies = [], []
                        for position, row in enumerate(rows):
                            bgr = cv2.imread(str(args.dataset / row['path']))
                            ref_bgr = cv2.imread(str(args.dataset / row['reference']))
                            if bgr is None or ref_bgr is None:
                                raise ValueError('Unreadable frame')
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            reference = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                            if position == 0:
                                for _ in range(args.warmup):
                                    runtime.process(rgb, restorer, detector, mode, args.conf)
                            restored, boxes, timings = runtime.process(
                                rgb, restorer, detector, mode, args.conf)
                            metric = quality(reference / 255, restored / 255)
                            record = {'image_id': row['image_id'], 'case': case,
                                      'candidate': label, **metric, 'timings_ms': timings,
                                      'boxes_xyxy_score_coco_class': boxes.tolist()}
                            stream.write(json.dumps(record, allow_nan=False) + '\n')
                            scores.append(metric)
                            latencies.append(timings['pipeline_ms'])
                        summary.append({'scene_id': manifest['scene_id'], 'case': case,
                                        'candidate': label, 'frames': len(rows),
                                        'psnr_db': float(np.mean([v['psnr_db'] for v in scores])),
                                        'ssim': float(np.mean([v['ssim'] for v in scores])),
                                        'pipeline_median_ms': float(np.median(latencies)),
                                        'pipeline_p95_ms': float(np.percentile(latencies, 95)),
                                        'pipeline_only_fps': 1000 / float(np.mean(latencies)),
                                        'map50_95': '', 'map50': '',
                                        'detection_accuracy': 'NOT_SCORED_WITHOUT_ANNOTATIONS'})
                        print(label, case, 'completed', flush=True)
            finally:
                if restorer:
                    restorer.close()
    finally:
        detector.close()
    with (args.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    paired = []
    lookup = {(r['case'], r['candidate']): r for r in summary}
    for row in summary:
        raw = lookup[row['case'], 'raw']
        resized = lookup[row['case'], 'resize_control']
        fp32 = lookup.get((row['case'], 'fp32'))
        paired.append({
            'scene_id': row['scene_id'], 'case': row['case'], 'candidate': row['candidate'],
            'psnr_gain_vs_raw_db': row['psnr_db'] - raw['psnr_db'],
            'ssim_gain_vs_raw': row['ssim'] - raw['ssim'],
            'psnr_gain_vs_resize_db': row['psnr_db'] - resized['psnr_db'],
            'psnr_delta_vs_fp32_db': row['psnr_db'] - fp32['psnr_db'] if fp32 else '',
            'ssim_delta_vs_fp32': row['ssim'] - fp32['ssim'] if fp32 else '',
            'median_speedup_vs_fp32': (fp32['pipeline_median_ms'] /
                                      row['pipeline_median_ms']) if fp32 else '',
        })
    with (args.output / 'comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    write_json(args.output / 'complete.json', {'completed': True, 'conditions': len(summary)})


def score(args):
    """Compute COCO AP using human-reviewed ground truth, separately per condition."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    manifest = read_manifest(args.dataset)
    plan = json.loads((args.results / 'plan.json').read_text())
    if plan['dataset_sha256'] != digest(args.dataset / 'manifest.json'):
        raise ValueError('Results belong to a different dataset')
    if not (args.results / 'complete.json').exists():
        raise ValueError('Inference matrix is incomplete')
    output = args.results / 'detection_metrics.json'
    if output.exists():
        raise FileExistsError('Detection scoring already exists; preserve the previous result')
    data = json.loads(args.annotations.read_text())
    ids = {r['image_id'] for r in manifest['records']}
    if set(data.get('reviewed_image_ids', [])) != ids:
        raise ValueError('Every sampled image must be explicitly reviewed, including empty images')
    if {im['id'] for im in data['images']} != ids:
        raise ValueError('Annotation image IDs do not match the prepared frames')
    dims = {im['id']: (im['width'], im['height']) for im in data['images']}
    for row in manifest['records']:
        if row['case'] == 'clear':
            im = cv2.imread(str(args.dataset / row['path']))
            if dims[row['image_id']] != (im.shape[1], im.shape[0]):
                raise ValueError('Annotation geometry differs from prepared frames')
    category_ids = {c['id'] for c in data['categories']}
    if not category_ids <= set(ROAD_CLASSES):
        raise ValueError('Use zero-based COCO road IDs: 0,1,2,3,5,7')
    if any(c['name'] != ROAD_CLASSES[c['id']] for c in data['categories']):
        raise ValueError('Category names must match the zero-based COCO road IDs')
    seen = set()
    for ann in data['annotations']:
        if (ann['id'] in seen or ann['image_id'] not in ids
                or ann['category_id'] not in category_ids):
            raise ValueError('Invalid annotation ID or category')
        seen.add(ann['id'])
        x, y, w, h = ann['bbox']
        iw, ih = dims[ann['image_id']]
        if not all(np.isfinite([x, y, w, h])) or min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError('Invalid annotation bbox')
        if x + w > iw + .01 or y + h > ih + .01:
            raise ValueError('Annotation bbox outside frame')
        ann.setdefault('area', w * h)
        ann.setdefault('iscrowd', 0)
    if not data['annotations']:
        raise ValueError('No ground-truth objects: AP is undefined, not zero or perfect')
    gt = COCO()
    gt.dataset = data
    gt.createIndex()
    metrics = []
    for label in plan['candidates']:
        records = [json.loads(line) for line in
                   (args.results / f'{label}_frames.jsonl').read_text().splitlines()]
        for case in manifest['cases']:
            subset = [r for r in records if r['case'] == case]
            if len(subset) != len(ids) or {r['image_id'] for r in subset} != ids:
                raise ValueError('Missing or duplicated predictions')
            predictions = []
            for row in subset:
                for x1, y1, x2, y2, confidence, cls in row['boxes_xyxy_score_coco_class']:
                    predictions.append({'image_id': row['image_id'], 'category_id': int(cls),
                                        'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': confidence})
            if predictions:
                result = gt.loadRes(predictions)
            else:
                result = COCO()
                result.dataset = {'images': data['images'], 'categories': data['categories'],
                                  'annotations': []}
                result.createIndex()
            evaluator = COCOeval(gt, result, 'bbox')
            evaluator.params.imgIds = sorted(ids)
            evaluator.params.catIds = sorted(category_ids)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            metrics.append({'candidate': label, 'case': case,
                            'map50_95': float(evaluator.stats[0]),
                            'map50': float(evaluator.stats[1]),
                            'recall_at_100': float(evaluator.stats[8])})
    write_json(output, {
        'scene_id': manifest['scene_id'], 'annotation_sha256': digest(args.annotations),
        'metrics': metrics, 'confidence': plan['confidence'],
        'scope': 'COCO AP; paired synthetic frame labels, not real-weather evidence'})
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('record')
    c.add_argument('--camera', type=int, default=0)
    c.add_argument('--frames', type=int, default=300)
    c.add_argument('--fps', type=float, default=30)
    c.add_argument('--output', type=Path, required=True)
    c.set_defaults(func=record)
    p = commands.add_parser('prepare')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene-id', required=True)
    p.add_argument('--split', choices=['development', 'validation', 'test'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--frames', type=int, default=60)
    p.add_argument('--stride', type=int, default=15)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--max-width', type=int, default=960)
    p.set_defaults(func=prepare)
    r = commands.add_parser('run')
    r.add_argument('--dataset', type=Path, required=True)
    r.add_argument('--kit', type=Path, required=True)
    r.add_argument('--detector', type=Path, required=True)
    r.add_argument('--restorer', action='append', default=[], metavar='LABEL=ENGINE')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--warmup', type=int, default=10)
    r.add_argument('--conf', type=float, default=.001)
    r.add_argument('--plan-only', action='store_true')
    r.set_defaults(func=run)
    s = commands.add_parser('score')
    s.add_argument('--dataset', type=Path, required=True)
    s.add_argument('--results', type=Path, required=True)
    s.add_argument('--annotations', type=Path, required=True)
    s.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

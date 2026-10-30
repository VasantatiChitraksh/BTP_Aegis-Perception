"""Weather preparation, fixed evaluation matrix and annotation safety checks."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'weather_suite', Path(__file__).resolve().parents[1] /
    'scripts/weather_testing/weather_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


@pytest.mark.parametrize('kind', suite.WEATHERS)
def test_weather_is_reproducible_and_does_not_change_input(kind):
    frame = np.full((48, 80, 3), 80, np.uint8)
    before = frame.copy()
    first = suite.weather(frame, kind, 2, 1.5, 42)
    np.testing.assert_array_equal(first, suite.weather(frame, kind, 2, 1.5, 42))
    np.testing.assert_array_equal(frame, before)
    assert first.dtype == np.uint8 and first.shape == frame.shape
    assert not np.array_equal(first, frame)
    assert not np.array_equal(first, suite.weather(frame, kind, 2, 2.5, 42))


@pytest.fixture
def prepared(tmp_path):
    video = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (80, 48))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((48, 80, 3), 70 + i * 10, np.uint8))
    writer.release()
    output = tmp_path / 'dataset'
    args = SimpleNamespace(video=video, output=output, stride=2, frames=3,
                           max_width=80, start=0, scene_id='fixture', split='development', seed=42)
    suite.prepare(args)
    return args


def test_prepare_preserves_pairs_and_detects_changed_inputs(prepared):
    manifest = suite.read_manifest(prepared.output)
    assert len(manifest['records']) == 39
    for case in manifest['cases']:
        rows = [r for r in manifest['records'] if r['case'] == case]
        assert [r['source_frame'] for r in rows] == [0, 2, 4]
        assert [r['image_id'] for r in rows] == [1, 2, 3]
    with pytest.raises(FileExistsError):
        suite.prepare(prepared)
    (prepared.output / manifest['records'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed input'):
        suite.read_manifest(prepared.output)


def test_run_matrix_and_refuse_unreviewed_annotations(prepared, tmp_path, monkeypatch):
    engine = tmp_path / 'fp16.engine'
    detector = tmp_path / 'yolo.engine'
    engine.write_bytes(b'test-only mock engine')
    detector.write_bytes(b'test-only mock detector')
    closed = []

    class Backend:
        requested_precision = 'fp16'

        def __init__(self, path):
            self.path = path

        def close(self):
            closed.append(self.path)

    def process(rgb, restorer, detector, mode, conf):
        return rgb, np.empty((0, 6)), {'pipeline_ms': 2.0}

    monkeypatch.setattr(suite, 'load_runtime', lambda kit: SimpleNamespace(
        Backend=Backend, process=process, environment=lambda: {'test_only': True}))
    args = SimpleNamespace(dataset=prepared.output, output=tmp_path / 'results', kit=tmp_path,
                           detector=detector, restorer=[f'fp16={engine}'], warmup=1,
                           conf=.001, plan_only=False)
    suite.run(args)
    assert len(closed) == 2
    assert json.loads((args.output / 'complete.json').read_text())['conditions'] == 39
    assert 'NOT_SCORED_WITHOUT_ANNOTATIONS' in (args.output / 'summary.csv').read_text()
    pytest.importorskip('pycocotools')
    score_args = SimpleNamespace(dataset=prepared.output, results=args.output,
                                 annotations=prepared.output / 'annotations.template.json')
    with pytest.raises(ValueError, match='explicitly reviewed'):
        suite.score(score_args)


@pytest.mark.parametrize('perfect', [True, False])
def test_coco_scoring_known_perfect_and_empty_predictions(prepared, tmp_path, perfect):
    pytest.importorskip('pycocotools')
    manifest = suite.read_manifest(prepared.output)
    annotations = json.loads((prepared.output / 'annotations.template.json').read_text())
    annotations['reviewed_image_ids'] = [1, 2, 3]
    annotations['annotations'] = [
        {'id': i, 'image_id': i, 'category_id': 2, 'bbox': [10, 10, 10, 10]}
        for i in (1, 2, 3)]
    annotation_file = tmp_path / 'reviewed.json'
    suite.write_json(annotation_file, annotations)
    results = tmp_path / 'scoring'
    results.mkdir()
    suite.write_json(results / 'plan.json', {
        'dataset_sha256': suite.digest(prepared.output / 'manifest.json'),
        'candidates': {'fixture': None}, 'confidence': .001})
    suite.write_json(results / 'complete.json', {'completed': True})
    records = [
        {'image_id': r['image_id'], 'case': r['case'],
         'boxes_xyxy_score_coco_class': [[10, 10, 20, 20, .9, 2]] if perfect else []}
        for r in manifest['records']]
    (results / 'fixture_frames.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in records))
    suite.score(SimpleNamespace(dataset=prepared.output, results=results,
                                annotations=annotation_file))
    metrics = json.loads((results / 'detection_metrics.json').read_text())['metrics']
    assert len(metrics) == 13
    for row in metrics:
        assert row['map50_95'] == pytest.approx(1.0 if perfect else 0.0)

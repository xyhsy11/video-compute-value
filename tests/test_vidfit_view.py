"""Presentation checks against saved public evidence; no simulated model results."""

import json
import shutil
from pathlib import Path

import pytest

from video_compute_value.vidfit_view import (
    E0, E1, E2, PUBLIC_FILES, build_app, e0_panel, e2_panel,
    export_dashboard, load_evidence, main, preview_frames,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def public_copy(tmp_path):
    for relative in PUBLIC_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def test_saved_observations_and_gate():
    evidence = load_evidence(ROOT)
    assert [row['correct'] for row in evidence['e0']] == [15, 18, 19, 18, 21]
    assert evidence['e1']['h_choice'] == .1875
    assert evidence['e2']['delta_fixed'] == pytest.approx(2/120)
    assert evidence['e2']['positive_folds'] == 2
    assert evidence['e2']['advancement'] is False
    assert len(evidence['sources']) == len(PUBLIC_FILES)


@pytest.mark.parametrize('relative', [f'{E1}/summary.json', f'{E1}/correctness.csv', f'{E2}/summary.json'])
def test_changed_verified_source_blocks_display(public_copy, relative):
    with (public_copy / relative).open('a', encoding='utf-8') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_evidence(public_copy)


def test_changed_e2_correctness_blocks_display(public_copy):
    path = public_copy / E2 / 'correctness.csv'
    text = path.read_text(encoding='utf-8').replace('S01Q1,1,', 'S01Q1,0,', 1)
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='Saved evidence disagrees'):
        load_evidence(public_copy)


def test_missing_source_is_not_a_zero_observation(public_copy):
    (public_copy / E2 / 'correctness.csv').unlink()
    with pytest.raises(FileNotFoundError):
        load_evidence(public_copy)


def test_public_projection_discards_unknown_fields(public_copy, tmp_path):
    path = public_copy / E0 / 'public_summary.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    value['private_annotation'] = '<script>PRIVATE_CANARY</script>'
    value['conditions'][0]['raw_answer'] = 'PRIVATE_CANARY'
    path.write_text(json.dumps(value), encoding='utf-8')
    evidence = load_evidence(public_copy)
    output = tmp_path / 'export'
    export_dashboard(evidence, output)
    assert 'PRIVATE_CANARY' not in (output / 'index.html').read_text(encoding='utf-8')
    assert 'PRIVATE_CANARY' not in (output / 'evidence.json').read_text(encoding='utf-8')


def test_export_keeps_limitations_and_source_links(tmp_path):
    evidence = load_evidence(ROOT)
    path = export_dashboard(evidence, tmp_path)
    text = path.read_text(encoding='utf-8')
    for phrase in ('MEASUREMENT_UNSTABLE', 'NOT_ADMITTED_E2A_ADVANCEMENT', 'hindsight',
                   '0/132', 'not a live remaining-budget meter', evidence['source_fingerprint']):
        assert phrase in text
    assert '<script' not in text
    assert '/private/' not in text
    assert '/blob/' + evidence['evidence_commit'] in text
    assert json.loads((tmp_path / 'evidence.json').read_text(encoding='utf-8')) == evidence


def test_output_cannot_target_frozen_results():
    with pytest.raises(ValueError, match='frozen results'):
        main(['--root', str(ROOT), '--output-dir', str(ROOT / E1)])


def test_chart_controls_show_actual_observations():
    evidence = load_evidence(ROOT)
    assert '2.53 s' in e0_panel(evidence, 'Request time (descriptive)')
    assert '65.11%' in e0_panel(evidence, 'Timing repetition range')
    assert 'Fold 5' in e2_panel(evidence, 'Outer-fold gains')
    assert '-4.17 pp' in e2_panel(evidence, 'Outer-fold gains')
    with pytest.raises(ValueError, match='Unknown E0'):
        e0_panel(evidence, 'invented')


def test_optional_gradio_app_builds(tmp_path):
    pytest.importorskip('gradio')
    pytest.importorskip('vidfit')
    evidence = load_evidence(ROOT)
    path = export_dashboard(evidence, tmp_path)
    app = build_app(evidence, path)
    labels = [c.get('props', {}).get('label') for c in app.config['components']]
    assert 'E0 · Measurement' in labels
    assert 'E1 · Headroom' in labels
    assert 'VidFit · Inspect frames' in labels
    assert not any(label in labels for label in ('Model preset', 'Backend', 'Run selected settings'))
    from gradio.data_classes import FileData
    video = next(c for c in app.blocks.values() if getattr(c, 'label', None) == 'Local video')
    upload = tmp_path / 'unchanged-upload.mp4'
    upload.write_bytes(b'preprocess identity test; not experiment evidence')
    assert video.preprocess(FileData(path=str(upload))) == str(upload)
    assert upload.read_bytes() == b'preprocess identity test; not experiment evidence'
    app.close()


def test_real_vidfit_cpu_preview(tmp_path):
    pytest.importorskip('vidfit')
    import av
    import numpy as np

    path = tmp_path / 'cpu-inspection.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(16):
            pixels = np.full((48, 64, 3), i*12, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    images, identities = preview_frames(str(path), 4)
    assert len(images) == identities['selected_frames'] == 4
    assert len({f['frame_index'] for f in identities['frames']}) == 4
    assert all(len(f['sha256_rgb']) == 64 for f in identities['frames'])
    assert 'no model answer' in identities['mode']

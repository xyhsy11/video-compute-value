"""Explicit-allowlist E2 continuation publication audit, including exact Git blobs."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import re
import subprocess

from PIL import Image

from video_compute_value.common import file_hash, read_json, write_json

CODE = {'.gitattributes', '.gitignore', 'AGENTS.md', 'README.md', 'configs/e002_continuation.yaml',
        'protocols/e002_continuation_v2.md', 'tests/test_e002_continuation.py',
        'tests/test_e002_statistics.py',
        'scripts/report_e002_continuation.py', 'scripts/audit_e002_continuation.py',
        'scripts/admit_e002b_continuation.py',
        'scripts/resume_e002_continuation.py',
        'scripts/verify_e002_statistics.py',
        'src/video_compute_value/e002_continuation.py'}
PUBLIC = {'resource_grant.json', 'resource_accounting.json', 'readiness_initial.json', 'cohort_summary.json',
          'split_summary.json', 'feature_schema.json', 'freeze.json', 'summary.json', 'verification.json', 'report.md', 'completion_report.md',
          'metrics.csv', 'fold_effects.csv', 'per_video_effects.csv', 'correctness.csv', 'gain_calibration.csv',
          'action_frequencies.csv', 'augmented_text_token_differences.csv', 'decision_seal_summary.json',
          'figure_manifest.json', 'practical_status.json', 'practical_policy_seal.json', 'policy_costs.csv',
          'cost_breakdown.csv', 'publication_audit.json', 'validation.json', 'completion_status.json',
          'input_probe_forecast.json', 'input_cohort_forecast.json', 'cpu_preflight_forecast.json',
          'E2A_collection_forecast.json', 'E2B_complete_forecast.json', 'E2B_screen_forecast.json', 'E2B_calibration_forecast.json',
          'E2B_evaluation_forecast.json', 'screen_timing_stability.json', 'calibration_timing_stability.json',
          'evaluation_timing_stability.json', 'blocker.json', 'stage_actual_vs_forecast.csv', 'statistics_verification.json'}


def paths(root, run):
    modules = ['account', 'data', 'records', 'collect', 'analysis', 'verify', 'practical']
    figures = read_json(run/'figure_manifest.json') if (run/'figure_manifest.json').exists() else {}
    return ([root/p for p in CODE]+[root/f'src/video_compute_value/e002v2_{name}.py' for name in modules]+
            [p for p in run.iterdir() if p.is_file() and p.name in PUBLIC]+
            [run/'figures'/name for name in figures])


def audit(root, run, revision=None, save=True):
    allowed = paths(root, run)
    corpus = []
    for path in allowed:
        assert path.is_file() and path.stat().st_size < 2*1024**2, path
        if path.suffix == '.png':
            with Image.open(path) as img:
                img.verify()
            continue
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'C:[\\/]+Users[\\/]|/(?:home|Users)/[^/]+/|hf_[A-Za-z0-9]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY', text), path
        if path.parent == run and path.suffix == '.json':
            assert not re.search(r'"(?:question|options|answer|decoded_output|parsed_answer|generated_token_ids|features|base_rgb)"\s*:', text), path
        corpus.append(text)
    import pyarrow.parquet as pq
    annotations = pq.read_table(next((root/'data/annotations').glob('*.parquet')), columns=['question', 'options']).to_pylist()
    prohibited = {s for r in annotations for s in [r['question'], *r['options']] if len(s) >= 20}
    combined = '\n'.join(corpus)
    assert not any(text in combined for text in prohibited), 'Restricted annotation text in publication'
    link_count = 0
    for path in [root/'README.md', run/'report.md', run/'completion_report.md']:
        for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text(encoding='utf-8')):
            if re.match(r'^(https?://|#)', target):
                continue
            assert (path.parent/target.split('#')[0]).exists(), (path, target)
            link_count += 1
    for name, value in read_json(run/'figure_manifest.json').items():
        assert file_hash(run/'figures'/name) == value['sha256']
        assert file_hash(run/value['data']) == value['data_sha256']
    history = read_json(run/'private/historical_inventory.json')
    assert all(file_hash(root/name) == sha for name, sha in history.items())
    ledger = read_json(run/'private/resource_ledger.json')
    task = read_json(run/'private/task.json')
    parent = read_json(root/task['parent_path'])
    for key in ('sessions', 'external_events'):
        assert ledger.get(key, [])[:len(parent.get(key, []))] == parent.get(key, [])
    charged = sum(s['active_seconds'] for s in ledger['sessions'])+sum(s['active_seconds'] for s in ledger.get('external_events', []))
    accounting = read_json(run/'resource_accounting.json')
    assert math.isclose(charged, ledger['active_seconds'], abs_tol=1e-5)
    assert accounting['cumulative_seconds'] == ledger['active_seconds'] <= 172800
    assert accounting['storage_current_bytes'] <= 20*1024**3
    assert accounting['video_transfer_and_interruption_upper_bound_bytes'] == ledger['video_transfer_bytes']+ledger.get('interrupted_transfer_upper_bound_bytes', 0) <= 1024**3
    for name, sha in read_json(run/'freeze.json')['source_files'].items():
        assert file_hash(root/name) == sha, name
    if read_json(run/'practical_status.json')['status'] == 'NOT_ADMITTED_E2A_ADVANCEMENT':
        for directory in ('timing_workers', 'timing_records', 'timing_stage'):
            assert not list((run/'private'/directory).glob('*.json')), directory
    if revision:
        prefix = ':' if revision == 'index' else revision+':'
        args = ['git', 'diff', '--cached', '--name-only'] if revision == 'index' else ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', revision]
        changed = subprocess.check_output(args, cwd=root, text=True).splitlines()
        names = {p.relative_to(root).as_posix() for p in allowed}
        assert set(changed) <= names, set(changed)-names
        for name in changed:
            assert not re.search(r'(^|/)(private|data|records|\.cache|\.venv)(/|$)', name)
            raw = subprocess.check_output(['git', 'show', prefix+name], cwd=root)
            assert hashlib.sha256(raw).hexdigest() == file_hash(root/name), name
    result = {'all_publication_checks_passed': True, 'files_checked': len(allowed), 'local_links_checked': link_count,
              'restricted_annotation_strings_checked': len(prohibited), 'historical_files_preserved': len(history),
              'git_revision_checked': revision, 'scientific_success_is_separate': True,
              'final_resource_lineage_and_cap_checks_passed': True,
              'all_executed_scientific_source_hashes_match': True}
    if save:
        write_json(run/'publication_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--revision')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(audit(Path(__file__).resolve().parents[1], args.run_dir.resolve(), args.revision, not args.check_only))

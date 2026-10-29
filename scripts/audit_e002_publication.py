"""E2 explicit-allowlist privacy, links, figure sources and optional Git-byte audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import subprocess

from PIL import Image

from video_compute_value.common import file_hash, read_json, write_json

PUBLIC = {'report.md', 'summary.json', 'verification.json', 'completion_report.md', 'cohort_summary.json',
          'split_summary.json', 'feature_schema.json', 'metrics.csv', 'fold_effects.csv', 'per_video_effects.csv',
          'forecast.json', 'resource_accounting.json', 'resource_amendment_request.md', 'compatibility.json',
          'implementation_freeze.json', 'status.json', 'progress.csv', 'figure_manifest.json', 'validation.json',
          'publication_audit.json', 'publication.json', 'implementation_plan.md', 'compatibility_verification.json'}
CODE = {'.gitattributes', 'AGENTS.md', 'README.md', 'configs/e002.yaml', 'protocols/e002_preacquisition_predictability_v1.md',
        'scripts/report_e002.py', 'scripts/audit_e002_publication.py', 'scripts/verify_e002_compatibility.py',
        'tests/test_e002.py'}


def audit(root, run, revision=None, *, save=True):
    public_paths = [p for p in run.iterdir() if p.is_file() and p.name in PUBLIC]
    public_paths += list((run/'figures').glob('e2_progress.*'))
    code_paths = [root/p for p in CODE]
    code_paths += list((root/'src/video_compute_value').glob('e002*.py'))
    paths = code_paths+public_paths
    forbidden = re.compile(r'C:[\\/]+Users[\\/]|/(?:home|Users)/[^/]+/|hf_[A-Za-z0-9]{25,}|'
                           r'-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for path in paths:
        assert path.stat().st_size < 2*1024**2, path.name
        if path.suffix == '.png':
            continue
        text = path.read_text(encoding='utf-8')
        assert not forbidden.search(text), path.name
        if path.parent == run and path.suffix == '.json':
            assert not re.search(r'"(?:decoded_output|parsed_answer|generated_token_ids|answer|options|question|base_rgb)"\s*:', text), path.name
    # Scan literal restricted annotation text and raw generated answers (except short letters).
    import pyarrow.parquet as pq
    annotations = pq.read_table(next((root/'data/annotations').glob('*.parquet')),
                                columns=['question', 'options']).to_pylist()
    prohibited = {s for r in annotations for s in [r['question'], *r['options']] if len(s) >= 20}
    corpus = '\n'.join(p.read_text(encoding='utf-8') for p in paths if p.suffix != '.png')
    assert not any(text in corpus for text in prohibited), 'Restricted annotation text present'
    links = 0
    for path in [root/'README.md', run/'report.md', run/'resource_amendment_request.md']:
        for match in re.finditer(r'\]\(([^\s)]+)\)', path.read_text(encoding='utf-8')):
            target = match.group(1)
            if re.match(r'^(https?://|#)', target):
                continue
            assert (path.parent/target.split('#')[0]).exists(), (path.name, target)
            links += 1
    figures = read_json(run/'figure_manifest.json')
    for name, detail in figures.items():
        assert file_hash(run/'figures'/name) == detail['sha256']
        assert file_hash(run/detail['data']) == detail['data_sha256']
    with Image.open(run/'figures/e2_progress.png') as image:
        image.verify()
    with (run/'progress.csv').open(encoding='utf-8', newline='') as handle:
        counts = list(csv.DictReader(handle))
    assert [int(r['completed']) for r in counts] == [0, 0, 0]
    assert [int(r['planned']) for r in counts] == [40, 120, 1080]
    if revision:
        if revision == 'index':
            names = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).splitlines()
            blob_prefix = ':'
        else:
            names = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', revision], cwd=root, text=True).splitlines()
            blob_prefix = revision+':'
        allowed = {p.relative_to(root).as_posix() for p in paths}
        assert set(names) <= allowed, set(names)-allowed
        for name in names:
            assert not re.search(r'(^|/)(private|data|records|\.venv|\.cache)(/|$)', name)
            raw = subprocess.check_output(['git', 'show', blob_prefix+name], cwd=root)
            assert hashlib.sha256(raw).hexdigest() == file_hash(root/name), f'Git bytes differ: {name}'
    result = {'checks_passed': True, 'allowlisted_files_checked': len(paths), 'local_links_checked': links,
              'restricted_annotation_strings_checked': len(prohibited), 'figure_sources_checked': len(figures),
              'git_revision_checked': revision, 'scientific_success': False}
    if save:
        write_json(run/'publication_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--revision')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(audit(Path(__file__).resolve().parents[1], args.run_dir.resolve(), args.revision, save=not args.check_only))

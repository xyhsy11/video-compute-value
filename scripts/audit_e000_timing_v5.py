"""Audit explicitly staged task-owned public blobs; no collection or historical writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from report_e000_timing_v5 import public_safe
from video_compute_value.common import file_hash, read_json


def audit(run, root):
    relative = run.relative_to(root).as_posix()
    fixed = {'README.md', '.gitattributes', 'configs/e000_timing_stabilization_v5.yaml',
             'protocols/e000_timing_stabilization_v5.md', 'tests/test_e000_timing_stabilization.py',
             'scripts/stabilize_e000_timing_v5.py', 'scripts/verify_e000_timing_v5.py',
             'scripts/verify_e000_timing_closeout_v5.py',
             'scripts/report_e000_timing_v5.py', 'scripts/audit_e000_timing_v5.py'}
    safe_names = {'freeze.json', 'schedule.json', 'summary.json', 'verification.json', 'resource_accounting.json',
                  'cpu_preparation_forecast.json', 'preflight_forecast.json', 'preflight_verification.json',
                  'diagnostic_forecast.json', 'confirmation_screen_forecast.json', 'confirmation.json',
                  'confirmation_status.json', 'aggregates.csv', 'question_ranges.csv', 'report.md',
                  'environment_summary.json', 'figure_manifest.json', 'validation.json', 'publication_audit.json',
                  'status.json', 'next_action.json', 'review.json'}
    safe_names |= {'failure_summary.json', 'closeout_verification.json'}
    safe_names |= {'cpu_preparation_summary.json', 'cpu_preparation_summary.csv'}
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).splitlines()
    if not staged:
        raise ValueError('No staged task changes')
    frozen = read_json(run / 'private/frozen.json')
    child_prefix = None
    if (run / 'confirmation.json').exists():
        child_prefix = read_json(run / 'confirmation.json')['run'] + '/'
    blobs = {}
    cohort = read_json(run / 'private/cohort.json')
    restricted_texts = [text for v in cohort['videos'] for q in v['questions']
                        for text in [q['question'], *q['options']] if len(text) >= 12]
    for name in staged:
        result_name = name.removeprefix(relative + '/')
        child_safe = bool(child_prefix and name.startswith(child_prefix) and name.removeprefix(child_prefix) in
                          {'freeze.json', 'schedule.json', 'status.json', 'summary.json', 'verification.json'})
        if name not in fixed and not child_safe and not (name.startswith(relative + '/') and (result_name in safe_names or
                                      result_name.startswith('figures/') and result_name.endswith('.png'))):
            raise ValueError('Not in explicit task publication allowlist: ' + name)
        blob = subprocess.check_output(['git', 'show', ':' + name], cwd=root)
        sha = hashlib.sha256(blob).hexdigest()
        if sha != file_hash(root / name):
            raise ValueError('Index differs from verified working bytes: ' + name)
        if name in frozen['source_files'] and sha != frozen['source_files'][name]:
            raise ValueError('Staged executed-source bytes differ from freeze: ' + name)
        if not name.endswith('.png'):
            text = blob.decode('utf-8')
            if name.startswith(relative + '/') or child_safe:
                public_safe(json.loads(text) if name.endswith('.json') else text)
                if any(t in text for t in restricted_texts):
                    raise ValueError('Restricted benchmark text in staged artifact')
        blobs[name] = sha
    inventory = read_json(run / 'private/historical_inventory.json')
    if any(file_hash(root / name) != sha for name, sha in inventory.items()):
        raise ValueError('Historical file changed')
    return {'passed': True, 'staged_files': len(staged), 'historical_files_unchanged': len(inventory),
            'staged_blob_sha256': blobs, 'exact_executed_bytes_match': True,
            'restricted_artifacts_excluded': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2))

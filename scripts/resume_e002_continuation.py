"""Canonical E2 resume driver with a whole-E2B resource admission before any E2B spending."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from video_compute_value.common import Blocked, digest, exclusive_lock, read_json, write_json
from video_compute_value.e002v2_account import Budget
from video_compute_value.e002v2_records import records
from admit_e002b_continuation import admit
from verify_e002_statistics import verify as verify_statistics


def resume(root, run):
    def operation(name):
        subprocess.run([sys.executable, '-m', 'video_compute_value.e002_continuation', name,
                        '--run-dir', str(run)], cwd=root, check=True)
    if not (run/'private/cohort_manifest.json').exists():
        operation('prepare')
    manifest_path = run/'private/call_manifest.json'
    complete = manifest_path.exists() and len(records(run, read_json(manifest_path))) == 1080
    if not complete:
        # Retain every previous admission before recalculating a remaining-stage forecast.
        for path in run.glob('*forecast.json'):
            value = read_json(path)
            write_json(run/f'private/admission_history/{path.stem}_{digest(value)}.json', value)
        operation('preflight')
        operation('collect')
    operation('fit')
    operation('analyze')
    verify_statistics(root, run)
    operation('verify')
    summary = read_json(run/'summary.json')
    if summary['predictive_value'] == 'PREDICTIVE_SIGNAL_IN_DEVELOPMENT_PILOT':
        prior = read_json(run/'practical_status.json') if (run/'practical_status.json').exists() else {}
        terminal = {'COMPLETE', 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT',
                    'INPUT_INTEGRITY_FAILED', 'NO_FEASIBLE_POLICY_AT_REFERENCE_BUDGET'}
        if prior.get('status') not in terminal:
            with exclusive_lock(run/'.lock'):
                admit(root, run)
            operation('evaluate')
            # A newly executed practical stage changes status and must preserve the E2A evidence.
            operation('verify')
    else:
        with exclusive_lock(run/'.lock'):
            write_json(run/'practical_status.json', {'status': 'NOT_ADMITTED_E2A_ADVANCEMENT',
                'reason': 'E2A failed its frozen advancement rule; expensive E2B was not run.',
                'timing_validity': 'UNMEASURED', 'practical_value': 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT'})
    operation('account')
    operation('report')
    with exclusive_lock(run/'.lock'):
        Budget(root, run).checkpoint('Canonical resume verification and report closeout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--wait-for-active-stage', action='store_true')
    args = parser.parse_args()
    if args.wait_for_active_stage:
        while True:
            try:
                with exclusive_lock(args.run_dir.resolve()/'.lock'):
                    break
            except Blocked as exc:
                if exc.status != 'BLOCKED_ENVIRONMENT':
                    raise
                time.sleep(5)
    resume(Path(__file__).resolve().parents[1], args.run_dir.resolve())

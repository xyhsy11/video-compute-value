"""Resource-only admission for the ENTIRE E2B screen + calibration + live study."""
from __future__ import annotations

import argparse
from pathlib import Path

from video_compute_value.common import Blocked, exclusive_lock, read_json
from video_compute_value.e002v2_account import Budget


def admit(root, run):
    summary = read_json(run/'summary.json')
    if summary['predictive_value'] != 'PREDICTIVE_SIGNAL_IN_DEVELOPMENT_PILOT':
        raise Blocked('NOT_ADMITTED_E2A_ADVANCEMENT', 'Do not spend E2B resources after failed E2A advancement')
    cpu = read_json(run/'private/cpu_preflight.json')['checks']
    construction = max(r['action_construction_seconds'] for values in cpu.values() for r in values.values())
    observed = [read_json(p)['latency_seconds'] for p in (run/'private/records').glob('*.json')]
    # Budget forecasting uses timing observations, never correctness or acquisition choices.
    conservative_per_call = max(max(observed)+construction,
        max(r['wall_seconds'] for values in cpu.values() for r in values.values())+14.)
    startups = [read_json(p)['load_seconds'] for p in (run/'private/workers').glob('*.json')
                if 'load_seconds' in read_json(p)]
    startups.append(read_json(root/'results/e002_20261029_073134_v1/private/compatibility_worker.json')['load_seconds'])
    # Screen:48+6 warmups; panel:900+6; evaluation:<=2520+6.
    call_upper = 54+906+2526
    raw = call_upper*(conservative_per_call+.1)+9*max(startups)
    budget = Budget(root, run)
    return budget.admit('E2B_complete', raw, storage=384*1024**2,
        detail={'scientific_fingerprint': read_json(run/'private/frozen.json')['fingerprint'],
                'screen_model_calls_with_warmups': 54, 'calibration_model_calls_with_warmups': 906,
                'live_model_calls_upper_bound_with_warmups': 2526,
                'nine_model_startups': True, 'conservative_measured_call_bound_seconds': conservative_per_call,
                'training_fitting_and_independent_verification_in_explicit_reserves': True,
                'no_stage_launch_until_whole_E2B_admitted': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with exclusive_lock(args.run_dir.resolve()/'.lock'):
        print(admit(root, args.run_dir.resolve()))

"""Executable, gated E2 continuation. Historical E0/E1/E2-v1 runs are read-only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from .common import Blocked, append_json, exclusive_lock, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare-probe', 'prepare', 'preflight', 'collect', 'resume',
        'fit', 'analyze', 'evaluate', 'verify', 'report', 'account', 'close', '_worker', '_fit_fold', '_timing_worker'))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--repetition', type=int, default=1)
    parser.add_argument('--worker-key')
    parser.add_argument('--fit-input', type=Path)
    parser.add_argument('--fit-output', type=Path)
    parser.add_argument('--phase')
    parser.add_argument('--session', type=int, default=1)
    args = parser.parse_args()
    if args.operation == '_fit_fold':
        from .e002v2_analysis import fit_payload
        if args.fit_output.exists():
            parser.error('A sealed fold output is immutable')
        write_json(args.fit_output, fit_payload(read_json(args.fit_input)))
        return 0
    if args.run_dir is None:
        parser.error('--run-dir identifies the initialized continuation target')
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'results') or not (run/'resource_grant.json').exists():
        parser.error('Invalid continuation target')
    if args.operation == '_worker':
        from .e002v2_collect import worker
        worker(ROOT, run, args.repetition, args.worker_key)
        return 0
    if args.operation == '_timing_worker':
        from .e002v2_practical import timing_worker
        timing_worker(ROOT, run, args.phase, args.session, args.worker_key)
        return 0
    from .e002v2_account import Budget
    from .e002v2_data import prepare
    from .e002v2_collect import admit_collection, collect, validate
    from .e002v2_analysis import analyze, fit
    with exclusive_lock(run/'.lock'):
        current = read_json(run/'private/resource_ledger.json')['active_seconds']
        for pointer in (ROOT/'.local').glob('*latest*ledger*.json'):
            target = (ROOT/read_json(pointer)['path']).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                raise Blocked('BLOCKED_RESOURCE', 'Invalid cumulative-ledger pointer')
            if target != (run/'private/resource_ledger.json').resolve() and read_json(target)['active_seconds'] > current:
                raise Blocked('BLOCKED_RESOURCE', 'A newer cumulative descendant exists; inherit its charges first')
        budget = Budget(ROOT, run)
        try:
            if args.operation in ('prepare-probe', 'prepare'):
                prepare(ROOT, run, budget, probe=args.operation == 'prepare-probe')
            elif args.operation == 'preflight':
                print(admit_collection(ROOT, run, budget), flush=True)
            elif args.operation == 'collect':
                validate(ROOT, run)
                collect(ROOT, run, budget)
            elif args.operation == 'fit':
                fit(ROOT, run)
            elif args.operation == 'analyze':
                print(analyze(ROOT, run), flush=True)
            elif args.operation == 'evaluate':
                from .e002v2_practical import evaluate
                evaluate(ROOT, run, budget)
            elif args.operation == 'resume':
                if not (run/'private/cohort_manifest.json').exists():
                    prepare(ROOT, run, budget)
                admit_collection(ROOT, run, budget)
                collect(ROOT, run, budget)
                fit(ROOT, run)
                result = analyze(ROOT, run)
                print('E2A verdict: '+result['predictive_value'], flush=True)
                if result['predictive_value'] == 'PREDICTIVE_SIGNAL_IN_DEVELOPMENT_PILOT':
                    from .e002v2_practical import evaluate
                    evaluate(ROOT, run, budget)
                else:
                    write_json(run/'practical_status.json', {'status': 'NOT_ADMITTED_E2A_ADVANCEMENT',
                        'reason': 'E2A did not pass its frozen advancement rule; expensive E2B was not run.',
                        'timing_validity': 'UNMEASURED', 'practical_value': 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT'})
            elif args.operation == 'verify':
                from .e002v2_verify import verify
                print(verify(ROOT, run), flush=True)
            elif args.operation == 'report':
                sys.path.insert(0, str(ROOT/'scripts'))
                from report_e002_continuation import report
                report(ROOT, run)
        except BaseException as exc:
            value = {'at': utc_now(), 'operation': args.operation, 'status': getattr(exc, 'status', 'PARTIAL'),
                     'error_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            append_json(run/'private/coordinator_failures.jsonl', value)
            write_json(run/'blocker.json', {k: value[k] for k in ('at', 'operation', 'status', 'error_type')})
            traceback.print_exc()
            return 2
        finally:
            budget.checkpoint('E2 continuation '+args.operation, reserve=600 if args.operation == 'close' else 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

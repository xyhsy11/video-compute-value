"""E2 continuous-work accounting with immutable historical parent and explicit closeout reserve."""
from __future__ import annotations

from datetime import datetime
import math

from .common import Blocked, file_hash, read_json, utc_now, write_json
from .resources import Ledger

LIMIT = 27600
RESOURCE = {'cumulative_execution_limit_hours': LIMIT/3600, 'additional_disk_limit_gib': 20,
            'video_acquisition_limit_gib': 1}


def checkpoint(root, run, name, *, reserve=0):
    task = read_json(run/'private/task.json')
    if file_hash(root/task['parent_path']) != task['parent_sha256']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Historical parent accounting changed')
    ledger = Ledger(run/'private/resource_ledger.json', root, RESOURCE)
    ledger.save(force_storage=True)
    now = utc_now()
    since = task['last_checkpoint']
    elapsed = 0. if task.get('closed_at') else max(0., (datetime.fromisoformat(now)-datetime.fromisoformat(since)).total_seconds())
    ledger.value['active_seconds'] += elapsed+reserve
    ledger.value['sessions'].append({'name': name, 'started_at': since if not task.get('closed_at') else now,
        'finished_at': now, 'active_seconds': elapsed+reserve, 'outcome': 'completed',
        'precharged_closeout_seconds': reserve})
    ledger.save()
    task['last_checkpoint'] = now
    if reserve:
        task['closed_at'] = now
        task['precharged_closeout_seconds'] = reserve
    elif task.get('closed_at') and name.startswith('Resume'):
        task.pop('closed_at')
    write_json(run/'private/task.json', task)
    value = {'at': now, 'authorized_limit_seconds': LIMIT, 'inherited_seconds': task['inherited_seconds'],
        'cumulative_seconds': ledger.value['active_seconds'], 'remaining_seconds': LIMIT-ledger.value['active_seconds'],
        'storage_current_bytes': ledger.value['storage_current_bytes'], 'storage_peak_bytes': ledger.value['storage_peak_bytes'],
        'storage_limit_bytes': 20*1024**3, 'video_transfer_bytes': ledger.value['video_transfer_bytes'],
        'video_transfer_and_interruption_upper_bound_bytes': ledger.value['video_transfer_bytes']+
            ledger.value.get('interrupted_transfer_upper_bound_bytes', 0), 'video_limit_bytes': 1024**3,
        'authoritative_resource_ledger': 'private/resource_ledger.json',
        'all_sessions_closed': all(s.get('finished_at') for s in ledger.value['sessions']),
        'precharged_closeout_seconds': task.get('precharged_closeout_seconds', 0),
        'new_video_transfer_bytes': 0}
    write_json(run/'resource_accounting.json', value)
    return ledger, value


def forecast(run, accounting):
    # Explicit unmeasured planning assumptions. NEVER an admission receipt.
    raw = {'1080_model_calls_at_assumed_3_seconds': 3240, 'acquisition_and_eligibility': 900,
           'three_model_loads': 270, 'compatibility_and_warmups': 180, 'training_verification_reporting': 600}
    total = 1.5*sum(raw.values())+180+300
    result = {'stage': 'E2A', 'measured_throughput': None, 'assumptions_only': True,
        'primary_model_calls': 960, 'repeatability_model_calls': 120, 'warmup_and_probe_model_calls': None,
        'raw_assumed_seconds': raw, 'safety_factor': 1.5, 'failure_allowance_seconds': 180,
        'closeout_reserve_seconds': 300, 'conditional_required_seconds': total,
        'remaining_seconds': accounting['remaining_seconds'], 'admitted': False,
        'reasons': ['40 new eligible source groups unavailable', 'no valid measured E2 probe throughput',
                    'conditional whole-stage forecast exceeds current remainder'],
        'conditional_additional_seconds': max(0, math.ceil(total-accounting['remaining_seconds'])),
        'E2B': {'planned_pipeline_executions_upper_bound': 1440, 'model_invocations_upper_bound': 2520,
                'timing_screen_and_training_calibration_calls': None, 'admitted': False,
                'note': 'Separate frozen timing screen, training-only panel, and full-stage forecast required.'},
        'storage_estimate': {'private_records_and_traces_allowance_bytes': 128*1024**2,
                             'new_video_bytes': None, 'additional_weights_bytes': 0},
        'time_note': 'Conditional planning estimate, not measured throughput or an unconditional completion guarantee.'}
    write_json(run/'forecast.json', result)
    return result

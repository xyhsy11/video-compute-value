"""Continuation-only cumulative budget; all historical ledger entries are immutable."""
from __future__ import annotations

from datetime import datetime
import math

from .common import Blocked, file_hash, read_json, utc_now, write_json
from .resources import Ledger

LIMIT = 172800


class Budget(Ledger):
    def __init__(self, root, run):
        self.run = run
        self.task = read_json(run/'private/task.json')
        grant = read_json(run/'resource_grant.json')
        if grant.get('human_reply') != 'Authorize 48 hours total' or grant.get('authorized_limit_seconds') != LIMIT:
            raise Blocked('BLOCKED_RESOURCE', 'Missing direct numerical continuation grant')
        if file_hash(root/self.task['parent_path']) != self.task['parent_sha256']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Historical parent ledger changed')
        super().__init__(run/'private/resource_ledger.json', root,
            {'cumulative_execution_limit_hours': 48, 'additional_disk_limit_gib': 20,
             'video_acquisition_limit_gib': 1})
        # Recover outstanding transfer reservations conservatively once.
        pending = self.value.pop('pending_video_transfer_bytes', 0)
        if pending:
            self.value['interrupted_transfer_upper_bound_bytes'] = self.value.get('interrupted_transfer_upper_bound_bytes', 0)+pending
            self.save()

    def consumed_seconds(self):
        extra = max(0., (datetime.fromisoformat(utc_now())-datetime.fromisoformat(self.task['last_checkpoint'])).total_seconds())
        return self.value['active_seconds']+extra

    def allow_video_transfer(self, size):
        total = (self.value['video_transfer_bytes']+self.value.get('interrupted_transfer_upper_bound_bytes', 0)
                 +self.value.get('pending_video_transfer_bytes', 0)+size)
        if total > 1024**3:
            raise Blocked('BLOCKED_RESOURCE', 'Cumulative video traffic including reservations exceeds 1 GiB')

    def checkpoint(self, name, reserve=0):
        self.save(force_storage=True)
        now = utc_now()
        seconds = max(0., (datetime.fromisoformat(now)-datetime.fromisoformat(self.task['last_checkpoint'])).total_seconds())
        self.value['active_seconds'] += seconds+reserve
        self.value['sessions'].append({'name': name, 'started_at': self.task['last_checkpoint'],
            'finished_at': now, 'active_seconds': seconds+reserve, 'outcome': 'completed',
            'precharged_closeout_seconds': reserve})
        self.task['last_checkpoint'] = now
        write_json(self.run/'private/task.json', self.task)
        self.save()
        summary = {'at': now, 'authorized_limit_seconds': LIMIT, 'inherited_seconds': self.task['inherited_seconds'],
            'cumulative_seconds': self.value['active_seconds'], 'remaining_seconds': LIMIT-self.value['active_seconds'],
            'storage_current_bytes': self.value['storage_current_bytes'], 'storage_peak_bytes': self.value['storage_peak_bytes'],
            'storage_limit_bytes': 20*1024**3, 'video_transfer_bytes': self.value['video_transfer_bytes'],
            'video_transfer_and_interruption_upper_bound_bytes': self.value['video_transfer_bytes']+
                self.value.get('interrupted_transfer_upper_bound_bytes', 0), 'video_limit_bytes': 1024**3,
            'authoritative_resource_ledger': 'private/resource_ledger.json', 'precharged_closeout_seconds': reserve}
        write_json(self.run/'resource_accounting.json', summary)
        return summary

    def admit(self, name, raw_seconds, *, storage=0, traffic=0, detail=None):
        self.checkpoint('Admission: '+name)
        required = raw_seconds*1.5+600+1200+600
        remaining = LIMIT-self.consumed_seconds()
        value = {'stage': name, 'at': utc_now(), 'raw_seconds': raw_seconds, 'safety_factor': 1.5,
                 'failure_reserve_seconds': 600, 'verification_reporting_reserve_seconds': 1200,
                 'closeout_reserve_seconds': 600, 'required_seconds': required, 'remaining_seconds': remaining,
                 'projected_additional_bytes': storage, 'projected_traffic_bytes': traffic,
                 'additional_seconds_needed': max(0, math.ceil(required-remaining)), 'detail': detail or {}}
        value['admitted'] = (required <= remaining and self.value['storage_current_bytes']+storage <= 20*1024**3
            and self.value['video_transfer_bytes']+self.value.get('interrupted_transfer_upper_bound_bytes', 0)+traffic <= 1024**3)
        write_json(self.run/f'{name}_forecast.json', value)
        self.check(projected_seconds=required, projected_bytes=storage)
        self.allow_video_transfer(traffic)
        if not value['admitted']:
            raise Blocked('BLOCKED_RESOURCE', 'Complete next stage does not fit authorized resources')
        return value

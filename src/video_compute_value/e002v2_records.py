"""Immutable E2 records and call receipts. Completed identities cannot be retried."""
from __future__ import annotations

from .common import Blocked, append_json, digest, read_json, read_jsonl, utc_now, write_json


def save_record(directory, identity, content):
    path = directory/(identity['identity']+'.json')
    value = {**content, **identity, 'status': 'completed'}
    value['record_hash'] = digest(value)
    if path.exists():
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Refusing to overwrite an immutable completed identity')
    write_json(path, value)
    return value


def records(run, manifest):
    expected = {r['identity']: r for r in manifest}
    found = {}
    for path in (run/'private/records').glob('*.json'):
        row = read_json(path)
        ident = row.get('identity')
        if (ident not in expected or path.stem != ident or row.get('status') != 'completed'
                or any(row.get(k) != v for k, v in expected[ident].items())
                or row.get('record_hash') != digest({k: v for k, v in row.items() if k != 'record_hash'})):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Changed fingerprint, identity, or completed record seal')
        found[ident] = row
    return found


def may_start(run, identity, done):
    if identity['identity'] in done:
        return False
    events = read_jsonl(run/'private/calls.jsonl')
    starts = [e for e in events if e.get('event') == 'started' and e.get('identity') == identity['identity']]
    if len(starts) >= 2:
        raise Blocked('PARTIAL', 'Two-start allowance exhausted for immutable identity')
    all_starts = {e['call_id'] for e in events if e.get('event') == 'started'}
    all_completed = {e['call_id'] for e in events if e.get('event') == 'completed'}
    if len(all_starts-all_completed) >= 6:
        raise Blocked('PARTIAL', 'Six failed/interrupted requests exhausted')
    return True


def recover_atomic_completions(run, done):
    """A durable completed record is never rerun if interruption preceded its terminal journal write."""
    events = read_jsonl(run/'private/calls.jsonl')
    starts = {r['call_id']: r for r in events if r['event'] == 'started'}
    completed = {r['call_id'] for r in events if r['event'] == 'completed'}
    recovered = 0
    for row in done.values():
        if row['call_id'] in completed:
            continue
        if row['call_id'] not in starts or row.get('model_invocation_ordinal', 0) < 1:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Atomic record has no recorded model invocation start')
        append_json(run/'private/calls.jsonl', {**starts[row['call_id']], 'event': 'completed', 'at': utc_now(),
            'record_hash': row['record_hash'], 'model_invocations': 1,
            'recovered_from_atomic_completed_record': True})
        recovered += 1
    return recovered

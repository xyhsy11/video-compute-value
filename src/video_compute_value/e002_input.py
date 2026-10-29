"""Label-independent candidate inventory, grouped splits, and immutable call manifest."""
from __future__ import annotations

from pathlib import Path

from .common import Blocked, digest, file_hash, read_json, write_json
from .e001_input import actions
from .e002_model import group_folds, ordered_groups
from .video import uniform_indices

ANNOTATION_REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
CONDITIONS = ('base_plain', 'base_observed', 'quarter1', 'quarter2', 'quarter3', 'quarter4',
              'uniform_addition', 'direct16')


def menu(video):
    result = actions(video)
    direct = uniform_indices(video['timestamps'], video['duration'], 16)
    return {'base_plain': result['base'], 'base_observed': result['base'],
            **{k: v for k, v in result.items() if k != 'base'},
            'direct16': {'indices': direct, 'timestamps': [video['timestamps'][i] for i in direct]}}


def inventory(root, run):
    """Reads labels neither for ordering nor eligibility. No network or video decoding."""
    import pyarrow.parquet as pq
    root, run = Path(root), Path(run)
    legacy = read_json(root/'results/e000_20261028_greedy_collection_v3/cohort_manifest.json')
    exposed = {v['source_id'] for v in legacy['videos']}
    annotation = root/f'data/annotations/{ANNOTATION_REVISION}.parquet'
    rows = pq.read_table(annotation, columns=['videoID', 'video_id', 'duration', 'url']).to_pylist()
    grouped = {}
    for row in rows:
        if row['duration'] == 'short':
            grouped.setdefault(row['videoID'], []).append(row)
    archive = read_json(root/'data/archive_index.json')
    if archive['revision'] != ANNOTATION_REVISION:
        raise Blocked('BLOCKED_INPUT', 'Official archive metadata revision changed')
    local = {p.stem: p for p in (root/'data/videos').glob('*.mp4')}
    log = []
    for source in ordered_groups(grouped, 'candidate'):
        question_rows = grouped[source]
        reason = ('previously_exposed' if source in exposed else
                  'not_three_original_questions' if len(question_rows) != 3 else
                  'local_candidate_requires_provenance_and_CFR_checks' if source in local else 'not_local')
        member = archive['members'].get(source)
        log.append({'source_id': source, 'reason': reason,
                    'parent_ids': sorted({r['video_id'] for r in question_rows}),
                    'urls': sorted({r['url'] for r in question_rows}),
                    'local_sha256': file_hash(local[source]) if source in local else None,
                    'official_member': member})
    candidates = [r for r in log if r['source_id'] not in exposed and r['reason'] != 'not_three_original_questions']
    first = candidates[:40]
    compressed = [r['official_member']['compressed_size'] for r in first if r['official_member']]
    summary = {'annotation_revision': ANNOTATION_REVISION, 'annotation_sha256': file_hash(annotation),
        'official_archive_index_sha256': file_hash(root/'data/archive_index.json'),
        'short_source_identifiers': len(grouped), 'excluded_exposed_sources': len(exposed),
        'ordered_candidate_identifiers': len(candidates), 'local_legacy_sources': len(exposed & set(local)),
        'local_unexposed_candidates': len((set(grouped)-exposed) & set(local)),
        'eligible_verified_new_source_groups': 0, 'frozen_source_groups': 0, 'planned_source_groups': 40,
        'planned_questions': 120, 'first_40_metadata_members': len(compressed),
        'first_40_compressed_payload_bytes': sum(compressed),
        'traffic_caveat': 'Metadata payload estimate excludes HTTP range overhead, retries, and eligibility replacements.',
        'provenance': 'Source identifiers and asset hashes do not exclude all alternate encodings or reuploads.',
        'status': 'BLOCKED_INPUT'}
    write_json(run/'private/candidate_inventory.json', log)
    write_json(run/'cohort_summary.json', summary)
    write_json(run/'split_summary.json', {'status': 'NOT_FROZEN', 'outer_folds': 5,
        'planned_groups_per_outer_fold': 8, 'inner_folds': 4, 'actual_groups_per_fold': None})
    return summary


def split_manifest(videos):
    if len(videos) != 40 or len({v['source_id'] for v in videos}) != 40:
        raise Blocked('BLOCKED_INPUT', 'Exactly 40 new verified source groups are required')
    groups = [v['source_id'] for v in videos]
    folds = group_folds(groups)
    result = {'outer': {g: int(f) for g, f in zip(groups, folds)}, 'inner': {}, 'repeat_sources': []}
    for fold in range(5):
        train = [g for g, f in zip(groups, folds) if f != fold]
        inner = group_folds(train, 4, f'inner:{fold}')
        result['inner'][str(fold)] = {g: int(f) for g, f in zip(train, inner)}
        test = [g for g, f in zip(groups, folds) if f == fold]
        result['repeat_sources'].append(ordered_groups(test, f'repeat:{fold}')[0])
    return result


def call_manifest(videos, splits, fingerprint):
    if any(len(v['questions']) != 3 for v in videos):
        raise Blocked('BLOCKED_INPUT', 'Every source must retain all three questions')
    calls = []
    for repetition in (1, 2):
        for video in videos:
            if repetition == 2 and video['source_id'] not in splits['repeat_sources']:
                continue
            for question in video['questions']:
                for condition in CONDITIONS:
                    row = {'fingerprint': fingerprint, 'source_id': video['source_id'],
                           'question_id': str(question['question_id']), 'condition': condition,
                           'repetition': repetition}
                    calls.append({**row, 'identity': digest(row)})
    calls.sort(key=lambda r: (r['repetition'], digest({'seed': 2026102902, 'identity': r['identity']})))
    if len(calls) != 1080 or len({r['identity'] for r in calls}) != 1080:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Manifest is not 960 primary + 120 repeat calls')
    return calls

"""Outcome-independent E2 source preparation with bounded, journaled ZIP ranges."""
from __future__ import annotations

from fractions import Fraction
import time
from urllib.parse import quote, urlparse, parse_qs
import zipfile

from .acquire import RangeReader, request
from .common import Blocked, append_json, digest, file_hash, read_json, utc_now, write_json
from .dataset import load_annotations
from .e002_input import ANNOTATION_REVISION, menu
from .e002_model import ordered_groups
from .video import inspect_video

OLD = 'results/e002_20261029_073134_v1'


def candidate_plan(root, run):
    prior = read_json(root/OLD/'private/candidate_inventory.json')
    candidates = [r for r in prior if r['reason'] != 'previously_exposed'][:60]
    assert len(candidates) == 60
    assert [r['source_id'] for r in candidates] == ordered_groups([r['source_id'] for r in prior
        if r['reason'] != 'previously_exposed'], 'candidate')[:60]
    sources = ['src/video_compute_value/'+n for n in ('e002v2_data.py', 'e002v2_account.py',
               'acquire.py', 'video.py', 'e001_input.py', 'e002_input.py', 'e002_features.py')]
    value = {'annotation_revision': ANNOTATION_REVISION, 'candidate_count': 60, 'required_eligible': 40,
             'candidates_sha256': digest(candidates), 'source_files': {p: file_hash(root/p) for p in sources},
             'source_order': [c['source_id'] for c in candidates],
             'rule': 'frozen candidate order; unchanged CFR, spatial layout, distinct additions and provenance'}
    value['fingerprint'] = digest(value)
    frozen = run/'preparation_freeze.json'
    if frozen.exists() and read_json(frozen) != value:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Preparation fingerprint changed')
    write_json(frozen, value)
    write_json(run/'private/candidates.json', candidates)
    return candidates


class JournalRangeReader(RangeReader):
    def __init__(self, url, size, budget):
        super().__init__(url, size, budget)
        self.budget = budget

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size <= 0:
            return b''
        if size > 8*1024**2:
            raise Blocked('BLOCKED_RESOURCE', 'Unexpected ZIP range size')
        reservation = size+4096
        self.budget.allow_video_transfer(reservation)
        self.budget.check(projected_seconds=600, projected_bytes=size)
        self.budget.value['pending_video_transfer_bytes'] = reservation
        self.budget.save()
        start, end = self.position, self.position+size-1
        event = {'at': utc_now(), 'range_start': start, 'range_end': end, 'payload_bytes': size,
                 'reserved_bytes': reservation, 'url_sha256': digest(self.url)}
        append_json(self.budget.run/'private/transfer_events.jsonl', {**event, 'event': 'started'})
        try:
            with request(self.url, {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                    raise Blocked('BLOCKED_INPUT', 'Server did not honor exact public byte range')
                content = response.read(size+1)
            if len(content) != size:
                raise Blocked('BLOCKED_INPUT', 'Incomplete ZIP range')
        except BaseException:
            self.budget.value.pop('pending_video_transfer_bytes', None)
            self.budget.value['interrupted_transfer_upper_bound_bytes'] = (
                self.budget.value.get('interrupted_transfer_upper_bound_bytes', 0)+reservation)
            self.budget.save()
            append_json(self.budget.run/'private/transfer_events.jsonl', {**event, 'event': 'failed_conservative_charge'})
            raise
        self.budget.value.pop('pending_video_transfer_bytes', None)
        self.budget.charge_transfer(reservation, video=True)
        append_json(self.budget.run/'private/transfer_events.jsonl', {**event, 'event': 'completed',
                    'charged_payload_and_overhead_bytes': reservation})
        self.position += size
        return content


def acquisition(root, run, budget, candidate):
    source, member = candidate['source_id'], candidate['official_member']
    if not member:
        raise Blocked('BLOCKED_INPUT', 'No official archive member')
    target = root/'data/videos'/f'{source}.mp4'
    receipts = run/'private/acquisition'
    receipt = receipts/f'{source}.json'
    if receipt.exists():
        saved = read_json(receipt)
        if saved['status'] == 'completed':
            if not target.exists() or file_hash(target) != saved['sha256']:
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Acquired source changed')
            return target, saved
        raise Blocked('BLOCKED_INPUT', 'Preserved failed acquisition requires explicit resume diagnosis')
    if target.exists():
        # Assets already acquired by this project may be reused, never counted as newly downloaded.
        if target.stat().st_size != member['size']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Existing source size differs from official metadata')
        saved = {'status': 'completed', 'source_id': source, 'sha256': file_hash(target),
                 'size': target.stat().st_size, 'new_transfer_bytes': 0, 'elapsed_seconds': 0., 'reused_local': True}
        write_json(receipt, saved)
        return target, saved
    budget.allow_video_transfer(member['compressed_size']+2*1024**2)
    budget.check(projected_seconds=600, projected_bytes=member['size'])
    legacy = read_json(root/'results/e000_20261028_greedy_collection_v3/cohort_manifest.json')
    repo = legacy['dataset']['repo']
    url = f"https://huggingface.co/datasets/{repo}/resolve/{ANNOTATION_REVISION}/{quote(member['archive']['path'])}"
    attempt = run/'private/acquisition_partials'/f'{source}_{time.time_ns()}.mp4.partial'
    attempt.parent.mkdir(parents=True, exist_ok=True)
    started, before = time.perf_counter(), budget.value['video_transfer_bytes']
    write_json(receipt, {'status': 'started', 'source_id': source, 'at': utc_now(),
                        'member_metadata_sha256': digest(member), 'partial_path': str(attempt)})
    try:
        with JournalRangeReader(url, member['archive']['size'], budget) as stream, zipfile.ZipFile(stream) as archive:
            with archive.open(member['member']) as data, attempt.open('wb') as output:
                total = 0
                while chunk := data.read(1024**2):
                    total += len(chunk)
                    if total > member['size']:
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Extracted source exceeds official size')
                    output.write(chunk)
        if attempt.stat().st_size != member['size']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Extracted source is incomplete')
        attempt.replace(target)
        saved = {'status': 'completed', 'source_id': source, 'sha256': file_hash(target),
                 'size': target.stat().st_size, 'new_transfer_bytes': budget.value['video_transfer_bytes']-before,
                 'elapsed_seconds': time.perf_counter()-started, 'official_member_sha256': digest(member)}
        write_json(receipt, saved)
        return target, saved
    except BaseException as exc:
        write_json(receipt, {'status': 'failed', 'source_id': source, 'error_type': type(exc).__name__,
            'message': str(exc), 'elapsed_seconds': time.perf_counter()-started,
            'new_transfer_bytes': budget.value['video_transfer_bytes']-before, 'partial_path': str(attempt)})
        raise


def source_url_id(url):
    value = urlparse(url)
    if value.netloc in ('www.youtube.com', 'youtube.com'):
        return parse_qs(value.query).get('v', [None])[0]
    if value.netloc == 'youtu.be':
        return value.path.strip('/')
    return None


def inspect_source(root, run, candidate, path, selected):
    started = time.perf_counter()
    legacy = read_json(root/'results/e000_20261028_greedy_collection_v3/cohort_manifest.json')['videos']
    if len(candidate['urls']) != 1 or source_url_id(candidate['urls'][0]) != candidate['source_id']:
        raise ValueError('Official URL/source identity mismatch or ambiguous parent')
    if len(candidate['parent_ids']) != 1:
        raise ValueError('Ambiguous official parent group')
    video = inspect_video(path, [8, 16], 65536)
    if (video['target_height'], video['target_width']) != (192, 320):
        raise ValueError('Unchanged native spatial recipe does not produce 192x320')
    if any(v['sha256'] == video['sha256'] for v in [*legacy, *selected]):
        raise ValueError('Duplicate previously exposed or selected source asset')
    if any(v.get('parent_id') == candidate['parent_ids'][0] for v in selected):
        raise ValueError('Shared official parent group')
    duration = Fraction(video['duration'])
    quarters = [int(4*Fraction(t)/duration) for t in video['selections']['8']['timestamps']]
    if [quarters.count(q) for q in range(4)] != [2]*4:
        raise ValueError('Base PTS do not provide two observations per quarter')
    menu(video)  # All distinct retained additions must be feasible before any answer.
    video.update(source_id=candidate['source_id'], parent_id=candidate['parent_ids'][0],
        provenance={'official_urls': candidate['urls'], 'distinct_original_source_id': True,
                    'known_overlap_detected': False,
                    'limitation': 'Unique official IDs/parents and hashes cannot exclude all unknown reuploads.'},
        preparation_inspection_seconds=time.perf_counter()-started)
    return video


def prepare(root, run, budget, *, probe=False):
    candidates = candidate_plan(root, run)
    progress_path = run/'private/eligibility_progress.json'
    progress = read_json(progress_path) if progress_path.exists() else {'selected': [], 'excluded': [], 'visited': []}
    remaining = [c for c in candidates if c['source_id'] not in progress['visited']]
    if probe:
        budget.admit('input_probe', 300, storage=max(c['official_member']['size'] for c in candidates[:1])+32*1024**2,
                     traffic=candidates[0]['official_member']['compressed_size']+2*1024**2,
                     detail={'purpose': 'one bounded first-candidate acquisition and eligibility measurement'})
        remaining = remaining[:1]
    else:
        first_receipt = read_json(run/'private/acquisition'/f"{candidates[0]['source_id']}.json")
        if first_receipt['status'] != 'completed' or first_receipt['elapsed_seconds'] <= 0:
            raise Blocked('BLOCKED_INPUT', 'A measured acquisition/inspection probe is required')
        payload = sum(c['official_member']['compressed_size'] for c in remaining)
        size = sum(c['official_member']['size'] for c in remaining)
        rate = max(.001, first_receipt['new_transfer_bytes']/first_receipt['elapsed_seconds'])
        raw = payload/rate + len(remaining)*max(30., progress.get('probe_inspection_seconds', 30.))
        budget.admit('input_cohort', raw, storage=size+128*1024**2,
            traffic=math_ceil_traffic(payload, len(remaining)),
            detail={'bounded_candidate_window': 60, 'required_eligible': 40,
                    'measured_probe_bytes_per_second': rate, 'remaining_candidates': len(remaining),
                    'full_window_payload_bytes': payload})
    for candidate in remaining:
        if len(progress['selected']) >= 40:
            break
        source = candidate['source_id']
        path, receipt = acquisition(root, run, budget, candidate)
        try:
            value = inspect_source(root, run, candidate, path, progress['selected'])
            progress['selected'].append(value)
            progress['probe_inspection_seconds'] = value['preparation_inspection_seconds'] if probe else progress.get('probe_inspection_seconds', 30.)
            print(f"Eligible sources: {len(progress['selected'])}/40; candidates visited {len(progress['visited'])+1}", flush=True)
        except (ValueError, RuntimeError) as exc:
            progress['excluded'].append({'source_id': source, 'reason': str(exc)})
            print(f"Label-independent exclusion: candidate {len(progress['visited'])+1}", flush=True)
        progress['visited'].append(source)
        write_json(progress_path, progress)
        budget.checkpoint('Acquisition and eligibility source '+str(len(progress['visited'])))
    if not probe:
        if len(progress['selected']) != 40:
            raise Blocked('BLOCKED_INPUT', 'Bounded candidate window did not yield exactly 40 eligible groups')
        annotation = root/f'data/annotations/{ANNOTATION_REVISION}.parquet'
        questions = load_annotations(annotation)
        for video in progress['selected']:
            video['questions'] = questions[video['source_id']]
            video['annotation_hash'] = digest(video['questions'])
        cohort = {'videos': progress['selected'], 'exclusions': progress['excluded'],
                  'annotation_revision': ANNOTATION_REVISION, 'annotation_sha256': file_hash(annotation)}
        write_json(run/'private/cohort_manifest.json', cohort)
        write_json(run/'cohort_summary.json', {'verified_source_groups': 40, 'questions': 120,
            'visited_candidates': len(progress['visited']), 'excluded_candidates': len(progress['excluded']),
            'source_independence_scope': 'Unique official source/parent IDs; no detected identical assets; unknown reuploads remain a limitation.'})
    return progress


def math_ceil_traffic(payload, count):
    # Includes directory/range overhead and a half-payload contingency; failures stop acquisition.
    import math
    return math.ceil(payload*1.5+count*1024**2)

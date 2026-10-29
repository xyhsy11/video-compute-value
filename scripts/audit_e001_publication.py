"""Read-only saved-data, figure/link, privacy and exact Git-byte audit for E1 publication."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import statistics
import subprocess

from PIL import Image
import yaml

from video_compute_value.common import file_hash, read_json, write_json
from video_compute_value.protocol import parse_answer, trial_identity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT/'results/e001_20261029_132707_v1'


def e0_saved_record_audit():
    parent = ROOT/'results/e000_20261028_greedy_collection_v3'
    frozen = yaml.safe_load((parent/'resolved_config.yaml').read_text(encoding='utf-8'))
    cohort, public = read_json(parent/'cohort_manifest.json'), read_json(parent/'public_summary.json')
    labels = {(v['source_id'], str(q['question_id'])): q['answer'] for v in cohort['videos'] for q in v['questions']}
    rows = [read_json(p) for p in (parent/'records').glob('*.json')]
    assert len(rows) == 384
    identities = set()
    for row in rows:
        expected = trial_identity(parent.name, frozen['config_hash'], row['source_id'], row['question_id'],
                                  row['budget'], row['repetition'])
        assert all(row[k] == value for k, value in expected.items())
        assert row['trial_id'] not in identities
        identities.add(row['trial_id'])
        parsed = parse_answer(row['decoded_output'])
        assert parsed == row['parsed_answer']
        assert row['correct'] == (parsed == labels[(row['source_id'], str(row['question_id']))])
    for condition in public['conditions']:
        selected = [r for r in rows if r['budget'] == condition['budget']]
        primary = [r for r in selected if r['repetition'] == 1]
        assert len(primary) == 24
        assert sum(r['correct'] for r in primary) == condition['rep1_correct_out_of_24']
        assert sum(r['correct'] for r in primary)/24 == condition['accuracy']
        by_question = {}
        for row in selected:
            by_question.setdefault((row['source_id'], row['question_id']), []).append(row)
        medians = [statistics.median(r['latency_seconds'] for r in group) for group in by_question.values()]
        assert statistics.median(medians) == condition['latency_seconds']['median']
        assert max(r['peak_allocated_bytes'] for r in selected) == condition['peak_allocated_bytes']
        assert max(r['peak_reserved_bytes'] for r in selected) == condition['peak_reserved_bytes']
        if condition['budget']:
            ranges = [(max(r['latency_seconds'] for r in group)-min(r['latency_seconds'] for r in group))/
                      statistics.median(r['latency_seconds'] for r in group) for group in by_question.values()]
            assert statistics.median(ranges) == condition['median_relative_timing_range']
    return len(rows)


def link_audit(paths):
    checked, failures = 0, []
    for path in paths:
        text = path.read_text(encoding='utf-8')
        for match in re.finditer(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text):
            target = match.group(1).strip('<>')
            if re.match(r'^(https?://|#|mailto:)', target):
                continue
            checked += 1
            if not (path.parent/target.split('#')[0]).exists():
                failures.append(f'{path.relative_to(ROOT)} -> {target}')
    assert not failures, failures
    return checked


def figure_audit(run):
    manifest = read_json(run/'figure_manifest.json')
    assert len(manifest) == 6
    for name, sha in manifest.items():
        path = run/'figures'/name
        assert file_hash(path) == sha
        with Image.open(path) as image:
            assert min(image.size) > 500
            image.verify()
    summary = read_json(run/'summary.json')
    with (run/'metrics.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            assert float(row['accuracy']) == summary['metrics'][row['metric']]
            assert [float(row['ci95_low']), float(row['ci95_high'])] == summary['descriptive_95pct_cluster_bootstrap'][row['metric']]
    with (run/'correctness.csv').open(encoding='utf-8', newline='') as handle:
        matrix = list(csv.reader(handle))
    assert len(matrix) == 25 and len(matrix[0]) == 7
    assert all(re.fullmatch(r'V0[1-8]Q[1-3]', row[0]) and set(row[1:]) <= {'0', '1'} for row in matrix[1:])
    best = max(range(1, 5), key=lambda a: summary['metrics'][f'quarter{a}'])
    with (run/'per_video_effects.csv').open(encoding='utf-8', newline='') as handle:
        effects = list(csv.DictReader(handle))
    assert len(effects) == 8
    for index, effect in enumerate(effects):
        rows = [[int(v) for v in row[1:]] for row in matrix[1+index*3:4+index*3]]
        oracle = sum(max(r[1:5]) for r in rows)/3
        random = sum(sum(r[1:5]) for r in rows)/12
        fixed = sum(r[best] for r in rows)/3
        assert abs(float(effect['oracle_minus_random_pp'])-(oracle-random)*100) < 1e-10
        assert abs(float(effect['oracle_minus_global_best_fixed_pp'])-(oracle-fixed)*100) < 1e-10
    return 6


def git_audit(run, revision):
    if revision == 'index':
        changed = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
        prefix = ':'
    else:
        changed = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', revision],
                                          cwd=ROOT, text=True).splitlines()
        prefix = revision+':'
    allowed = {'.gitattributes', 'AGENTS.md', 'README.md', 'configs/e001.yaml', 'protocols/e001_additional_clip_headroom_v1.md',
        'docs/e0_reproduction.md', 'tests/test_e001.py', 'scripts/report_e001.py', 'scripts/audit_e001_publication.py',
        'src/video_compute_value/e001.py', 'src/video_compute_value/e001_input.py',
        'src/video_compute_value/e001_analysis.py', 'src/video_compute_value/e001_verify.py'}
    public_files = {'freeze.json', 'runtime_resource_grant.json', 'status.json', 'forecast.json',
        'resource_accounting.json', 'summary.json', 'verification.json', 'metrics.csv', 'correctness.csv',
        'per_video_effects.csv', 'e0_plot_data.csv', 'figure_manifest.json', 'report.md',
        'validation.json', 'publication_audit.json', 'preflight_verification.json', 'publication_sources.json',
        'augmented_text_token_differences.csv'}
    result_prefix = run.relative_to(ROOT).as_posix()+'/'
    for name in changed:
        relative = name.removeprefix(result_prefix)
        assert name in allowed or (name.startswith(result_prefix) and
            (relative in public_files or relative.startswith('figures/') and relative.endswith('.png'))), name
        assert not re.search(r'(^|/)(private|records|data|models|\.venv|\.cache)(/|$)', name), name
        raw = subprocess.check_output(['git', 'show', prefix+name], cwd=ROOT)
        assert len(raw) < 2*1024**2, name
        if not name.endswith('.png'):
            text = raw.decode('utf-8')
            assert not re.search(r'C:[\\/]+Users[\\/]+[\w.-]+[\\/]|/(?:home|Users)/[\w.-]+/|'
                                 r'-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}|hf_[A-Za-z0-9]{25,}', text), name
            if name.startswith(result_prefix) and name.endswith('.json'):
                assert not re.search(r'"(?:decoded_output|parsed_answer|generated_token_ids|answer|options|question)"\s*:', text), name
    frozen = read_json(run/'private/frozen.json')
    for name, sha in frozen['source_files'].items():
        raw = subprocess.check_output(['git', 'show', prefix+name], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == sha, f'Executed/Git scientific bytes differ: {name}'
    provenance = run/'publication_sources.json'
    if provenance.exists():
        provenance = read_json(provenance)
        files = {**provenance['source_files_sha256'],
                 **{result_prefix+k: v for k, v in provenance['public_data_files_sha256'].items()},
                 **{result_prefix+'figures/'+k: v for k, v in provenance['figure_files_sha256'].items()}}
        for name, sha in files.items():
            raw = subprocess.check_output(['git', 'show', prefix+name], cwd=ROOT)
            assert hashlib.sha256(raw).hexdigest() == sha, f'Publication/Git byte mismatch: {name}'
    assert subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip() == 'https://github.com/xyhsy11/video-compute-value.git'
    return len(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--revision', help='index for staged blobs, or an actual committed Git revision')
    parser.add_argument('--check-only', action='store_true', help='Inspect without modifying the saved receipt')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    records = e0_saved_record_audit()
    links = link_audit([ROOT/'README.md', ROOT/'docs/e0_reproduction.md',
                        ROOT/'protocols/e001_additional_clip_headroom_v1.md', run/'report.md'])
    figures = figure_audit(run)
    verification = read_json(run/'verification.json')
    assert read_json(run/'validation.json')['tests_sha256'] == file_hash(ROOT/'tests/test_e001.py')
    assert verification['statistics_independently_verified']
    assert verification['summary_sha256'] == file_hash(run/'summary.json')
    required = ('complete_288_identities', 'native_input_integrity', 'exact_base_retention',
                'matched_augmented_visual_budgets', 'effective_greedy_generation',
                'independent_token_decoding_and_scoring', 'fresh_process_repeatability',
                'e0_historical_preservation', 'preflight_passed')
    assert all(verification[k] for k in required)
    preserved = read_json(run/'private/historical_preservation.json')
    assert all(file_hash(ROOT/name) == sha for name, sha in preserved.items())
    resource, ledger = read_json(run/'resource_accounting.json'), read_json(run/'private/resource_ledger.json')
    charged = sum(s['active_seconds'] for s in ledger['sessions']) + sum(e['active_seconds'] for e in ledger['external_events'])
    assert abs(charged-ledger['active_seconds']) < .00001
    assert resource['cumulative_seconds'] == ledger['active_seconds'] <= 27600
    assert resource['remaining_seconds'] == 27600-ledger['active_seconds']
    assert all(s.get('finished_at') for s in ledger['sessions'])
    assert resource['storage_current_bytes'] <= resource['storage_limit_bytes'] == 20*1024**3
    assert resource['video_transfer_and_interruption_upper_bound_bytes'] <= 1024**3
    assert resource['completed_e1_model_invocations'] == verification['completed_calls']
    from video_compute_value.e001_input import actions
    from video_compute_value.protocol import video_input
    cohort = read_json(run/'private/cohort_manifest.json')
    menu = read_json(run/'private/action_manifest.json')
    assert all(actions(video_input(v)) == menu[v['source_id']] for v in cohort['videos'])
    a, b = [read_json(run/f'private/probe_receipt_{i}.json') for i in (0, 1)]
    for first, second in zip(a['rows'], b['rows']):
        for key in ('input_tensor_hashes', 'input_metadata_hash', 'generated_token_ids',
                    'decoded_output', 'parsed_answer', 'generation_policy_sha256', 'processed_rgb_hashes'):
            assert first[key] == second[key]
        assert first['peak_allocated_bytes'] > 0
    if args.revision:
        changed = git_audit(run, args.revision)
    else:
        changed = None
    receipt = {'passed': True, 'independently_recounted_e0_records': records,
               'local_markdown_links_checked': links, 'figures_checked': figures,
               'figure_values_agree_with_saved_sources': True,
               'e1_saved_record_verification_passed': verification['all_checks_passed'],
               'actions_independently_regenerated_from_label_free_metadata': True,
               'cumulative_resources_independently_reconciled': True,
               'e0_historical_files_rehashed': len(preserved),
               'scientific_git_bytes_checked': args.revision is not None,
               'git_revision_or_index': args.revision, 'explicit_task_paths_audited': changed,
               'raw_data_and_outputs_remain_private': True}
    if not args.check_only:
        write_json(run/'publication_audit.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()

"""Regenerate public research figures and the E1 report from saved, verified data only."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from video_compute_value.common import atomic_text, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
E0 = ROOT/'results/e000_20261028_greedy_collection_v3'
DEFAULT_RUN = ROOT/'results/e001_20261029_132707_v1'


def export_csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def e0_figures(run):
    summary = read_json(E0/'public_summary.json')
    conditions = sorted((c for c in summary['conditions'] if c['budget']), key=lambda c: c['budget'])
    rows = [{'frames': c['budget'], 'accuracy': c['accuracy'],
        'median_request_seconds': c['latency_seconds']['median'],
        'median_relative_timing_range': c['median_relative_timing_range'],
        'peak_allocated_gib': c['peak_allocated_bytes']/1024**3,
        'peak_reserved_gib': c['peak_reserved_bytes']/1024**3} for c in conditions]
    export_csv(run/'e0_plot_data.csv', list(rows[0]), rows)
    budgets = [r['frames'] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout='constrained')
    ax.plot(budgets, [r['accuracy']*100 for r in rows], 'o-', color='#22698e')
    ax.set(xlabel='Uniformly selected frames per request', ylabel='Accuracy (%)', ylim=(0, 100),
           xticks=budgets, title='E0: first-pass accuracy on 24 exposed questions')
    for r in rows:
        ax.annotate(f"{r['accuracy']*100:.1f}", (r['frames'], r['accuracy']*100), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=9)
    fig.savefig(run/'figures/e0_accuracy.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout='constrained')
    ax.plot(budgets, [r['median_request_seconds'] for r in rows], 'o-', color='#a74c3a')
    ax.set(xlabel='Uniformly selected frames per request', ylabel='Median full request (seconds)',
           xticks=budgets, ylim=(0, None), title='E0: descriptive latency — MEASUREMENT_UNSTABLE')
    fig.savefig(run/'figures/e0_latency.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout='constrained')
    ax.plot(budgets, [r['peak_allocated_gib'] for r in rows], 'o-', label='Allocated (includes weights)')
    ax.plot(budgets, [r['peak_reserved_gib'] for r in rows], 's--', label='Reserved by allocator')
    ax.set(xlabel='Uniformly selected frames per request', ylabel='Absolute peak PyTorch memory (GiB)',
           xticks=budgets, ylim=(0, None), title='E0: memory across 360 video requests')
    ax.legend(fontsize=9)
    fig.savefig(run/'figures/e0_memory.png', dpi=170)
    plt.close(fig)
    return rows


def e1_figures(run, summary):
    metrics = summary['metrics']
    token_rows = read_json(run/'preflight_verification.json')['augmented_text_token_differences']
    export_csv(run/'augmented_text_token_differences.csv', list(token_rows[0]), token_rows)
    names = ['base', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition',
             'random', 'best_fixed', 'oracle', 'oracle_stop']
    labels = ['Base', 'Q1', 'Q2', 'Q3', 'Q4', 'Uniform\naddition', 'Expected\nrandom',
              'Best\nfixed*', 'Oracle*', 'Oracle +\nstop*']
    values = [100*metrics[k] for k in names]
    fig, ax = plt.subplots(figsize=(11, 5.2), layout='constrained')
    ax.bar(range(len(names)), values,
           color=['#337c99']*6+['#7b8187']+['#c18538']*3, width=.72)
    for i, value in enumerate(values):
        ax.text(i, value+1.2, f'{value:.1f}', ha='center', fontsize=9)
    ax.set(xticks=range(len(names)), xticklabels=labels, ylabel='First-pass accuracy (%)', ylim=(0, 106),
           title='E1: retained evidence plus an additional clip (24 questions, 8 exposed videos)')
    ax.tick_params(axis='x', labelsize=9)
    ax.legend(handles=[Patch(color='#337c99', label='Executed fixed input'),
        Patch(color='#7b8187', label='Exact expected random choice'),
        Patch(color='#c18538', label='* Retrospective comparator')],
        loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3, fontsize=8)
    fig.savefig(run/'figures/e1_accuracy.png', dpi=180)
    plt.close(fig)
    with (run/'correctness.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.reader(handle))
    matrix = np.asarray([[int(v) for v in r[1:]] for r in rows[1:]])
    fig, ax = plt.subplots(figsize=(8.5, 8.6), layout='constrained')
    ax.imshow(matrix, cmap=ListedColormap(['#e5b2a7', '#70a994']), vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=range(6), xticklabels=['Base (8)', 'Q1 (16)', 'Q2 (16)', 'Q3 (16)', 'Q4 (16)', 'Uniform + (16)'],
           yticks=range(24), yticklabels=[r[0] for r in rows[1:]],
           title='E1: anonymous question correctness, first pass\n24 questions grouped into 8 source videos')
    ax.tick_params(axis='both', labelsize=9)
    for y in np.arange(2.5, 23, 3):
        ax.axhline(y, color='white', linewidth=3)
    ax.legend(handles=[Patch(color='#70a994', label='Correct'),
                       Patch(color='#e5b2a7', label='Incorrect or invalid')],
              loc='upper center', bbox_to_anchor=(.5, -.05), ncol=2, fontsize=9)
    fig.savefig(run/'figures/e1_correctness.png', dpi=180)
    plt.close(fig)
    # Decompose the cohort-level best fixed comparator using its globally selected
    # quarter (earliest quarter breaks exact ties); also export within-video maxima.
    fixed = max(range(1, 5), key=lambda a: metrics[f'quarter{a}'])
    per_video = [{'video': row['video'],
        'oracle_minus_random_pp': row['oracle_minus_random']*100,
        'oracle_minus_global_best_fixed_pp': (row['oracle']-row[f'quarter{fixed}'])*100,
        'oracle_minus_local_best_fixed_pp': row['oracle_minus_best_fixed']*100,
        'global_best_fixed_quarter': fixed} for row in summary['per_video']]
    export_csv(run/'per_video_effects.csv', list(per_video[0]), per_video)
    fig, ax = plt.subplots(figsize=(8.6, 5.2), layout='constrained')
    y = np.arange(8)
    ax.scatter([r['oracle_minus_random_pp'] for r in per_video], y-.12,
               marker='o', s=48, label='Oracle − expected random')
    ax.scatter([r['oracle_minus_global_best_fixed_pp'] for r in per_video], y+.12,
               marker='s', s=42, label=f'Oracle − global best fixed (Q{fixed})')
    ax.axvline(0, color='#777777', linewidth=1)
    ax.set(yticks=y, yticklabels=[r['video'] for r in per_video],
           xlabel='Paired accuracy difference (percentage points)',
           title='E1: effects for all 8 videos (3 questions per video)')
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=1, fontsize=9)
    fig.savefig(run/'figures/e1_per_video.png', dpi=180)
    plt.close(fig)


def report(run, summary, e0):
    m, status, checked = summary['metrics'], read_json(run/'status.json'), read_json(run/'verification.json')
    ci = summary['descriptive_95pct_cluster_bootstrap']
    local_ties = sum(abs(row['oracle_minus_best_fixed']) < 1e-12 for row in summary['per_video'])
    loo = [row['oracle_minus_best_fixed']*100 for row in summary['leave_one_video_out']]
    lines = ['# Experiment 1 — additional-clip headroom', '', f"**{status['verdict']}**", '',
        'This is an accuracy-only diagnostic on eight exposed Video-MME short videos and all',
        '24 original questions from the official annotation test split. They are development',
        'data. E0 remains **MEASUREMENT_UNSTABLE**. E1 does not validate latency savings.', '',
        '## Design and completed evidence', '',
        'We retain the exact Uniform-8 source frames, original timestamps and RGB pixels.',
        'Four acquisitions add eight distinct frames from a fixed quarter; uniform addition',
        'adds eight frames across the original timeline. All augmented conditions contain',
        '16 frames and matched visual tokens at the unchanged 192 × 320 resolution.',
        'Uniform addition shares the base target times, so its additions are the nearest',
        'unused neighbors of base frames; comparisons apply to this specific nested control.',
        'Timestamp formatting can change text-token counts at matched visual budgets;',
        '[the per-video token table](augmented_text_token_differences.csv) records these differences.',
        'This is full native-video re-encoding; temporal grouping and contextual representations can change.', '',
        'The frozen Qwen3-VL-4B-Instruct model/processor/tokenizer revision is',
        '`ebb281ec70b05090aa6165b016eac8ec08e71b17` (BF16, SDPA, visual-only).',
        'The prompt/parser and explicit effective greedy policy are inherited from corrected E0.',
        'See the [prospective protocol](../../protocols/e001_additional_clip_headroom_v1.md)',
        'and [timestamped freeze](freeze.json). Labels are used only after inference.', '',
        f"Collection: **{status['completed_collection_requests']}/288** requests: two fresh-process passes",
        'of all 24 questions × six conditions. Pass 1 is primary; pass 2 is a repeatability check.',
        f"Exact repeatability: **{checked['repeatable_cells']}/144** cells for inputs, tokens, raw outputs and answers.",
        f"Completed calls by kind: {checked['completed_calls_by_kind']}. Failed calls: {checked['failed_calls']};",
        f"unknown interrupted calls (upper bound): {checked['interrupted_calls_upper_bound']}.",
        'Warmups and probes do not enter accuracy. Completed invalid answers score zero;',
        'runtime failures would be missing measurements. All 24 questions remain in the denominator.', '',
        '## Accuracy and headroom', '', '| Condition/statistic | Correct / denominator | Accuracy |',
        '|---|---:|---:|']
    for name in ('base', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition'):
        lines.append(f"| {name} | {summary['correct_counts'][name]}/24 | {m[name]*100:.2f}% |")
    for name, count in (('random', f"{summary['random_correct_action_pairs']}/96 action pairs"),
                        ('best_fixed', f"{max(summary['correct_counts'][f'quarter{i}'] for i in range(1, 5))}/24"),
                        ('oracle', f"{summary['oracle_correct_count']}/24"),
                        ('oracle_stop', f"{summary['oracle_stop_correct_count']}/24")):
        lines.append(f'| {name} (retrospective/expected) | {count} | {m[name]*100:.2f}% |')
    lines += ['', '| Paired contrast | Difference (pp) | Descriptive 95% cluster-bootstrap interval (pp) |',
              '|---|---:|---:|']
    for name in ('oracle_minus_random', 'oracle_minus_best_fixed', 'oracle_minus_uniform', 'oracle_minus_base'):
        lines.append(f'| {name} | {m[name]*100:.2f} | [{ci[name][0]*100:.2f}, {ci[name][1]*100:.2f}] |')
    visual = read_json(run/'preflight_verification.json')['visual_tokens_by_condition']
    lines += ['', 'Descriptive request observations below use the 24 first-pass requests per',
              'condition. All 288 individual timings/memory records are retained privately.',
              'These times do not pass or repair E0’s timing-stability gate. Peaks include weights.', '',
              '| Condition | Visual tokens | Input-token range | Median request (s) | Peak allocated / reserved (GiB) |',
              '|---|---:|---:|---:|---:|']
    for name in ('base', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition'):
        memory = summary['absolute_peak_memory_bytes'][name]
        lo, hi = summary['input_token_ranges'][name]
        lines.append(f"| {name} | {visual[name][0]} | {lo}–{hi} | {summary['descriptive_primary_median_request_seconds'][name]:.3f} | {memory['allocated']/1024**3:.3f} / {memory['reserved']/1024**3:.3f} |")
    lines += ['', 'The forced-acquisition oracle chooses one of four quarters, excluding uniform addition',
        'and stopping. Random is the exact expected result over all four choices. Best fixed',
        'is the strongest single quarter for this cohort, selected retrospectively. Oracle plus',
        'stop may keep the base answer, so its gain is nonnegative by construction.', '',
        f"Base-wrong questions rescued by at least one quarter: {summary['rescued_base_wrong_questions']}.",
        f"Base-correct questions harmed by every quarter: {summary['base_correct_harmed_by_every_clip']}.",
        f"Average action-induced harm: {summary['harmed_base_correct_action_pairs']}/96 question/action pairs",
        f"({summary['average_action_induced_harm']*100:.2f}%). Mixed successful/unsuccessful choices:",
        f"{summary['mixed_success_questions']} questions across {summary['mixed_success_videos']} videos.",
        f"Invalid answers by condition: {summary['invalid_answers']}.", '',
        f"The frozen exploratory advancement rule **{'passed' if status['advancement_rule_passed'] else 'did not pass'}**.",
        'It requires both oracle-minus-random and oracle-minus-best-fixed ≥10 pp, oracle',
        'strictly above base and uniform addition, mixed choices in ≥3 videos, and every',
        'integrity/collection/greedy/repeatability/independent-verification check. These are',
        'engineering thresholds, not established statistical standards.', '',
        'Bootstrap intervals use 10,000 resamples of complete source-video clusters, seed',
        '20261029; all questions/actions stay together, and best fixed is recomputed in',
        'each resample. Eight clusters provide weak uncertainty estimates. Repetitions',
        'and clip actions are not additional independent examples. No significance claim.', '',
        '## Figures and saved sources', '',
        '![Accuracy by executed condition and retrospective comparator](figures/e1_accuracy.png)', '',
        'Figure 1. First-pass accuracy on 24 questions from eight development videos. Orange',
        'bars are retrospective comparators; enumeration requires real additional model calls.',
        'Source: [metrics.csv](metrics.csv), [summary.json](summary.json).', '',
        '![Anonymous correctness matrix](figures/e1_correctness.png)', '',
        'Figure 2. All 24 first-pass questions, grouped into eight source videos, against six',
        'executed conditions. Anonymous IDs contain no questions, labels or model answers.',
        'Source: [correctness.csv](correctness.csv).', '',
        '![All video-level paired effects](figures/e1_per_video.png)', '',
        'Figure 3. Paired effects for every source (three questions each). Best fixed uses the',
        'same globally selected quarter across videos; within-video best-fixed contrasts are',
        'also exported and explicitly distinguished. Sources: [per_video_effects.csv](per_video_effects.csv)',
        'and the per-video rows in [summary.json](summary.json).', '',
        'Recreate every figure and this report:', '', '```powershell',
        f'.venv\\Scripts\\python.exe scripts/report_e001.py --run-dir results/{run.name}', '```', '',
        '## Sensitivity, verification and limitations', '',
        '| Omitted source | Oracle − random (pp) | Oracle − reselected best fixed (pp) |',
        '|---|---:|---:|']
    for row in summary['leave_one_video_out']:
        lines.append(f"| {row['omitted_video']} | {row['oracle_minus_random']*100:.2f} | {row['oracle_minus_best_fixed']*100:.2f} |")
    lines += ['', f'The hindsight best fixed quarter within each video matches its oracle in {local_ties}/8 sources.',
        'These label-selected local comparators are not validated policies. The positive',
        'full-cohort result does not demonstrate question-specific selection beyond',
        'source-specific preferred quarters. After leaving out one source and reselecting',
        f'best fixed, oracle-minus-best-fixed ranges from {min(loo):.2f} to {max(loo):.2f} pp;',
        f'it falls below 10 pp in {sum(v < 10 for v in loo)}/8 omissions. This sensitivity does',
        'not change the frozen full-cohort advancement rule, but limits the finding.']
    lines += ['', f"E1 base predictions disagree with corrected E0 in {checked['e0_base_prediction_discrepancies']}/24 cells.",
        f"All {checked['e0_preserved_files']} historical E0 evidence files match their captured SHA-256 hashes.",
        'Independent verification redecodes saved generated tokens with the frozen tokenizer,',
        'reparses and scores all 288 records, checks real worker generation receipts, regenerates',
        'all statistics/bootstrap intervals, and independently counts each primary endpoint.',
        'See [verification.json](verification.json), [forecast.json](forecast.json),',
        '[resource accounting](resource_accounting.json) and [source freeze](freeze.json).', '',
        'Timings and absolute allocated/reserved peaks are saved in summary.json as descriptive',
        'observations. No validated latency, FLOP saving, trained controller or practical ability',
        'to select the right clip is established. The oracle is limited to this four-action menu.',
        'No prompt/model/cohort search or training was performed. A negative result applies to',
        'this configuration and exposed development menu; it does not disprove adaptive video',
        'understanding generally. Do not start Experiment 2 automatically.', '',
        'The project-wide cap is 27,600 seconds after the explicit +7,200-second resource-only',
        'amendment. The ledger inherits every prior charge; storage remains 20 GiB including',
        'environment/caches/Git and video acquisition remains capped at 1 GiB. No new downloads',
        'or paid services. Restricted questions/options/labels/outputs/manifests remain local.', '',
        '## Reproduce or resume', '', '```powershell',
        f'$run = "results/{run.name}"',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 resume --config configs/e001.yaml --run-dir $run',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 analyze --config configs/e001.yaml --run-dir $run',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 verify --config configs/e001.yaml --run-dir $run',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 close --config configs/e001.yaml --run-dir $run',
        '```', '', 'Resume skips valid completed identities and never repeats completed answers.',
        'Historical reproduction requires the original executed sources and matching scientific',
        'fingerprints. A new collection requires a new run directory, inherited latest ledger,',
        'the existing local restricted data/model cache, and sufficient explicitly authorized time.', '']
    atomic_text(run/'report.md', '\n'.join(lines))


def research_readme(run, summary, e0):
    relative = run.relative_to(ROOT).as_posix()
    status = read_json(run/'status.json')
    verdict = status.get('verdict', status['status'])
    e1_result = (f"H_choice = {summary['metrics']['oracle_minus_random']*100:.2f} pp; "
                 f"oracle − best fixed = {summary['metrics']['oracle_minus_best_fixed']*100:.2f} pp"
                 if summary else 'No complete E1 result; see the preserved blocker')
    lines = ['# video-compute-value', '',
        'Processing more video can improve an answer, but it also adds computational cost.',
        'We study whether inexpensive observations can help decide when additional visual',
        'evidence is worth acquiring. Before building a decision policy, we first check',
        'whether our measurements are reliable and whether choosing a different clip can',
        'meaningfully change an answer.', '',
        'We use a frozen video-language model and multiple-choice video questions. Our',
        'current evidence comes from a small, exposed development cohort. We have not',
        'trained a controller or demonstrated an adaptive policy or validated speedup.',
        'This repository is separate from `video-refresh-benchmark`.', '',
        '| Experiment | Question and endpoint | Data and controls | Current result |',
        '|---|---|---|---|',
        '| E0: feasibility and measurement | Can we fit, reproduce, score and time requests? Operational gates, descriptive accuracy and memory. | 8 short videos / 24 questions; uniform 8/16/32/48/64 frames; separate text-only diagnostic; 3 video repetitions. | Complete and independently verified; **MEASUREMENT_UNSTABLE**. |',
        f'| E1: additional-clip headroom | Can hindsight choice among four added clips improve accuracy? Primary: oracle − exact expected random. | Same exposed 8 videos / 24 questions; base 8 frames, four quarter additions and uniform addition at 16 frames; 2 passes. | {e1_result}. |', '',
        'Both experiments use **Video-MME v1**, short category, the official annotation',
        '**test split**, revision `ead1408f75b618502df9a1d8e0950166bf0a2a0b`.',
        'We use all three questions per source, unchanged. Because these videos and',
        'questions have been exposed during development, neither experiment is an',
        'untouched confirmation study. The eight videos are the independent clusters;',
        'questions, candidate clips and repetitions are not independent additional videos.', '',
        'The shared model, processor and tokenizer are **Qwen/Qwen3-VL-4B-Instruct**,',
        'revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16 and SDPA, with frozen',
        'weights. Inputs are visual-only at the unchanged 192 × 320 canvas: no audio,',
        'external subtitles, transcripts or generated captions. We keep the original',
        'question and option order, prompt, strict answer parser and explicit greedy',
        'generation controls fixed. One CUDA worker runs at a time, with no quantization',
        'or offloading.', '',
        '## Experiment 0: establish the measurement foundation', '',
        'A frame-count comparison is useful only if it measures reproducible inputs and',
        'real model requests. We varied the uniformly sampled frame budget while keeping',
        'the cohort, model, spatial preprocessing and question fixed. Uniform-64 is a',
        'sampled reference, not every frame or an accuracy upper bound. Uniform sets are',
        'not necessarily nested.', '',
        'The corrected greedy collection completed **360 video requests and 24 separate',
        'text-only requests**, with 32 collection warmups and 25 preflight calls counted',
        'separately. The first repetition determines accuracy; three repetitions measure',
        'video-request repeatability. All inputs and generated tokens repeated exactly.',
        'Only the 48-frame setting met the proposed 20% median timing-range criterion.',
        'The operational verdict therefore remains **MEASUREMENT_UNSTABLE**.', '',
        '| Uniform frames | Correct / 24 | Accuracy | Descriptive median request (s) | Median repetition range |',
        '|---:|---:|---:|---:|---:|']
    for row in e0:
        lines.append(f"| {row['frames']} | {round(row['accuracy']*24)}/24 | {row['accuracy']*100:.2f}% | {row['median_request_seconds']:.3f} | {row['median_relative_timing_range']*100:.2f}% |")
    lines += ['', 'The text-only diagnostic answered 14/24 correctly (58.33%). It is a separate',
        'language-prior diagnostic, not a zero-cost adaptive choice. The historical',
        'sampled collection is preserved but excluded from these corrected-greedy results:',
        'checkpoint defaults had backfilled sampling despite a requested `do_sample=False`.',
        'The corrected adapter records and verifies the effective greedy policy.', '',
        'Regenerate all six research figures and the E1 report from saved data with:', '',
        '```powershell', f'.venv\\Scripts\\python.exe scripts/report_e001.py --run-dir {relative}', '```', '',
        f'![E0 accuracy by uniform frame budget]({relative}/figures/e0_accuracy.png)', '',
        'E0 accuracy uses the first pass over 24 questions from eight videos. More frames',
        'did not improve every answer monotonically. Source:',
        f'[{relative}/e0_plot_data.csv]({relative}/e0_plot_data.csv), exported from the',
        '[corrected E0 aggregates](results/e000_20261028_greedy_collection_v3/public_summary.json).', '',
        f'![E0 descriptive request latency with instability warning]({relative}/figures/e0_latency.png)', '',
        'E0 latency first takes the median of three repetitions per question, then the',
        'median over 24 questions. The timer includes complete video decoding and full',
        'inference. **Timing stability failed; these values do not validate efficiency.**',
        f'Source: [e0_plot_data.csv]({relative}/e0_plot_data.csv).', '',
        f'![E0 allocated and reserved GPU memory]({relative}/figures/e0_memory.png)', '',
        'Absolute PyTorch peaks over 360 video requests include resident weights.',
        'Reserved allocator memory differs from live allocated memory and total device',
        'usage. Uniform-64 alone uses the frozen in-timer cache-trim policy.',
        f'Source: [e0_plot_data.csv]({relative}/e0_plot_data.csv).', '',
        'A separate CPU stage diagnostic completed 48 preparations and reproduced the',
        'saved input hashes. It attributed about 98–99% of preparation time to the combined',
        'decode/selection/RGB/resize/stacking span. It did not identify one unique OS or',
        'driver cause, and does not repair the failed full-request timing gate. This is an',
        '[E0 diagnostic](results/e000_20261028_cpu_stage_v4/report.md), not another scientific experiment.', '',
        'The [corrected collection report](results/e000_20261028_greedy_collection_v3/completion_report.md)',
        'and [independent verification](results/e000_20261028_greedy_collection_v3/independent_final_verification.json)',
        'preserve the detailed gates, failures and resource history. E0 is limited by eight',
        'development videos and unstable timing; its frame ratios do not demonstrate',
        'proportional latency savings or an adaptive method.', '',
        '## Experiment 1: does choosing an additional clip matter?', '',
        'E0 changes a uniform frame budget, which does not isolate the choice of added',
        'evidence. E1 starts from the **same exact Uniform-8 frames** and adds eight',
        'distinct frames from one of four equal-duration quarters. A uniform-addition',
        'control retains the same starting frames and distributes eight new frames across',
        'the original timeline. Historical Uniform-16 is not this nested control.', '',
        'Uniform addition shares the base target times, so its new frames are the nearest',
        'unused neighbors of base frames. Comparisons apply to this specific nested control.',
        'All sources meet the inherited CFR-only eligibility rules; VFR handling is untested.', '',
        'The independent variable is the acquisition action: base only, quarter 1,',
        'quarter 2, quarter 3, quarter 4, or uniform addition. All five augmented conditions',
        'use 16 distinct frames, equal visual-token budgets and unchanged spatial dimensions.',
        f'Timestamp formatting can change text-token counts: [per-video differences]({relative}/augmented_text_token_differences.csv).',
        'Matched visual budgets do not guarantee identical FLOPs or latency.',
        'Original timestamps are retained. Adding frames can change native temporal',
        'pairing, timestamp tags, token positions and contextual representations. This is',
        'retained visual evidence followed by full re-encoding, not cached incremental inference.', '',
        'The primary endpoint is **H_choice = A_oracle − A_random**. The forced-acquisition',
        'oracle chooses the best of the four quarter actions afterward using labels.',
        'A_random is the exact average over every question/action pair. Best fixed is the',
        'strongest single quarter for the entire development cohort, selected afterward.',
        'The primary oracle excludes stopping and uniform addition. We separately report',
        'oracle plus stop, whose gain over base is nonnegative by construction.', '',
        f'**E1 verdict: {verdict}**', '']
    if summary:
        m = summary['metrics']
        loo = [row['oracle_minus_best_fixed']*100 for row in summary['leave_one_video_out']]
        local_ties = sum(abs(row['oracle_minus_best_fixed']) < 1e-12 for row in summary['per_video'])
        lines += ['Collection completed **288/288 requests**: 144 primary requests and all 144',
            'repeated in a fresh process. The first pass supplies accuracy; repetition is',
            'only a strict input/output repeatability check. No favorable pass was selected.', '',
            '| Quantity | First-pass accuracy |', '|---|---:|']
        for key, label in (('base', 'Base'), ('random', 'Exact expected random'), ('best_fixed', 'Best fixed quarter'),
                           ('uniform_addition', 'Uniform addition'), ('oracle', 'Forced-acquisition oracle'),
                           ('oracle_stop', 'Oracle plus stop')):
            lines.append(f'| {label} | {m[key]*100:.2f}% |')
        interval = summary['descriptive_95pct_cluster_bootstrap']['oracle_minus_random']
        lines += ['', f"**H_choice = {m['oracle_minus_random']*100:.2f} pp** (descriptive 95% cluster-bootstrap",
            f'interval [{interval[0]*100:.2f}, {interval[1]*100:.2f}] pp). Oracle minus best fixed is',
            f"{m['oracle_minus_best_fixed']*100:.2f} pp; oracle minus uniform addition is",
            f"{m['oracle_minus_uniform']*100:.2f} pp; oracle minus base is {m['oracle_minus_base']*100:.2f} pp.", '',
            f'![E1 accuracy by condition and retrospective comparator]({relative}/figures/e1_accuracy.png)', '',
            'E1 first-pass accuracy on all 24 questions from eight exposed videos. Orange',
            'bars are retrospective quantities. The expected random result enumerates all',
            'four quarters; oracle enumeration requires the additional real model calls.',
            f'Sources: [metrics.csv]({relative}/metrics.csv) and [summary.json]({relative}/summary.json).', '',
            f'![E1 anonymous question correctness matrix]({relative}/figures/e1_correctness.png)', '',
            'Every question remains in the denominator, including always-correct, always-wrong',
            'and invalid answers. Rows are anonymous and grouped into eight sources; no',
            'question text, ground-truth labels or generated answers are published.',
            f'Source: [correctness.csv]({relative}/correctness.csv).', '',
            f'![E1 paired effects for every source video]({relative}/figures/e1_per_video.png)', '',
            'Paired effects for all eight videos (three questions each). The best-fixed',
            'comparator uses the same globally selected quarter for every video. The report',
            'also supplies leave-one-video-out sensitivity and explicitly labeled local maxima.',
            f'Source: [per_video_effects.csv]({relative}/per_video_effects.csv).', '',
            f"Mixed successful/unsuccessful choices occurred in {summary['mixed_success_questions']} questions",
            f"across {summary['mixed_success_videos']} videos. The frozen exploratory advancement rule",
            f"**{'passed' if status['advancement_rule_passed'] else 'did not pass'}**. It required both oracle-minus-random",
            'and oracle-minus-best-fixed ≥10 pp, oracle above base and uniform addition,',
            'mixed choices in ≥3 videos, and every integrity/collection/greedy/repeatability/',
            'independent-verification check. These thresholds are development judgments.', '',
            'Intervals use 10,000 resamples of complete video clusters, seed 20261029,',
            'retaining all questions/actions and recomputing best fixed in every resample.',
            'Eight clusters give weak uncertainty estimates; we make no significance claim.', '']
        lines += ['The hindsight best fixed quarter selected separately for each video matches',
            f'the oracle in {local_ties}/8 sources. We therefore have not shown question-specific',
            'selection beyond source-specific quarter preferences. Leave-one-video-out',
            f'oracle-minus-best-fixed ranges from {min(loo):.2f} to {max(loo):.2f} pp and falls below',
            f'the 10 pp threshold in {sum(v < 10 for v in loo)}/8 omissions. The frozen rule passes on',
            'the full cohort; this sensitivity limits how strongly we can generalize it.', '']
    else:
        lines += ['No completed accuracy or illustrative E1 result graph is reported. The',
                  'preserved status explains the blocker; unexercised real paths remain unverified.', '']
    lines += [f'See the [E1 protocol](protocols/e001_additional_clip_headroom_v1.md), [report]({relative}/report.md),',
        f'[verification]({relative}/verification.json) and [timestamped freeze]({relative}/freeze.json).',
        'A hindsight oracle only shows what was possible inside this four-action menu.',
        'It cannot show that a practical policy can select the right clip before seeing',
        'its answer. E1 is accuracy-only: its descriptive times do not validate latency',
        'or computational efficiency. No controller training or Experiment 2 was started.', '',
        '## What the evidence supports', '',
        'We can preserve exact visual inputs, execute the frozen model and independently',
        'score saved outputs. E0 also shows that full video preparation can dominate a',
        'request, so fewer selected frames need not yield proportional savings. Its timing',
        'instability remains unresolved. E1 tests only retrospective development headroom',
        'within a fixed menu. Learning a useful acquisition policy, validating its costs,',
        'and confirming its behavior on independent data remain future hypotheses.', '',
        '## Reproduction and current resources', '',
        'Use the existing pinned environment. The [setup and E0 reproduction guide](docs/e0_reproduction.md)',
        'preserves installation, acquisition and measurement details. Python 3.12.14,',
        'PyTorch 2.8.0/CUDA 12.8, Transformers 4.57.1, PyAV 16.0.1 and the other exact',
        'software/source fingerprints are saved with the runs. Original restricted',
        'videos, annotations and immutable weights are required for local verification.', '',
        '```powershell', f'$run = "{relative}"',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 account --config configs/e001.yaml --run-dir $run',
        '.venv\\Scripts\\python.exe -m pytest -q --basetemp .cache/e001-regression',
        '.venv\\Scripts\\ruff.exe check src tests scripts',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 resume --config configs/e001.yaml --run-dir $run',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 analyze --config configs/e001.yaml --run-dir $run',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 verify --config configs/e001.yaml --run-dir $run',
        '.venv\\Scripts\\python.exe scripts/report_e001.py --run-dir $run',
        '.venv\\Scripts\\python.exe -m video_compute_value.e001 close --config configs/e001.yaml --run-dir $run', '```', '',
        'The CLI also implements `preflight` and `run` for a fresh named run whose resource',
        'grant and ledger have first been inherited from the latest closed project ledger.',
        'It freezes before outcomes, checks actual CPU/native/GPU paths, and admits full',
        'collection only after a real-probe forecast with a 1.5 factor and explicit reserves.',
        'Resume checks fingerprints and skips only valid completed identities; it never',
        'retries a completed wrong or invalid answer. Source/configuration changes require',
        'a new named version. Reports and resource amendments do not change scientific fingerprints.', '',
        'The user explicitly increased cumulative runtime by 7,200 seconds on 2026-10-29,',
        'from 20,400 to **27,600 seconds (7h40m)**, including all prior experiments, failures,',
        'setup, tests, analysis and publication. A new directory does not renew the budget.',
        f'The [current cumulative accounting]({relative}/resource_accounting.json) points to the',
        'latest local ledger. Limits remain **20 GiB** additional project storage, including',
        'environment/caches/Git, and **1 GiB** video acquisition. E1 reuses existing assets;',
        'no paid service, new acquisition, model substitution or background-app termination.', '',
        '## Data access and references', '',
        'Video-MME is restricted to academic research, and its source videos retain their',
        'owners’ copyrights. Obtain the data through the [official project and access terms](https://github.com/MME-Benchmarks/Video-MME#-dataset).',
        'We publish identifiers/hashes and derived sanitized statistics, not benchmark',
        'questions/options/labels, raw generated answers, videos, weights or private manifests.',
        'The official annotation file is `videomme/test-00000-of-00001.parquet` in the',
        '`lmms-eval/Video-MME` dataset repository at the frozen revision above.', '',
        '- Chaoyou Fu et al. (2025). *Video-MME: The First-Ever Comprehensive Evaluation',
        '  Benchmark of Multi-modal LLMs in Video Analysis*. CVPR.',
        '  [Official paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Video-MME_The_First-Ever_Comprehensive_Evaluation_Benchmark_of_Multi-modal_LLMs_in_CVPR_2025_paper.pdf).',
        '- Shuai Bai et al. (2025). *Qwen3-VL Technical Report*. arXiv:2511.21631.',
        '  [Paper](https://arxiv.org/abs/2511.21631), [official model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),',
        '  [official code and citation metadata](https://github.com/QwenLM/Qwen3-VL).', '']
    atomic_text(ROOT/'README.md', '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary, verification = read_json(run/'summary.json'), read_json(run/'verification.json')
    if not verification['statistics_independently_verified'] or verification['summary_sha256'] != file_hash(run/'summary.json'):
        raise RuntimeError('Require a saved, independently verified real E1 summary before plotting')
    (run/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    e0 = e0_figures(run)
    e1_figures(run, summary)
    report(run, summary, e0)
    research_readme(run, summary, e0)
    write_json(run/'figure_manifest.json', {p.name: file_hash(p) for p in sorted((run/'figures').glob('*.png'))})
    print(f'Regenerated six figures and report from saved real data in {run.name}')


if __name__ == '__main__':
    main()

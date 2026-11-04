"""Public-safe historical E0/E1/E2 presentation, with optional VidFit CPU inspection.

This is a presentation bridge, not an alternate collection runner. Only explicitly
listed public summaries and anonymous correctness tables are read. Frozen runs are
never written, and model inference is not exposed by the application.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import sys
from pathlib import Path

E0 = "results/e000_20261028_greedy_collection_v3"
E1 = "results/e001_20261029_132707_v1"
E2 = "results/e002_20261029_080758_continuation_v2"
TIMING = "results/e000_20261029_150300_timing_stabilization_v5"
RECOVERY = "results/e000_20261104_045654_runtime_recovery_v7"
ACCEPTANCE = "results/e000_20261104_acceptance25_v1"
EVIDENCE_COMMIT = "791f8120b3fddf34c6d987959b2d0241ee16a493"
REPOSITORY = "https://github.com/xyhsy11/video-compute-value"
PUBLIC_FILES = (
    f"{E0}/public_summary.json",
    f"{E0}/independent_final_verification.json",
    f"{E1}/summary.json",
    f"{E1}/verification.json",
    f"{E1}/correctness.csv",
    f"{E2}/summary.json",
    f"{E2}/verification.json",
    f"{E2}/statistics_verification.json",
    f"{E2}/correctness.csv",
    f"{E2}/practical_status.json",
    f"{E2}/resource_accounting.json",
    f"{TIMING}/status.json",
)
RECOVERY_FILES = tuple(f"{RECOVERY}/{name}" for name in (
    "runtime_resolution.json", "summary.json", "verification.json", "freeze.json", "resource_accounting.json"))
ACCEPTANCE_FILES = tuple(f"{ACCEPTANCE}/{name}" for name in (
    "authorization.json", "freeze.json", "summary.json", "verification.json", "resource_accounting.json"))
LABELS = {
    "base": "Base · 8 frames",
    "quarter1": "Add quarter 1 · 16 frames",
    "quarter2": "Add quarter 2 · 16 frames",
    "quarter3": "Add quarter 3 · 16 frames",
    "quarter4": "Add quarter 4 · 16 frames",
    "uniform_addition": "Nested uniform · 16 frames",
    "direct16": "Direct uniform · 16 frames",
    "random": "Expected random quarter",
    "best_fixed": "Best fixed · hindsight",
    "fixed": "Train-selected fixed quarter",
    "V": "V · base visual observations",
    "Q": "Q · question observations",
    "VQ": "VQ · visual + question",
    "oracle": "Oracle · hindsight",
    "oracle_stop": "Oracle + stop · hindsight",
}


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric observation")
    return value


def _same(actual, expected, name):
    if not math.isclose(_finite(actual), _finite(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Saved evidence disagrees: {name}")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _correctness(rows, groups):
    expected = [f"{groups}{i:02}Q{q}" for i in range(1, len(rows) // 3 + 1) for q in (1, 2, 3)]
    _require([r["anonymous_question"] for r in rows] == expected, "Anonymous question identity mismatch")
    for row in rows:
        for key, value in row.items():
            if key == "anonymous_question":
                continue
            allowed = {"0", "1", "2", "3"} if key.endswith("_action") else {"0", "1"}
            _require(value in allowed, "Invalid saved correctness/action value")


def _recompute(rows, policies=()):
    count = len(rows)
    names = ["base", "uniform_addition"] + (["direct16"] if policies else [])
    values = {name: sum(int(r[name]) for r in rows) / count for name in names}
    for q in range(1, 5):
        values[f"quarter{q}"] = sum(int(r[f"quarter{q}"]) for r in rows) / count
    values["random"] = sum(values[f"quarter{q}"] for q in range(1, 5)) / 4
    values["best_fixed"] = max(values[f"quarter{q}"] for q in range(1, 5))
    values["oracle"] = sum(max(int(r[f"quarter{q}"]) for q in range(1, 5)) for r in rows) / count
    values["oracle_stop"] = sum(
        max(int(r[k]) for k in ("base", "quarter1", "quarter2", "quarter3", "quarter4")) for r in rows
    ) / count
    for name in policies:
        values[name] = sum(int(r[f"quarter{int(r[name + '_action']) + 1}"]) for r in rows) / count
    return values


def load_evidence(root):
    """Fail closed on missing, changed, incomplete, or inconsistent saved evidence."""
    root = Path(root).resolve()
    data, sources = {}, []
    selected_files = PUBLIC_FILES + (RECOVERY_FILES if (root / RECOVERY).exists() else ())
    selected_files += ACCEPTANCE_FILES if (root / ACCEPTANCE).exists() else ()
    for relative in selected_files:
        path = (root / relative).resolve()
        _require(path.is_relative_to(root), "Evidence path escapes the project")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        sources.append({"path": relative, "sha256": digest})
        text = raw.decode("utf-8")
        if relative.endswith(".csv"):
            data[relative] = list(csv.DictReader(io.StringIO(text)))
        else:
            data[relative] = json.loads(text, parse_constant=_reject_constant)
    hashes = {s["path"]: s["sha256"] for s in sources}
    for run in (E1, E2):
        verification = data[f"{run}/verification.json"]
        _require(verification["summary_sha256"] == hashes[f"{run}/summary.json"], "Summary hash mismatch")
    _require(data[f"{E1}/verification.json"]["matrix_sha256"] == hashes[f"{E1}/correctness.csv"],
             "E1 matrix hash mismatch")
    _require(data[f"{E0}/independent_final_verification.json"]["passed"] is True, "E0 saved-record audit failed")
    _require(data[f"{E1}/verification.json"]["all_checks_passed"] is True, "E1 integrity audit failed")
    _require(data[f"{E2}/verification.json"]["all_E2A_checks_passed"] is True, "E2A integrity audit failed")
    _require(data[f"{E2}/statistics_verification.json"]["all_statistical_checks_passed"] is True,
             "E2A statistical audit failed")
    e1, e2 = data[f"{E1}/summary.json"], data[f"{E2}/summary.json"]
    matrix1, matrix2 = data[f"{E1}/correctness.csv"], data[f"{E2}/correctness.csv"]
    _require(len(matrix1) == e1["questions"] == 24 and e1["videos"] == 8, "E1 cohort mismatch")
    _require(len(matrix2) == e2["questions"] == 120 and e2["source_groups"] == 40, "E2A cohort mismatch")
    _correctness(matrix1, "V")
    _correctness(matrix2, "S")
    for run, summary, matrix, policies in ((E1, e1, matrix1, ()), (E2, e2, matrix2, ("fixed", "V", "Q", "VQ"))):
        computed = _recompute(matrix, policies)
        for name, value in summary["metrics"].items():
            if name in computed:
                _same(value, computed[name], f"{run}/{name}")
    _same(e1["metrics"]["oracle_minus_random"], e1["metrics"]["oracle"] - e1["metrics"]["random"], "H_choice")
    _same(e2["delta_fixed"], e2["metrics"]["VQ"] - e2["metrics"]["fixed"], "Delta_fixed")
    _same(e2["delta_question"], e2["metrics"]["VQ"] - e2["metrics"]["V"], "Delta_question")
    positive = sum(f["delta_fixed"] > 0 for f in e2["fold_effects"])
    _require(len(e2["fold_effects"]) == 5, "E2A fold count mismatch")
    advancement = (e2["delta_fixed"] >= .05 and positive >= 4
                   and e2["descriptive_conditional_95pct_intervals"]["fixed"][0] > 0)
    _require(not advancement and e2["E2B_status"] == "NOT_ADMITTED_E2A_ADVANCEMENT", "Frozen E2A gate changed")
    _require(data[f"{E2}/practical_status.json"]["status"] == e2["E2B_status"], "E2B status mismatch")
    _require(data[f"{TIMING}/status.json"]["historical_E0"] == "MEASUREMENT_UNSTABLE", "E0 verdict changed")
    e0 = []
    for condition in data[f"{E0}/public_summary.json"]["conditions"]:
        if condition["budget"] == 0:
            continue
        _require(condition["complete"] and condition["successful"] == 72, "Incomplete E0 condition")
        _same(condition["accuracy"], condition["rep1_correct_out_of_24"] / 24, "E0 accuracy")
        e0.append({
            "frames": condition["budget"], "accuracy": condition["accuracy"],
            "correct": condition["rep1_correct_out_of_24"],
            "seconds": condition["latency_seconds"]["median"],
            "range": condition["median_relative_timing_range"],
            "allocated_gib": condition["peak_allocated_bytes"] / 1024**3,
            "reserved_gib": condition["peak_reserved_bytes"] / 1024**3,
        })
    _require([r["frames"] for r in e0] == [8, 16, 32, 48, 64], "E0 budget mismatch")
    resource_keys = ("at", "cumulative_seconds", "authorized_limit_seconds", "storage_current_bytes",
                     "storage_limit_bytes", "video_transfer_and_interruption_upper_bound_bytes", "video_limit_bytes")
    resource = {k: data[f"{E2}/resource_accounting.json"][k] for k in resource_keys}
    # Projection is intentional: arbitrary fields, private manifests, answers and labels never enter the export.
    projected = {
        "schema": "video-compute-value.vidfit-view.v1", "evidence_commit": EVIDENCE_COMMIT,
        "e0": e0,
        "e1": {"metrics": {k: e1["metrics"][k] for k in LABELS if k in e1["metrics"]},
               "h_choice": e1["metrics"]["oracle_minus_random"],
               "h_interval": e1["descriptive_95pct_cluster_bootstrap"]["oracle_minus_random"],
               "matrix": [{k: r[k] for k in ("anonymous_question", "base", "quarter1", "quarter2", "quarter3", "quarter4", "uniform_addition")} for r in matrix1]},
        "e2": {"metrics": {k: e2["metrics"][k] for k in LABELS if k in e2["metrics"]},
               "delta_fixed": e2["delta_fixed"], "delta_question": e2["delta_question"],
               "interval": e2["descriptive_conditional_95pct_intervals"]["fixed"],
               "folds": [{k: f[k] for k in ("fold", "delta_fixed", "delta_question")} for f in e2["fold_effects"]],
               "positive_folds": positive, "advancement": advancement},
        "resources": resource, "sources": sources,
        "source_fingerprint": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "verification": "Saved summary hashes, anonymous correctness and frozen E2A gate checked",
    }
    if f"{RECOVERY}/runtime_resolution.json" in data:
        resolution = data[f"{RECOVERY}/runtime_resolution.json"]
        verification = data[f"{RECOVERY}/verification.json"]
        summary = data[f"{RECOVERY}/summary.json"]
        _require(resolution['verified'] == verification['passed'] == summary['integrity_passed'],
                 'Recovery verification status mismatch')
        _require(resolution['preflight_completed'] + resolution['diagnostic_completed']
                 == verification['completed_model_calls'], 'Recovery call accounting mismatch')
        _require(resolution['selected_policy'] == summary['selected_policy'], 'Recovery policy mismatch')
        metrics = [{k: row[k] for k in ('arm', 'budget', 'S')} for row in summary['conditions'] if row['S'] is not None]
        _require(metrics == verification['independent_metrics'], 'Recovery metrics disagree with verification')
        projected['runtime_recovery'] = {k: resolution[k] for k in (
            'status', 'successful_model_loads', 'preflight_completed', 'diagnostic_completed',
            'failed_model_initializations', 'timing_status', 'selected_policy', 'verified',
            'root_cause', 'confirmation_status')}
        projected['runtime_recovery']['conditions'] = [
            {k: row[k] for k in ('arm', 'budget', 'S', 'primary_requests')} for row in summary['conditions']]
        projected['runtime_recovery']['fingerprint'] = data[f"{RECOVERY}/freeze.json"]['fingerprint']
    if f'{ACCEPTANCE}/summary.json' in data:
        from .common import digest
        from .e000_acceptance import select_policy
        amended = data[f'{ACCEPTANCE}/summary.json']
        proof = data[f'{ACCEPTANCE}/verification.json']
        frozen = data[f'{ACCEPTANCE}/freeze.json']
        _require(proof['passed'] and proof['summary_sha256'] == digest(amended), 'Amended summary seal mismatch')
        _require(digest({k: v for k, v in frozen.items() if k != 'fingerprint'}) == frozen['fingerprint']
                 == amended['fingerprint'], 'Amendment fingerprint mismatch')
        for path, expected in frozen['source_files'].items():
            _require(hashes[path] == expected, 'Amendment source evidence changed')
        _require(amended['threshold'] == .25 and amended['new_model_calls'] == 0
                 and amended['decision_timing'] == 'USER_AMENDED_AFTER_OBSERVATION', 'Amendment scope changed')
        selected = select_policy(amended['conditions'], .25, integrity=amended['integrity_passed'])
        _require(selected == amended['selected_policy'] == 'B' and amended['good'] is True,
                 'Amended 25% decision disagrees')
        for row in amended['conditions']:
            original = next(c for c in projected['runtime_recovery']['conditions']
                            if (c['arm'], c['budget']) == (row['arm'], row['budget']))
            _same(row['S'], original['S'], 'Amended timing statistic')
        projected['e0_acceptance'] = {key: amended[key] for key in (
            'status', 'good', 'threshold', 'selected_policy', 'conditions', 'decision_timing',
            'source_model_calls_reused', 'new_model_calls', 'original_20pct_status', 'confirmation_status')}
    return projected


CSS = """
.vcv {--ink:#172e3b;--muted:#526875;--line:#d8e3e7;--teal:#087f80;--orange:#ab5e10;
 color:var(--ink);font:15px/1.6 'Segoe UI',Arial,sans-serif;max-width:1260px;margin:auto;}
.vcv * {box-sizing:border-box}.vcv h1{font-size:38px;line-height:1.15;letter-spacing:-1.1px;margin:12px 0}
.vcv h2{font-size:25px;line-height:1.3;margin:26px 0 12px}.vcv h3{font-size:18px;margin:0 0 8px}
.vcv p{margin:8px 0 15px}.vcv a{color:#006b7e;text-decoration:underline}.vcv small{font-size:12px}
.vcv .eyebrow{letter-spacing:2px;text-transform:uppercase;font-size:11px;font-weight:700;color:var(--teal)}
.vcv .muted{color:var(--muted)}.vcv .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.vcv .card,.vcv .chart{background:white;border:1px solid var(--line);border-radius:16px;padding:22px;margin:12px 0}
.vcv .card{margin:0}.vcv .num{font-size:32px;font-weight:700;line-height:1.3;color:var(--teal)}
.vcv .badge{display:inline-block;border-radius:6px;padding:3px 8px;font-size:11px;font-weight:700;
 background:#e2f3ee;color:#155a51;margin:8px 0;max-width:100%;overflow-wrap:anywhere}
.vcv .amber{background:#fff0d5;color:#845009}
.vcv .rose{background:#fbe9e9;color:#8e303c}.vcv .notice{border-left:4px solid #c27823;background:#fff6e7;
 padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0}.vcv .flow{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap}
.vcv .step{flex:1;min-width:135px;border:1px solid var(--line);padding:12px;border-radius:10px;background:#f5f9fa}
.vcv .arrow{align-self:center;color:var(--teal);font-size:22px}.vcv .barrow{display:grid;
 grid-template-columns:minmax(160px,270px) 1fr 88px;align-items:center;gap:12px;margin:12px 0}
.vcv .track{background:#edf3f5;height:22px;border-radius:5px;overflow:hidden}
.vcv .bar{background:var(--teal);height:100%;border-radius:5px}.vcv .hindsight{background:var(--orange)}
.vcv .value{text-align:right;font-variant-numeric:tabular-nums;font-weight:650}
.vcv .tablewrap{overflow-x:auto}.vcv table{border-collapse:collapse;width:100%;font-size:13px}
.vcv th,.vcv td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}
.vcv th{background:#edf4f5}.vcv .yes{color:#126459;font-weight:700}.vcv .no{color:#9c3741;font-weight:700}
.vcv .cell1{background:#d7efe7;text-align:center}.vcv .cell0{background:#f6e5e7;text-align:center}
.vcv code{font-size:11px;overflow-wrap:anywhere}.vcv .source{font-size:12px;color:var(--muted);margin-top:14px}
.vcv .nav{position:sticky;top:0;background:#f5f9faf5;padding:14px;display:flex;gap:18px;flex-wrap:wrap;
 border-bottom:1px solid var(--line);z-index:5}.vcv section{scroll-margin-top:90px}
.vcv details{border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.vcv summary{cursor:pointer;font-weight:650}.vcv svg{width:100%;height:auto;max-height:300px}
@media(max-width:720px){.vcv .grid{grid-template-columns:1fr}.vcv h1{font-size:30px}
 .vcv .barrow{grid-template-columns:minmax(100px,160px) 1fr 65px;font-size:12px}.vcv .card,.vcv .chart{padding:16px}}
"""


def esc(value):
    return html.escape(str(value), quote=True)


def link(path, label):
    ref = 'main' if path.startswith((RECOVERY + '/', ACCEPTANCE + '/')) else EVIDENCE_COMMIT
    return f'<a href="{REPOSITORY}/blob/{ref}/{esc(path)}" target="_blank" rel="noopener noreferrer">{esc(label)}</a>'


def source(path, label="Saved evidence"):
    provenance = ('published decision record; SHA-256 in Evidence' if path.startswith((RECOVERY + '/', ACCEPTANCE + '/'))
                  else f'frozen source commit {EVIDENCE_COMMIT[:7]}')
    return f'<p class="source">{link(path, label)} · {provenance}</p>'


def wrap(content):
    return f'<style>{CSS}</style><div class="vcv">{content}</div>'


def bars(rows, unit="%", maximum=100):
    parts = []
    for label, value, hindsight in rows:
        value = _finite(value)
        width = max(0, min(100, value / maximum * 100))
        parts.append(f'<div class="barrow"><span>{esc(label)}</span><div class="track">'
                     f'<div class="bar {"hindsight" if hindsight else ""}" style="width:{width:.5f}%"></div>'
                     f'</div><span class="value">{value:.2f}{esc(unit)}</span></div>')
    return "".join(parts)


def flow(steps):
    return '<div class="flow">' + '<span class="arrow" aria-hidden="true">→</span>'.join(
        f'<div class="step"><b>{esc(title)}</b><br><small>{esc(detail)}</small></div>' for title, detail in steps
    ) + '</div>'


def overview(evidence):
    e1, e2 = evidence["e1"], evidence["e2"]
    recovery = evidence.get('runtime_recovery')
    current_runtime = (f'<div class="notice"><b>Current E0 runtime recovery:</b> {esc(recovery["status"])}. '
                       f'{recovery["preflight_completed"] + recovery["diagnostic_completed"]}/132 new screen calls; '
                       f'timing: {esc(recovery["timing_status"])}. See E0 for the new evidence.</div>'
                       if recovery else '')
    acceptance = evidence.get('e0_acceptance')
    if acceptance:
        current_runtime = ('<div class="card"><span class="badge">E0 GOOD · AMENDED ≤25% RULE</span>'
                           '<p><b>Resident policy B accepted:</b> 18.48% at 8 frames and 22.53% at 64 frames. '
                           'Same 132 verified calls; user-directed criterion change after observation. '
                           'The original 20% decision is preserved.</p></div>')
    e0_card = (
        '<div class="card"><span class="eyebrow">01 · Measurement / E0</span><h2>Resident timing accepted</h2>'
        '<div class="num">GOOD · ≤25%</div><p>18.48% at 8 frames / 22.53% at 64 frames.</p>'
        '<span class="badge">RESIDENT POLICY B ACCEPTED</span>'
        '<p>8 sources · 8 questions · 3 sessions<br>User-amended after observation.</p></div>'
        if acceptance else
        '<div class="card"><span class="eyebrow">01 · Measurement / E0</span><h2>Can we trust the cost?</h2>'
        '<div class="num">62.50–87.50%</div><p>Accuracy across 8–64 uniform frames.</p>'
        '<span class="badge amber">MEASUREMENT_UNSTABLE</span><p>8 sources · 24 questions<br>Timing did not qualify.</p></div>'
    )
    return wrap(
        '<p class="eyebrow">VidFit × video-compute-value</p>'
        '<h1>When is more video worth it?</h1>'
        '<p class="muted">Three experiments, one evidence trail. Saved measurements from October–November 2026.</p>'
        + current_runtime +
        '<div class="grid">' + e0_card +
        '<div class="card"><span class="eyebrow">02 · Headroom / E1</span><h2>Does the clip choice matter?</h2>'
        f'<div class="num">+{100 * e1["h_choice"]:.2f} pp</div><p>Hindsight oracle minus expected random.</p>'
        '<span class="badge">RETROSPECTIVE HEADROOM</span><p>8 exposed sources · 24 questions<br>Accuracy-only diagnostic.</p></div>'
        '<div class="card"><span class="eyebrow">03 · Prediction / E2A</span><h2>Can we choose beforehand?</h2>'
        f'<div class="num">+{100 * e2["delta_fixed"]:.2f} pp</div><p>Held-out VQ minus train-selected fixed.</p>'
        '<span class="badge rose">PREDICTIVE VALUE NOT DEMONSTRATED</span>'
        '<p>40 new sources · 120 questions<br>E2B advancement failed.</p></div></div>'
        '<h2>The research path</h2>' + flow([
            ("E0 · reliable measurement?", "Resident B accepted at amended 25%; original 20% result retained."
             if acceptance else "Input/output repeatability passed; timing stability failed."),
            ("E1 · acquisition opportunity?", "Retain 8 base frames, add 8 from one of four quarters."),
            ("E2A · predictable gain?", "Train cheap selectors; evaluate on held-out source groups."),
            ("E2B · practical value", "Not admitted by the frozen E2A advancement rule."),
        ]) +
        '<div class="notice"><b>What the evidence supports.</b> E1 shows limited hindsight opportunity. '
        'E2A does not demonstrate predictive value for this observation family. '
        'Practical efficiency remains unevaluated; E2B was not admitted.</div>'
        '<h2>Shared frozen answerer</h2><div class="card"><b>Qwen3-VL-4B-Instruct · BF16 · SDPA</b>'
        '<p>Visual-only, 192 × 320 canvas, unchanged question/option order, strict parser and greedy generation. '
        'One CUDA worker at a time; no offloading or quantization.</p>'
        '<small>Model / processor / tokenizer revision</small><br>'
        '<code>ebb281ec70b05090aa6165b016eac8ec08e71b17</code></div>'
        '<p class="muted">Sources are the independent units. Questions, candidate actions and repeated calls do not '
        'create more independent videos. E0/E1 use exposed development data; E2A is a development pilot.</p>'
    )


def e0_panel(evidence, metric="Accuracy"):
    settings = {"Accuracy": ("accuracy", 100, "%", 100),
                "Request time (descriptive)": ("seconds", 1, " s", 7),
                "Timing repetition range": ("range", 100, "%", 100),
                "Peak allocated memory": ("allocated_gib", 1, " GiB", 12)}
    _require(metric in settings, "Unknown E0 display metric")
    key, scale, unit, maximum = settings[metric]
    chart = bars([(f'Uniform-{r["frames"]}', r[key] * scale, False) for r in evidence["e0"]], unit, maximum)
    rows = "".join(f'<tr><td>{r["frames"]}</td><td>{r["correct"]}/24</td><td>{100*r["accuracy"]:.2f}%</td>'
                   f'<td>{r["seconds"]:.3f}</td><td>{100*r["range"]:.2f}%</td>'
                   f'<td>{r["allocated_gib"]:.2f} / {r["reserved_gib"]:.2f}</td></tr>' for r in evidence["e0"])
    return wrap('<span class="eyebrow">E0 · frozen feasibility and measurement</span><h1>Measurement foundation and current recovery</h1>'
                '<span class="badge amber">MEASUREMENT_UNSTABLE</span>' + flow([
                    ("Input", "8 sources / 24 questions; visual-only."),
                    ("Vary", "Uniform 8, 16, 32, 48 or 64 frames; sets need not be nested."),
                    ("Measure", "First-pass accuracy; three repetitions for video timing."),
                ]) + f'<div class="chart"><h3>{esc(metric)}</h3>{chart}</div>'
                '<div class="notice">The fixed timing target was a median relative repetition range ≤20%. '
                'Only Uniform-48 met it. Descriptive request times cannot validate efficiency or speedup claims.</div>'
                '<div class="tablewrap"><table><thead><tr><th>Frames</th><th>Correct</th><th>Accuracy</th>'
                '<th>Median request (s)</th><th>Repetition range</th><th>Allocated / reserved (GiB)</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>'
                '<p>360 video requests + 24 separate text-only requests; warmups and preflight are separate. '
                'The text-only diagnostic scored 14/24 (58.33%). Memory is the in-process PyTorch peak including weights.</p>'
                + recovery_panel(evidence) +
                '<details><summary>Historical failed timing follow-up · 2026-10-29</summary><div class="card">'
                '<span class="badge rose">BLOCKED_MODEL_RUNTIME</span><p>0/132 planned model calls completed. '
                'All 16 CPU preparations matched historical inputs. Model initialization crashed before generation; '
                'fresh/resident worker stability and confirmation were unavailable in this historical attempt.</p></div></details>'
                + source(f"{E0}/public_summary.json") + source(f"{TIMING}/report.md", "Timing follow-up report"))


def recovery_panel(evidence):
    recovery = evidence.get('runtime_recovery')
    if recovery is None:
        return ''
    complete = recovery['preflight_completed'] + recovery['diagnostic_completed']
    rows = ''.join(f'<tr><td>{esc(c["arm"])}</td><td>{c["budget"]}</td>'
                   f'<td>{c["primary_requests"]}/24</td><td>'
                   + (f'{100*c["S"]:.2f}%' if c['S'] is not None else 'Unavailable')
                   + '</td></tr>' for c in recovery['conditions'])
    original = ('<h2>Runtime recovery · 2026-11-04</h2><div class="card">'
            f'<span class="badge">{esc(recovery["status"])}</span>'
            f'<h3>{complete}/132 new screen calls completed</h3>'
            f'<p>{recovery["successful_model_loads"]} successful fresh model loads; '
            f'{recovery["failed_model_initializations"]} failed initializations. '
            f'{recovery["preflight_completed"]}/12 preflight calls. The original model, packages and driver were retained.</p>'
            f'<p><b>Timing screen:</b> {esc(recovery["timing_status"])}<br>'
            f'<b>Selected worker policy:</b> {esc(recovery["selected_policy"] or "None")}<br>'
            f'<b>Confirmation:</b> {esc(recovery["confirmation_status"])}</p>'
            '<div class="tablewrap"><table><thead><tr><th>Arm</th><th>Frames</th><th>Primaries</th>'
            f'<th>Median relative range</th></tr></thead><tbody>{rows}</tbody></table></div>'
            '<p>The original frozen criterion was ≤20% at both frame budgets with all integrity checks passing. '
            'Current successful loading does not establish the cause of the old crash. '
            'Historical E0 remains MEASUREMENT_UNSTABLE.</p></div>'
            + source(f'{RECOVERY}/report.md', 'New runtime recovery report'))
    acceptance = evidence.get('e0_acceptance')
    if not acceptance:
        return original
    rows = ''.join(f'<tr><td>{esc(c["arm"])}</td><td>{c["budget"]}</td><td>{100*c["S"]:.2f}%</td>'
                   f'<td class="{"yes" if c["passes_amended_25pct"] else "no"}">'
                   f'{"PASS" if c["passes_amended_25pct"] else "FAIL"}</td></tr>' for c in acceptance['conditions'])
    return ('<h2>Current E0 · amended 25% acceptance</h2><div class="card">'
            '<span class="badge">GOOD · RESIDENT POLICY B ACCEPTED</span>'
            '<h3>≤25% timing variation at both tested budgets</h3>'
            '<p>Policy B: <b>18.48% at 8 frames</b> and <b>22.53% at 64 frames</b>. '
            '132 verified source calls; 11 successful model loads. This amendment adds zero model calls.</p>'
            '<div class="tablewrap"><table><thead><tr><th>Policy</th><th>Frames</th>'
            f'<th>Median relative range</th><th>Amended ≤25%</th></tr></thead><tbody>{rows}</tbody></table></div>'
            '<p><b>User-amended after observation · 2026-11-04.</b> The measurements are unchanged. '
            'The original ≤20% decision remains recorded. Full confirmation has not been run; '
            'this is operational acceptance for the tested 8/64-frame resident-worker screen.</p></div>'
            + source(f'{ACCEPTANCE}/report.md', 'New E0 acceptance report')
            + '<details><summary>Original recovery decision · frozen ≤20% rule</summary>' + original + '</details>')


def e1_matrix(evidence):
    names = ["base", "quarter1", "quarter2", "quarter3", "quarter4", "uniform_addition"]
    rows = "".join('<tr><td>' + esc(row["anonymous_question"]) + '</td>' + ''.join(
        f'<td class="cell{row[name]}">{"✓" if row[name] == "1" else "×"}</td>' for name in names
    ) + '</tr>' for row in evidence["e1"]["matrix"])
    return ('<details><summary>Inspect all 24 anonymous questions</summary><p>✓ correct · × incorrect. '
            'First pass only; every question remains in the denominator. Three rows per independent source.</p>'
            '<div class="tablewrap"><table><thead><tr><th>Question</th><th>Base</th><th>Qtr 1</th>'
            '<th>Qtr 2</th><th>Qtr 3</th><th>Qtr 4</th><th>Uniform add</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></details>')


def e1_panel(evidence):
    e1 = evidence["e1"]
    methods = ["base", "quarter1", "quarter2", "quarter3", "quarter4", "uniform_addition", "random", "best_fixed", "oracle"]
    chart = bars([(LABELS[k], e1["metrics"][k]*100, k in ("best_fixed", "oracle")) for k in methods])
    return wrap('<span class="eyebrow">E1 · retrospective four-clip headroom</span><h1>The choice can change the answer</h1>'
                + flow([("Retain", "The exact 8 base frames spanning the video."),
                        ("Choose", "Quarter 1, 2, 3 or 4; add 8 distinct frames."),
                        ("Re-encode", "16 total frames; same visual-token budget."),
                        ("Compare", "First-pass accuracy; repeat to verify inputs and outputs.")]) +
                '<div class="chart"><h3>First-pass accuracy · 8 sources / 24 questions</h3>' + chart +
                '<small>Teal: observed conditions / expected random. Orange: quantities chosen using hindsight.</small></div>'
                f'<div class="grid"><div class="card"><h3>Choice headroom</h3><div class="num">+{e1["h_choice"]*100:.2f} pp</div>'
                f'<p>Oracle − expected random<br>Descriptive 95% cluster interval: [{e1["h_interval"][0]*100:.2f}, '
                f'{e1["h_interval"][1]*100:.2f}] pp</p></div>'
                '<div class="card"><h3>Collection</h3><div class="num">288 / 288</div>'
                '<p>144 primary requests + 144 fresh-process repetitions. Accuracy uses the first pass.</p></div>'
                '<div class="card"><h3>Development rule</h3><span class="badge">PASSED</span>'
                '<p>Mixed action outcomes in 7 questions across 5 sources; integrity checks passed.</p></div></div>'
                '<div class="notice"><b>Hindsight is not an executable policy.</b> The oracle uses labels after '
                'collection. A separate hindsight fixed quarter per source matched the oracle in 8/8 sources. '
                'Question-specific selection and practical efficiency are not established.</div>' + e1_matrix(evidence)
                + source(f"{E1}/summary.json") + source("protocols/e001_additional_clip_headroom_v1.md", "Frozen E1 protocol"))


def fold_chart(folds):
    parts = ['<svg viewBox="0 0 760 235" role="img" aria-label="VQ minus fixed accuracy in five held-out folds; zero at center">',
             '<line x1="380" y1="12" x2="380" y2="198" stroke="#8aa1ad"/>']
    for i, fold in enumerate(folds):
        value = fold["delta_fixed"] * 100
        width = abs(value) * 25
        x = 380 if value >= 0 else 380 - width
        y = 18 + i * 35
        color = "#087f80" if value > 0 else "#a7444b"
        parts.append(f'<text x="12" y="{y+17}" font-size="14" fill="#172e3b">Fold {fold["fold"]+1}</text>'
                     f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="24" rx="3" fill="{color}"/>'
                     f'<text x="660" y="{y+17}" font-size="14" fill="#172e3b">{value:+.2f} pp</text>')
    parts.append('<text x="330" y="226" fill="#526875" font-size="13">0 pp · VQ − fixed</text></svg>')
    return ''.join(parts)


def e2_panel(evidence, display="Accuracy"):
    e2 = evidence["e2"]
    _require(display in ("Accuracy", "Outer-fold gains"), "Unknown E2 display")
    methods = ["base", "random", "fixed", "V", "Q", "VQ", "uniform_addition", "direct16", "oracle"]
    chart = (bars([(LABELS[k], e2["metrics"][k]*100, k == "oracle") for k in methods])
             if display == "Accuracy" else fold_chart(e2["folds"]))
    lower, upper = [x*100 for x in e2["interval"]]
    return wrap('<span class="eyebrow">E2A · pre-acquisition predictability</span><h1>The cheap predictor did not pass</h1>'
                '<span class="badge rose">PREDICTIVE_VALUE_NOT_DEMONSTRATED_FOR_THIS_OBSERVATION_FAMILY</span>'
                + flow([("Observe", "Only base frames / permitted metadata (V) and original question text (Q)."),
                        ("Train", "Small V / Q / VQ ridge selectors; training-only scaling and controls."),
                        ("Hold out", "5 outer source folds × 8 sources; seal decisions before scoring."),
                        ("Evaluate", "40 new sources, all 120 original questions.")]) +
                f'<div class="chart"><h3>{esc(display)} · held-out development pilot</h3>{chart}</div>'
                f'<div class="grid"><div class="card"><h3>VQ − train-fixed</h3><div class="num">{e2["delta_fixed"]*100:+.2f} pp</div>'
                f'<p>Conditional descriptive 95% source interval: [{lower:.2f}, {upper:.2f}] pp</p></div>'
                f'<div class="card"><h3>Positive held-out folds</h3><div class="num">{e2["positive_folds"]} / 5</div>'
                '<p>Frozen rule required at least 4.</p></div>'
                '<div class="card"><h3>Precision</h3><span class="badge amber">INCONCLUSIVE_LOW_PRECISION</span>'
                '<p>Explicit question contribution: NOT_DEMONSTRATED.</p></div></div>'
                '<h2>The frozen advancement gate</h2><div class="tablewrap"><table><thead><tr><th>Criterion</th>'
                '<th>Required</th><th>Observed</th><th>Result</th></tr></thead><tbody>'
                f'<tr><td>VQ − fixed</td><td>≥5.00 pp</td><td>{e2["delta_fixed"]*100:+.2f} pp</td><td class="no">FAIL</td></tr>'
                f'<tr><td>Positive outer folds</td><td>≥4 / 5</td><td>{e2["positive_folds"]} / 5</td><td class="no">FAIL</td></tr>'
                f'<tr><td>Interval lower bound</td><td>&gt;0.00 pp</td><td>{lower:.2f} pp</td><td class="no">FAIL</td></tr>'
                '</tbody></table></div><div class="notice"><b>E2B: NOT_ADMITTED_E2A_ADVANCEMENT.</b> '
                'Expensive practical-stage collection was not run. Timing remains UNQUALIFIED; practical value is '
                'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT. More remaining budget does not override the scientific gate.</div>'
                '<p>960 primary calls + 120 fresh-process repeatability calls; 4 warmups are separate. '
                'Accuracy is computed from saved first-pass correctness. Random is the exact four-action expectation; '
                'the oracle is nondeployable hindsight.</p>'
                + source(f"{E2}/summary.json") + source(f"{E2}/statistics_verification.json", "Independent statistical verification"))


def evidence_panel(evidence):
    r = evidence["resources"]
    rows = ''.join(f'<tr><td>{link(s["path"], s["path"])}</td><td><code>{s["sha256"]}</code></td></tr>'
                   for s in evidence["sources"])
    resource_bars = bars([
        (f'Wall time · {r["cumulative_seconds"]/3600:.2f} / {r["authorized_limit_seconds"]/3600:.0f} h',
         100*r["cumulative_seconds"]/r["authorized_limit_seconds"], False),
        (f'Storage · {r["storage_current_bytes"]/1024**3:.2f} / 20 GiB',
         100*r["storage_current_bytes"]/r["storage_limit_bytes"], False),
        (f'Acquired video · {r["video_transfer_and_interruption_upper_bound_bytes"]/1024**3:.3f} / 1 GiB',
         100*r["video_transfer_and_interruption_upper_bound_bytes"]/r["video_limit_bytes"], False),
    ])
    return wrap('<span class="eyebrow">Evidence and provenance</span><h1>Every chart has a source</h1>'
                '<span class="badge">SAVED-RECORD CHECKS PASSED</span><p>Summary hashes match the saved verifiers. '
                'All displayed E1/E2 accuracy values are independently recomputed from the anonymous public '
                'correctness tables. The frozen E2A gate is recalculated on load.</p>'
                '<div class="notice">These are historical observations, not new VidFit benchmark measurements. '
                'VidFit preview uses its own sampling path and is presented separately from frozen E0/E1/E2 inputs.</div>'
                f'<h2>Historical resource snapshot</h2><p class="muted">Recorded {esc(r["at"])}. '
                'This is the experiment closeout snapshot, not a live remaining-budget meter. It includes inherited '
                'historical charges. Presentation work is accounted separately without editing frozen ledgers.</p>'
                f'<div class="chart">{resource_bars}</div><details><summary>Inspect source paths and SHA-256 hashes</summary>'
                f'<div class="tablewrap"><table><thead><tr><th>Public source</th><th>SHA-256</th></tr></thead><tbody>{rows}</tbody></table></div></details>'
                '<p>Exports contain public aggregates and anonymous correctness only. Video pixels, annotations, '
                'reference labels, raw generated answers and private manifests stay outside the export.</p>'
                f'<p class="source">Source-set fingerprint: <code>{evidence["source_fingerprint"]}</code></p>')


def export_dashboard(evidence, output):
    """Create a self-contained HTML view and public-safe JSON; never write into results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels = [("overview", "Overview", overview(evidence)), ("e0", "E0", e0_panel(evidence)),
              ("e1", "E1", e1_panel(evidence)), ("e2", "E2A / E2B", e2_panel(evidence)),
              ("evidence", "Evidence", evidence_panel(evidence))]
    nav = '<nav class="nav" aria-label="Experiments">' + ''.join(
        f'<a href="#{key}">{label}</a>' for key, label, _ in panels) + '</nav>'
    body = ''.join(f'<section id="{key}">{panel}</section>' for key, _, panel in panels)
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>VidFit · video-compute-value experiments</title>'
                f'<style>body{{background:#f4f8fa;margin:0;padding:24px}}{CSS}</style></head>'
                f'<body><main class="vcv">{nav}{body}</main></body></html>')
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return output / "index.html"


def preview_frames(video, budget):
    """Exercise VidFit's real CPU preview, with no model or processor loading."""
    from vidfit import SelectionSpec, VideoAnalyzer
    from vidfit.core.video import evidence_image

    _require(video is not None, "Choose a local video first")
    _require(budget in (4, 8, 16, 32, 48, 64), "Choose a supported preview budget")
    info, frames, _ = VideoAnalyzer(backend="inspect").preview(video, SelectionSpec.uniform(int(budget)))
    images = [(evidence_image(f), f"#{f.frame_index} · {f.timestamp_s:.3f} s") for f in frames]
    identities = [{"frame_index": f.frame_index, "timestamp_s": f.timestamp_s,
                   "sha256_rgb": f.sha256_rgb} for f in frames]
    return images, {"mode": "VidFit CPU inspection; no model answer or accuracy",
                    "selected_frames": len(frames), "duration_s": info["duration_s"], "frames": identities}


def build_app(evidence, export_path):
    import gradio as gr
    import vidfit

    with gr.Blocks(title="VidFit · E0 / E1 / E2", analytics_enabled=False, fill_width=True) as app:
        gr.Markdown(f"**VidFit {vidfit.__version__} · video-compute-value** · saved experiment browser")
        with gr.Tab("Overview"):
            gr.HTML(overview(evidence))
        with gr.Tab("E0 · Measurement"):
            metric = gr.Dropdown(choices=["Accuracy", "Request time (descriptive)", "Timing repetition range",
                                          "Peak allocated memory"], value="Accuracy", label="Display metric")
            chart0 = gr.HTML(e0_panel(evidence))
            metric.change(lambda m: e0_panel(evidence, m), metric, chart0, queue=False)
        with gr.Tab("E1 · Headroom"):
            gr.HTML(e1_panel(evidence))
        with gr.Tab("E2 · Prediction"):
            display = gr.Radio(["Accuracy", "Outer-fold gains"], value="Accuracy", label="Compare")
            chart2 = gr.HTML(e2_panel(evidence))
            display.change(lambda d: e2_panel(evidence, d), display, chart2, queue=False)
        with gr.Tab("Evidence / export"):
            gr.HTML(evidence_panel(evidence))
            gr.File(value=str(export_path), label="Self-contained public-safe HTML", interactive=False)
            gr.File(value=str(export_path.parent / "evidence.json"), label="Public-safe chart data", interactive=False)
        with gr.Tab("VidFit · Inspect frames"):
            gr.Markdown("## See the source frames\n"
                        "Local CPU inspection through VidFit. Choose a local video and preview uniform sampling. "
                        "This is separate from the frozen experiment inputs; the gallery is not included in the historical export.")
            # Preserve the uploaded file: include_audio=False would invoke external ffmpeg
            # to create a muted copy. VidFit's frame decoder itself ignores audio.
            video = gr.Video(label="Local video", sources=["upload"])
            budget = gr.Dropdown([4, 8, 16, 32, 48, 64], value=8, label="Preview frame budget")
            preview = gr.Button("Preview with VidFit", variant="primary")
            gallery = gr.Gallery(label="Actual source frames · index and timestamp", columns=4, height=520)
            identities = gr.JSON(label="Source frame identities · no model answers")
            preview_event = preview.click(preview_frames, [video, budget], [gallery, identities], concurrency_limit=1)
            for control in (video, budget):
                control.change(lambda: ([], None), outputs=[gallery, identities],
                               cancels=[preview_event], queue=False)
    return app.queue(default_concurrency_limit=1, max_size=4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vidfit-root", type=Path, help="Optional local VidFit checkout; no installation changes")
    parser.add_argument("--serve", action="store_true", help="Launch loopback-only VidFit/Gradio view")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output_dir or root / "build/vidfit-experiments").resolve()
    _require(not output.is_relative_to(root / "results"), "Output must not modify frozen results")
    evidence = load_evidence(root)
    path = export_dashboard(evidence, output)
    print(f"Saved-record verification passed. HTML: {path}", flush=True)
    if not args.serve:
        return 0
    if args.vidfit_root:
        sys.path.insert(0, str(args.vidfit_root.resolve() / "src"))
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    temp = root / ".cache/vidfit-ui"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(temp)
    os.environ["TMP"] = str(temp)
    os.environ["TEMP"] = str(temp)
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key, ""), "127.0.0.1", "localhost", "::1"]))
    app = build_app(evidence, path)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
               css=".gradio-container {max-width:1400px !important;} body {background:#f4f8fa;}",
               theme="default", show_error=True, max_file_size="1000mb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

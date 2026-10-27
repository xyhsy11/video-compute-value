"""Reproducible descriptive quality addendum; no question text or labels are published."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from video_compute_value.common import atomic_text, command_output, digest, exclusive_lock, file_hash, read_json, write_json
from video_compute_value.resources import Ledger
from video_compute_value.runner import records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root, run = Path.cwd(), args.run_dir.resolve()
    if not run.is_relative_to(root):
        raise SystemExit("Run must be inside the project")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    with exclusive_lock(run / ".lock"):
        ledger = Ledger(run / "resource_ledger.json", root, config)
        with ledger.track("descriptive-quality-addendum"):
            verification, summary = read_json(run / "verification.json"), read_json(run / "summary.json")
            if not verification["records_valid"] or not summary["all_collection_complete"]:
                raise SystemExit("A complete, independently verified saved collection is required")
            all_rows = records(run)
            repeat_groups = defaultdict(list)
            for row in all_rows:
                if row["budget"]:
                    repeat_groups[(row["source_id"], row["question_id"], row["budget"])].append(row)
            audit = [{"source_id": source, "question_id": question, "budget": budget,
                      "repetitions": len(group),
                      "identical_prepared_inputs": len({digest([r["input_tensor_hashes"], r["input_metadata_hash"]]) for r in group}) == 1,
                      "identical_parsed_answers": len({r["parsed_answer"] for r in group}) == 1,
                      "identical_generated_tokens": len({tuple(r["generated_token_ids"]) for r in group}) == 1}
                     for (source, question, budget), group in sorted(repeat_groups.items())]
            flagged = [cell for cell in audit if not cell["identical_parsed_answers"] or not cell["identical_prepared_inputs"]]
            write_json(run / "repeatability_assessment.json", {"cells": audit, "flagged_cells": flagged,
                       "input_mismatch_cells": sum(not c["identical_prepared_inputs"] for c in audit),
                       "answer_disagreement_cells": sum(not c["identical_parsed_answers"] for c in audit)})
            rows = [r for r in all_rows if r["repetition"] == 1]
            video = [r for r in rows if r["budget"]]
            text = {r["question_id"]: r for r in rows if not r["budget"]}
            groups = defaultdict(list)
            for row in video:
                groups[(row["source_id"], row["question_id"])].append(row)
            questions = [{"source_id": source, "question_id": question,
                          "incorrect_video_conditions_out_of_5": sum(not r["correct"] for r in group),
                          "invalid_video_conditions_out_of_5": sum(not r["valid_answer"] for r in group),
                          "text_only_correct": text[question]["correct"]}
                         for (source, question), group in sorted(groups.items())]
            invalid = sum(not r["valid_answer"] for r in video)
            text_invalid = sum(not r["valid_answer"] for r in text.values())
            text_correct = sum(r["correct"] for r in text.values())
            video_conditions = [r for r in summary["conditions"] if r["budget"]]
            data = {"video_rep1_valid_outputs": len(video) - invalid, "video_rep1_outputs": len(video),
                    "text_valid_outputs": len(text) - text_invalid, "text_outputs": len(text),
                    "text_correct_out_of_24": text_correct, "questions": questions,
                    "code_sha256": file_hash(__file__),
                    "analysis_git_revision": command_output(["git", "rev-parse", "HEAD"]).get("stdout"),
                    "denominators": "The 120 video outputs are five conditions on the SAME 24 questions in eight videos; not 120 independent examples"}
            write_json(run / "quality_assessment.json", data)
            accuracies = [r["accuracy"] for r in video_conditions]
            usable = f"{len(video) - invalid}/{len(video)} repetition-1 video outputs met the fixed answer format; text-only: {len(text) - text_invalid}/24."
            comparison = (f"Video accuracy ranged from {min(accuracies):.1%} to {max(accuracies):.1%} across the five budgets; "
                          f"text-only scored {text_correct}/24 ({text_correct / 24:.1%}).")
            table = ["| Source | Question | Incorrect video conditions / 5 | Invalid conditions / 5 | Text-only correct |",
                     "|---|---|---:|---:|---|"]
            for q in questions:
                table.append(f"| {q['source_id']} | {q['question_id']} | {q['incorrect_video_conditions_out_of_5']} | "
                             f"{q['invalid_video_conditions_out_of_5']} | {q['text_only_correct']} |")
            persistent = [q["question_id"] for q in questions if q["incorrect_video_conditions_out_of_5"] == 5]
            stability = (f"\n## Repeatability and timing audit\n\n"
                f"Prepared-input mismatches: {sum(not c['identical_prepared_inputs'] for c in audit)}/{len(audit)} cells. "
                f"Parsed-answer disagreements: {sum(not c['identical_parsed_answers'] for c in audit)}/{len(audit)} cells.\n\n")
            if flagged:
                stability += "Flagged question/budget cells: " + ", ".join(
                    f"{c['question_id']} at Uniform-{c['budget']}" for c in flagged) + ".\n\n"
            unstable_budgets = [r["budget"] for r in video_conditions if not r["timing_stability_target_passed"]]
            stability += f"Budgets exceeding the proposed median relative timing range of 20%: {unstable_budgets or 'none'}. "
            stability += "Individual timing cells above 50% remain flagged in `summary.json`; completed trials were never selectively replaced.\n"
            assessment = (stability + "\n## Descriptive task-quality assessment\n\n" + usable + "\n\n" + comparison +
                " These are descriptive comparisons on 24 questions clustered in eight videos. Repetitions do not increase sample size. "
                "Useful visual reasoning, adaptive sampling, and statistical generalization are not established.\n\n" +
                f"Questions incorrect under every video budget: {', '.join(persistent) if persistent else 'none'}. "
                "Every question is listed below; errors are counted over the five conditions, without selecting repetitions.\n\n" +
                "\n".join(table) + "\n\nReproduce this addendum with `python scripts/report_quality.py --config configs/e000.yaml --run-dir " +
                str(run.relative_to(root)).replace("\\", "/") + "`. Its code hash is recorded in `quality_assessment.json`.\n")
            atomic_text(run / "quality_assessment.md", assessment.lstrip())
            marker = "\n<!-- quality-assessment -->\n"
            base = (run / "report.md").read_text(encoding="utf-8").split(marker)[0]
            final_status = read_json(run / "status.json")["status"]
            next_actions = {
                "FEASIBILITY_CONFIRMED": "Preserve these sources as development exposure and specify a separate experiment on new sources to test the research claim.",
                "REPEATABILITY_UNRESOLVED": "Investigate the flagged question/budget cells in a separately named deterministic-inference diagnostic; preserve every original trial.",
                "MEASUREMENT_UNSTABLE": "Investigate flagged timing cells and machine state in a separately named diagnostic; do not selectively replace slow trials.",
                "INPUT_INTEGRITY_FAILED": "Diagnose the recorded input-integrity failures before any new named collection.",
            }
            base = base.replace("Next action: resolve the recorded blocker, then run `prepare`, `smoke`, and `run`\n"
                                "using the exact commands in the README. Use `resume` after interrupted collection.",
                                "Next action: " + next_actions.get(final_status, "Resolve the remaining verification gates using the preserved records."))
            base = base.replace("only once real records exist. Real GPU paths remain unverified until exercised.",
                                "in the following saved-record addendum. The real GPU path was exercised for all required requests.")
            base = base.replace("Frozen provenance files and raw trials remain local because annotations and labels\n"
                                "are restricted; safe source identifiers/hashes and aggregate reports may be committed.",
                                "Question/label-bearing cohort and raw-trial artifacts remain local. Settings, source\n"
                                "identifiers/hashes, exposure identifiers and aggregate reports may be committed.")
            atomic_text(run / "report.md", base + marker + assessment)
    print(run / "quality_assessment.md")


if __name__ == "__main__":
    main()

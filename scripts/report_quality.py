"""Reproducible descriptive quality addendum; no question text or labels are published."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from video_compute_value.common import atomic_text, command_output, exclusive_lock, file_hash, read_json, write_json
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
            rows = [r for r in records(run) if r["repetition"] == 1]
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
            assessment = ("\n## Descriptive task-quality assessment\n\n" + usable + "\n\n" + comparison +
                " These are descriptive comparisons on 24 questions clustered in eight videos. Repetitions do not increase sample size. "
                "Useful visual reasoning, adaptive sampling, and statistical generalization are not established.\n\n" +
                f"Questions incorrect under every video budget: {', '.join(persistent) if persistent else 'none'}. "
                "Every question is listed below; errors are counted over the five conditions, without selecting repetitions.\n\n" +
                "\n".join(table) + "\n\nReproduce this addendum with `python scripts/report_quality.py --config configs/e000.yaml --run-dir " +
                str(run.relative_to(root)).replace("\\", "/") + "`. Its code hash is recorded in `quality_assessment.json`.\n")
            atomic_text(run / "quality_assessment.md", assessment.lstrip())
            marker = "\n<!-- quality-assessment -->\n"
            base = (run / "report.md").read_text(encoding="utf-8").split(marker)[0]
            atomic_text(run / "report.md", base + marker + assessment)
    print(run / "quality_assessment.md")


if __name__ == "__main__":
    main()

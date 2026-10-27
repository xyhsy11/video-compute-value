"""Frozen protocol constants: labels are deliberately absent from inference objects."""

import random
import re
from dataclasses import dataclass

from .common import Blocked, digest

PROMPT = "Choose the best option for the question.\nReply with one letter only: A, B, C, or D."
PARSER_VERSION = "strict-option-v1"
ANSWER_RE = re.compile(r"\s*(?:Answer:\s*)?([ABCD])[.!]?\s*", re.IGNORECASE)
MEASUREMENT_VERSION = "resident-local-file-per-question-host-sync-v1"


@dataclass(frozen=True)
class InferenceQuestion:
    question_id: str
    question: str
    options: tuple[str, str, str, str]


def question_input(record):
    options = tuple(record["options"])
    if len(options) != 4:
        raise ValueError("Exactly four official options required")
    return InferenceQuestion(str(record["question_id"]), record["question"], options)


def video_input(record):
    """Whitelist decoder metadata; full cohort entries also contain private labels."""
    allowed = ("path", "file_size", "selections", "timestamps", "duration", "frame_count", "height", "width",
               "target_height", "target_width", "fps")
    return {key: record[key] for key in allowed}


def make_prompt(question: InferenceQuestion):
    return f"{PROMPT}\n\nQuestion:\n{question.question}\n\n" + "\n".join(question.options)


def parse_answer(output):
    match = ANSWER_RE.fullmatch(output)
    return match.group(1).upper() if match else None


def validate_config(config):
    expected = {
        "experiment_id": "e000_frozen_videoqa_feasibility_v1",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct", "dataset": "Video-MME v1",
        "duration_category": "short", "num_videos": 8, "questions_per_video": 3,
        "frame_budgets": [8, 16, 32, 48, 64], "reference_budget": 64, "repetitions": 3,
        "batch_size": 1, "seed": 20261027, "max_new_tokens": 8, "do_sample": False,
        "num_beams": 1, "max_pixels_per_frame": 65536, "attention_backend": "sdpa",
        "cumulative_execution_limit_hours": 4, "additional_disk_limit_gib": 20,
        "video_acquisition_limit_gib": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise Blocked("INPUT_INTEGRITY_FAILED", f"{key} changes Experiment 0; use a new version")
    return config


def make_order(config, cohort):
    rng = random.Random(config["seed"])
    budgets = list(config["frame_budgets"])
    rng.shuffle(budgets)
    blocks = []
    questions = [(v["source_id"], q["question_id"]) for v in cohort["videos"] for q in v["questions"]]
    for repetition in range(1, config["repetitions"] + 1):
        rotated = budgets[repetition - 1:] + budgets[:repetition - 1]
        for budget in rotated:
            order = list(questions)
            rng.shuffle(order)
            blocks.append({"budget": budget, "repetition": repetition, "questions": order})
    blocks.append({"budget": 0, "repetition": 1, "questions": questions})
    return blocks


def trial_identity(run_id, config_hash, source_id, question_id, budget, repetition):
    value = {"run_id": run_id, "config_hash": config_hash, "source_id": source_id,
             "question_id": question_id, "budget": budget, "repetition": repetition}
    return {**value, "trial_id": digest(value)}

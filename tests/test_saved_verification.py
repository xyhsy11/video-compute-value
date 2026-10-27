"""Corrupt synthetic saved records must fail independent verification.

All files live under pytest's temporary directory, never real experiment results.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from video_compute_value.analysis import analyze, verify
from video_compute_value.common import digest, write_json
from video_compute_value.protocol import make_order, trial_identity


@pytest.mark.parametrize("corruption", [None, "tokens", "timestamps", "receipt", "memory"])
def test_independent_verifier_detects_corruption(tmp_path, monkeypatch, corruption):
    import transformers
    import video_compute_value.runner as runner
    config = yaml.safe_load((Path(__file__).parents[1] / "configs" / "e000.yaml").read_text())
    run = tmp_path / "synthetic-saved-record-fixture"
    run.mkdir()
    videos = []
    for i in range(8):
        questions = [{"question_id": f"q{i}-{j}", "question": "Synthetic test question",
                      "options": ["A. a", "B. b", "C. c", "D. d"], "answer": "A"} for j in range(3)]
        videos.append({"source_id": f"v{i}", "questions": questions, "target_height": 32, "target_width": 32,
                       "frame_count": 64, "fps": "1", "selections": {"8": {
                           "indices": [4, 12, 20, 28, 36, 44, 52, 60],
                           "timestamps": [str(n) for n in (4, 12, 20, 28, 36, 44, 52, 60)]}}})
    cohort = {"videos": videos}
    write_json(run / "cohort_manifest.json", cohort)
    write_json(run / "execution_order.json", make_order(config, cohort))
    monkeypatch.setattr(runner, "validate_frozen", lambda *a, **k: {"config_hash": "synthetic", "model_path": "."})
    tokenizer = SimpleNamespace(decode=lambda ids, **k: "A" if ids == [1] else "B")
    processor = SimpleNamespace(tokenizer=tokenizer,
        _calculate_timestamps=lambda indices, fps, merge: [(indices[i] + indices[i + 1]) / (2 * fps)
                                                           for i in range(0, len(indices), 2)])
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    row = {**trial_identity(run.name, "synthetic", "v0", "q0-0", 8, 1),
           "kind": "trial", "call_id": "synthetic-call", "generated_token_ids": [1], "decoded_output": "A",
           "parsed_answer": "A", "valid_answer": True, "correct": True, "output_tokens": 1,
           "input_tokens": 14, "visual_tokens": 4, "tensor_shapes": {"input_ids": [1, 14]},
           "latency_seconds": 1.0, "baseline_allocated_bytes": 10, "baseline_reserved_bytes": 20,
           "peak_allocated_bytes": 30, "peak_reserved_bytes": 40,
           "incremental_peak_allocated_bytes": 20, "incremental_peak_reserved_bytes": 20,
           "frames_decoded": 64, "frames_selected": 8, "frames_processor": 8, "frames_represented": 8,
           "temporal_padding": 0, "source_indices": videos[0]["selections"]["8"]["indices"],
           "actual_timestamps": videos[0]["selections"]["8"]["timestamps"],
           "model_timestamps": [8.0, 24.0, 40.0, 56.0], "model_timestamp_text": ["8.0", "24.0", "40.0", "56.0"],
           "video_grid_thw": [[4, 2, 2]], "input_tensor_hashes": {"synthetic": "fixture"}}
    row["input_metadata_hash"] = digest({k: row[k] for k in
        ("tensor_shapes", "source_indices", "actual_timestamps", "model_timestamps", "video_grid_thw")})
    events = [{"call_id": row["call_id"], "kind": "trial", "event": "started", "trial_id": row["trial_id"], "budget": 8},
              {"call_id": row["call_id"], "kind": "trial", "event": "completed", "model_invoked": True, "elapsed_seconds": 1.1}]
    if corruption == "tokens":
        row["generated_token_ids"] = [2]
    elif corruption == "timestamps":
        row["model_timestamp_text"][0] = "999.0"
    elif corruption == "receipt":
        events = []
    elif corruption == "memory":
        row["incremental_peak_allocated_bytes"] = 123
    write_json(run / "records" / (row["trial_id"] + ".json"), row)
    (run / "worker_calls.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    analyze(config, run)
    verification = verify(config, run)
    assert verification["records_valid"] is (corruption is None)
    assert not verification["complete_collection"]  # One synthetic row is never a complete result.

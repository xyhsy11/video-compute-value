"""Synthetic fixtures test code paths only; they are never experiment results."""

from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from video_compute_value.analysis import summarize
from video_compute_value.common import Blocked, digest, exclusive_lock, read_json, write_json
from video_compute_value.dataset import load_annotations, ranked_sources
from video_compute_value.measurement import timed_request
from video_compute_value.protocol import make_order, make_prompt, parse_answer, question_input, trial_identity, video_input
from video_compute_value.resources import Ledger
from video_compute_value.video import decode_selected, inspect_video, letterbox, spatial_layout, spatial_size, uniform_indices


@pytest.mark.parametrize("text,expected", [("A", "A"), (" Answer: b.\n", "B"), ("D!", "D"),
    ("", None), ("A or B", None), ("A because it is visible", None), ("Answer: C\nD", None),
    ("The answer is A", None), ("AB", None), ("<A>", None)])
def test_strict_generated_output_parser(text, expected):
    assert parse_answer(text) == expected


def test_prompt_cannot_leak_label():
    labelled = {"question_id": "q1", "question": "Which visible shape?", "options": ["A. square", "B. circle", "C. oval", "D. cube"], "answer": "C", "secret": "private"}
    question = question_input(labelled)
    assert set(asdict(question)) == {"question_id", "question", "options"}
    assert "private" not in make_prompt(question)
    assert parse_answer("A or B") is None  # Letters in the prompt cannot rescue an invalid generated answer.
    video = {key: None for key in ("path", "file_size", "selections", "timestamps", "duration", "frame_count",
                                   "height", "width", "target_height", "target_width", "fps",
                                   "content_height", "content_width", "pad_top", "pad_left")}
    video.update(questions=[labelled], label_hash="secret")
    assert "questions" not in video_input(video) and "label_hash" not in video_input(video)


def test_fractional_pts_and_earlier_tie():
    assert uniform_indices([0, 1, 2, 3], 4, 2) == [1, 3]
    assert uniform_indices([0, 2], 4, 1) == [1]  # Target 2 is an exact match.
    assert uniform_indices([0, 2], 2.5, 1) == [1]
    assert uniform_indices([0, 2, 4], 6, 1) == [1]  # Target 3 is tied; choose timestamp 2.
    assert uniform_indices([Fraction(0), Fraction(2, 3)], Fraction(4, 3), 1) == [1]
    assert uniform_indices([0, 1], 2, 2) == [0, 1]  # Targets .5, 1.5; tie chooses earlier.


def test_nonmonotonic_and_duplicate_sampling_rejected():
    for times in ([0, 0, 1], [0, 2, 1], [-1, 0, 1]):
        with pytest.raises(ValueError):
            uniform_indices(times, 3, 2)
    with pytest.raises(ValueError, match="distinct"):
        uniform_indices([0, 1], 2, 4)


@pytest.mark.parametrize("height,width", [(1080, 1920), (1920, 1080), (480, 640), (32, 6400)])
def test_spatial_grid_has_fixed_per_frame_cap(height, width):
    h, w = spatial_size(height, width)
    assert h % 32 == w % 32 == 0
    assert h * w <= 65536
    assert spatial_size(height, width) == (h, w)  # Budget is deliberately not an argument.


def test_official_rows_keep_question_option_order_and_outcome_independent_rank(tmp_path):
    rows = [{"videoID": source, "video_id": str(i), "duration": "short", "question_id": f"{source}-{j}",
             "question": f"Original {j}", "options": ["A. a", "B. b", "C. c", "D. d"], "answer": "A"}
            for i, source in enumerate(("source-a", "source-b", "source-c")) for j in range(3)]
    file = tmp_path / "annotations.json"
    write_json(file, rows)
    first = load_annotations(file)
    rank = ranked_sources(first, 20261027)
    for row in rows:
        row["answer"] = "D"
    write_json(file, rows)
    assert ranked_sources(load_annotations(file), 20261027) == rank
    assert first["source-a"][1]["question"] == "Original 1"
    assert first["source-a"][1]["options"] == rows[1]["options"]


def test_letterboxing_preserves_16_by_9_content_and_fixed_black_padding():
    from PIL import Image, ImageDraw
    layout = spatial_layout(720, 1280)
    assert layout["content_width"] / layout["content_height"] == 16 / 9
    assert (layout["target_height"], layout["target_width"]) == (192, 320)
    image = Image.new("RGB", (1280, 720), "white")
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((460, 180, 820, 540), fill="red")
    output = letterbox(image, layout)
    assert np.all(output[:6] == 0) and np.all(output[-6:] == 0)
    red = (output[:, :, 0] > 200) & (output[:, :, 1] < 50) & (output[:, :, 2] < 50)
    yy, xx = np.where(red)
    assert abs((xx.max() - xx.min()) - (yy.max() - yy.min())) <= 1


def test_sequential_decode_actual_frame_counts_and_reproducible_pixels(tmp_path):
    import av
    path = tmp_path / "测试 video with spaces.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=16)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(96):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    video = inspect_video(path, [8, 16, 32, 48, 64], 65536)
    first, details = decode_selected(video, 64)
    second, details2 = decode_selected(video, 64)
    assert first.shape == (64, 64, 64, 3)
    assert details["frames_decoded"] == 96 and details["frames_selected"] == 64
    assert details == details2 and np.array_equal(first, second)


def test_native_pinned_video_processor_does_not_resample_or_resize():
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from transformers.video_utils import VideoMetadata
    from video_compute_value.model import validate_grid
    processor = Qwen3VLVideoProcessor()
    for budget in (8, 16, 32, 48, 64):
        result = processor(videos=[torch.zeros(budget, 3, 160, 320, dtype=torch.uint8)],
                           video_metadata=[VideoMetadata(total_num_frames=200, fps=25, duration=8,
                                                         frames_indices=list(range(budget)))],
                           do_resize=False, do_sample_frames=False, return_tensors="pt")
        grid = result["video_grid_thw"].tolist()
        validate_grid(grid, budget, 160, 320, budget // 2 * 10 * 20 // 4)
        assert result["pixel_values_videos"].shape == (budget // 2 * 10 * 20, 3 * 2 * 16 * 16)


def test_primary_measurement_boundary_excludes_hashing_and_scoring():
    events = []
    class CUDA:
        def synchronize(self):
            events.append("sync")
        def reset_peak_memory_stats(self):
            events.append("reset")
        def memory_allocated(self):
            return 10
        def memory_reserved(self):
            return 20
        def max_memory_allocated(self):
            return 30
        def max_memory_reserved(self):
            return 40
    class Adapter:
        class Torch:
            cuda = CUDA()
        torch = Torch()
        def prepare(self, *args):
            events.append("open_decode_select_resize_tokenize")
            return {"input": "cpu"}, {}
        def transfer(self, inputs):
            events.append("transfer")
            return inputs
        def generate(self, inputs):
            events.append("generate_cpu_output_decode_parse")
            return {"parsed_answer": "A"}
    ticks = iter((100, 1_000_000_100))
    def clock():
        events.append("timer")
        return next(ticks)
    record, cpu = timed_request(Adapter(), None, None, 8, clock=clock)
    assert events == ["sync", "reset", "timer", "open_decode_select_resize_tokenize", "transfer",
                      "generate_cpu_output_decode_parse", "sync", "timer"]
    assert record["latency_seconds"] == 1 and record["peak_allocated_bytes"] == 30
    assert record["incremental_peak_allocated_bytes"] == 20 and cpu == {"input": "cpu"}


def test_atomic_records_and_duplicate_prevention(tmp_path):
    from video_compute_value.runner import records
    identity = trial_identity("run", "hash", "v", "q", 8, 1)
    write_json(tmp_path / "records" / "one.json", identity)
    assert records(tmp_path) == [identity]
    write_json(tmp_path / "records" / "two.json", identity)
    with pytest.raises(Blocked, match="Duplicate"):
        records(tmp_path)
    with exclusive_lock(tmp_path / ".lock"):
        with pytest.raises(Blocked, match="Another process"):
            with exclusive_lock(tmp_path / ".lock"):
                pass


def test_ledgers_preserve_prior_resource_cost_across_resumptions(tmp_path):
    config = {"cumulative_execution_limit_hours": 4, "additional_disk_limit_gib": 20,
              "video_acquisition_limit_gib": 1}
    path = tmp_path / "ledger.json"
    ledger = Ledger(path, tmp_path, config)
    with ledger.track("failed attempt"):
        ledger.charge_transfer(123, video=True)
    elapsed = read_json(path)["active_seconds"]
    resumed = Ledger(path, tmp_path, config)
    with resumed.track("resume"):
        assert resumed.value["video_transfer_bytes"] == 123
        assert resumed.consumed_seconds() >= elapsed
        with pytest.raises(Blocked):
            resumed.allow_video_transfer(1024 ** 3)


def synthetic_rows():
    rows = []
    for budget in (8, 16, 32, 48, 64, 0):
        for q in range(24):
            for rep in range(1, 2 if budget == 0 else 4):
                rows.append({**trial_identity("synthetic", "test", f"v{q // 3}", f"q{q}", budget, rep),
                    "correct": rep != 1, "valid_answer": True, "parsed_answer": "A", "generated_token_ids": [1],
                    "decoded_output": "A", "latency_seconds": float(rep), "visual_tokens": budget,
                    "input_tokens": budget + 10, "output_tokens": 1, "peak_allocated_bytes": 20,
                    "peak_reserved_bytes": 30, "input_tensor_hashes": {"x": "same"}, "input_metadata_hash": "same"})
    return rows


def test_analysis_fixed_denominators_rep1_only_and_question_medians():
    config = {"num_videos": 8, "questions_per_video": 3, "reference_budget": 64,
              "frame_budgets": [8, 16, 32, 48, 64], "repetitions": 3}
    result = summarize(config, synthetic_rows(), [], [])
    assert result["successful_video_calls"] == 360 and result["successful_text_calls"] == 24
    for row in result["conditions"][:-1]:
        assert row["accuracy"] == 0  # Repetitions 2/3 are correct; cannot improve primary accuracy.
        assert row["latency_seconds"]["median"] == 2
        assert row["median_relative_timing_range"] == 1
        assert not row["timing_stability_target_passed"]
    removed = synthetic_rows()[0]
    incomplete = synthetic_rows()[1:]
    failure = {**removed, "kind": "trial"}
    first = summarize(config, incomplete, [failure], [])["conditions"][0]
    assert first["accuracy"] is None and first["latency_seconds"]["median"] is None
    assert first["planned"] == 72 and first["attempted"] == 72 and first["failed_without_success"] == 1
    assert first["unattempted"] == 0


def test_order_identity_are_deterministic_and_budget_order_rotates():
    config = {"seed": 20261027, "frame_budgets": [8, 16, 32, 48, 64], "repetitions": 3}
    cohort = {"videos": [{"source_id": f"v{i}", "questions": [{"question_id": f"q{i}-{j}"} for j in range(3)]} for i in range(8)]}
    order = make_order(config, cohort)
    assert order == make_order(config, cohort) and sum(len(b["questions"]) for b in order) == 384
    assert order[0]["budget"] != order[5]["budget"] != order[10]["budget"]
    assert digest(order) != digest(order[::-1])

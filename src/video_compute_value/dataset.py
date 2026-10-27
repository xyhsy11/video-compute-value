"""Original Video-MME annotations, outcome-independent cohort selection, private labels."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .acquire import acquire_video, archive_index
from .common import Blocked, digest, file_hash, read_json, utc_now, write_json
from .video import inspect_video


def load_annotations(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = read_json(path)
    grouped = defaultdict(list)
    for row in rows:
        # The official parquet is one question per row; official JSON is one video per entry.
        source_id = str(row.get("videoID", row.get("video_id", "")))
        if row.get("duration") != "short":
            continue
        if not source_id:
            raise ValueError("Annotation lacks original videoID")
        questions = row.get("questions", [row])
        for q in questions:
            answer = q["answer"]
            if answer not in ("A", "B", "C", "D") or len(q["options"]) != 4:
                raise ValueError("Invalid official multiple-choice annotation")
            grouped[source_id].append({"question_id": str(q["question_id"]), "question": q["question"],
                                       "options": list(q["options"]), "answer": answer,
                                       "dataset_video_id": str(row.get("video_id", source_id))})
    for source, questions in grouped.items():
        if len(questions) != 3 or len({q["question_id"] for q in questions}) != 3:
            raise ValueError(f"Expected three unique official questions for {source}")
    return dict(grouped)


def ranked_sources(annotations, seed):
    return sorted(annotations, key=lambda source: (digest({"seed": seed, "source_id": source}), source))


def prepare_cohort(config, run, root, ledger, annotation_path, metadata, video_dir=None, acquire=False):
    run = Path(run)
    if (run / "cohort_manifest.json").exists():
        cohort = read_json(run / "cohort_manifest.json")
        if cohort["config_hash"] != digest(config):
            raise Blocked("INPUT_INTEGRITY_FAILED", "Cohort configuration changed")
        validate_cohort(cohort, config)
        return cohort
    annotations = load_annotations(annotation_path)
    exclusions, selected = [], []
    index = archive_index(metadata, root, ledger) if acquire else {}
    directory = Path(video_dir) if video_dir else Path(root, "data", "videos")
    for rank, source in enumerate(ranked_sources(annotations, config["seed"])):
        ledger.check()
        matches = [directory / (source + ext) for ext in (".mp4", ".mkv", ".webm")]
        existing = [path for path in matches if path.exists()]
        if len(existing) > 1:
            exclusions.append({"source_id": source, "rank": rank, "reason": "multiple local files; ambiguous"})
            continue
        path = existing[0] if existing else None
        if path is None and acquire and source in index:
            try:
                path = acquire_video(source, index[source], metadata, root, ledger)
            except Blocked as exc:
                if exc.status == "BLOCKED_RESOURCE":
                    write_json(run / "private" / "selection_progress.json", {"selected": selected, "excluded": exclusions})
                    raise
                exclusions.append({"source_id": source, "rank": rank, "reason": str(exc)})
                continue
        if path is None:
            exclusions.append({"source_id": source, "rank": rank, "reason": "local file unavailable"})
            continue
        try:
            video = inspect_video(path, config["frame_budgets"], config["max_pixels_per_frame"])
        except (ValueError, OSError, RuntimeError) as exc:
            exclusions.append({"source_id": source, "rank": rank, "reason": f"decoding/timing eligibility: {exc}"})
            continue
        questions = annotations[source]
        video.update(source_id=source, parent_id=source, selection_rank=rank, questions=questions,
                     annotation_hash=digest(questions), label_hash=digest([q["answer"] for q in questions]),
                     acquisition_source={**metadata, "archives": None, "local_path": str(path)})
        selected.append(video)
        if len(selected) == config["num_videos"]:
            break
    write_json(run / "private" / "selection_progress.json", {"selected": selected, "excluded": exclusions})
    if len(selected) != config["num_videos"]:
        raise Blocked("BLOCKED_INPUT", f"Only {len(selected)}/8 eligible complete videos; see private selection_progress.json")
    cohort = {"frozen_at": utc_now(), "config_hash": digest(config), "dataset": metadata,
              "annotation_file_sha256": file_hash(annotation_path), "videos": selected, "excluded": exclusions}
    write_json(run / "cohort_manifest.json", cohort)
    write_json(run / "exposure_manifest.json", {"purpose": "development pilot; never untouched confirmation data",
                                               "sources": [{"source_id": v["source_id"], "sha256": v["sha256"]}
                                                           for v in selected]})
    write_json(run / "source_provenance.json", {"dataset": metadata, "annotation_file_sha256": file_hash(annotation_path),
                                               "sources": [{k: v[k] for k in ("source_id", "sha256", "annotation_hash", "label_hash")}
                                                           for v in selected], "excluded": exclusions})
    return cohort


def validate_cohort(cohort, config):
    videos = cohort["videos"]
    if len(videos) != config["num_videos"] or len({v["source_id"] for v in videos}) != len(videos):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Cohort size/uniqueness mismatch")
    question_ids = []
    for video in videos:
        if file_hash(video["path"]) != video["sha256"] or digest(video["questions"]) != video["annotation_hash"]:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Video or annotation changed")
        if len(video["questions"]) != config["questions_per_video"]:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Question count changed")
        question_ids.extend(q["question_id"] for q in video["questions"])
    if len(set(question_ids)) != len(question_ids):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Duplicate question IDs")

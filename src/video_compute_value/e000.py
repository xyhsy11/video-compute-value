"""Experiment 0 command line. Every public operation shares the same resource ledger."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml

from .common import Blocked, append_json, exclusive_lock, read_jsonl, utc_now, write_json
from .protocol import validate_config
from .resources import Ledger, environment


def set_local_paths(root):
    for name, relative in (("HF_HOME", ".cache/huggingface"), ("HF_HUB_CACHE", ".cache/huggingface/hub"),
                           ("MPLCONFIGDIR", ".cache/matplotlib"), ("TEMP", ".cache/tmp"), ("TMP", ".cache/tmp")):
        path = Path(root, relative).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def status(run, value, message):
    completed_model_calls = [event for event in read_jsonl(Path(run, "worker_calls.jsonl"))
                             if event.get("event") == "completed" and event.get("model_invoked")]
    write_json(Path(run, "status.json"), {"status": value, "message": message, "updated_at": utc_now(),
                                         "real_inference": bool(completed_model_calls),
                                         "completed_model_calls": len(completed_model_calls),
                                         "planned_video_calls": 360, "planned_text_calls": 24})


def preflight(config, run, root):
    record = environment(root)
    write_json(run / "environment.json", record)
    missing = [k for k in ("torch", "torchvision", "transformers", "accelerate") if not record["packages"].get(k)]
    if missing:
        raise Blocked("BLOCKED_ENVIRONMENT", "Missing inference packages: " + ", ".join(missing))
    if not record["cuda"] or not record["cuda"]["available"]:
        raise Blocked("BLOCKED_RESOURCE", "A CUDA GPU with the pinned PyTorch stack is required")
    status(run, "PARTIAL", "Preflight passed; cohort, fit, freeze and collection gates remain")


def prepare(args, config, run, root, ledger):
    from .acquire import dataset_metadata
    from .dataset import prepare_cohort
    if args.annotations:
        if not args.annotation_revision or len(args.annotation_revision) != 40:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Local annotations require --annotation-revision with immutable source SHA")
        annotation_path = Path(args.annotations).resolve()
        metadata = {"repo": config["annotation_repo"], "revision": args.annotation_revision,
                    "acquisition_source": "Existing local official annotations", "archives": []}
        if args.acquire:
            raise Blocked("BLOCKED_INPUT", "Use --acquire without --annotations to inspect the official archive index")
    else:
        annotation_path, metadata = dataset_metadata(config, root, ledger)
    prepare_cohort(config, run, root, ledger, annotation_path, metadata, args.video_dir, args.acquire)
    status(run, "PARTIAL", "Eight complete videos frozen; model fit and collection not yet validated")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("preflight", "prepare", "smoke", "run", "resume", "analyze", "verify", "worker", "account"))
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--run-dir", required=True, type=Path)
    result.add_argument("--root", type=Path, default=Path.cwd())
    result.add_argument("--annotations", type=Path)
    result.add_argument("--annotation-revision")
    result.add_argument("--video-dir", type=Path)
    result.add_argument("--acquire", action="store_true", help="Acquire only pilot members via verified HTTP ranges")
    result.add_argument("--block", type=int, help=argparse.SUPPRESS)
    result.add_argument("--mode", choices=("probe", "collect"), help=argparse.SUPPRESS)
    result.add_argument("--event-name", help="Description for an externally timed setup/validation event")
    result.add_argument("--event-seconds", type=float)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root, run = args.root.resolve(), args.run_dir.resolve()
    if not run.is_relative_to(root) or not args.config.resolve().is_relative_to(root):
        raise SystemExit("Configuration and run directory must be inside --root")
    config = validate_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    run.mkdir(parents=True, exist_ok=True)
    set_local_paths(root)
    os.environ["OMP_NUM_THREADS"] = str(config["cpu_threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["cpu_threads"])
    if args.command == "worker":
        from .runner import worker
        return worker(config, run, root, args.block, args.mode)
    with exclusive_lock(run / ".lock"):
        ledger = Ledger(run / "resource_ledger.json", root, config)
        with ledger.track(args.command):
            try:
                if args.command == "preflight":
                    preflight(config, run, root)
                elif args.command == "prepare":
                    prepare(args, config, run, root, ledger)
                elif args.command in ("smoke", "run", "resume"):
                    from .runner import coordinate
                    coordinate(args.command, config, run, root, ledger, args.config.resolve())
                elif args.command in ("analyze", "verify"):
                    from .analysis import analyze, verify
                    analyze(config, run)
                    if args.command == "verify":
                        if not verify(config, run, root)["records_valid"]:
                            return 3
                elif args.command == "account":
                    if args.event_seconds is None or args.event_seconds < 0 or not args.event_name:
                        raise ValueError("account requires nonnegative --event-seconds and --event-name")
                    ledger.value["active_seconds"] += args.event_seconds
                    ledger.value.setdefault("external_events", []).append({"name": args.event_name,
                        "active_seconds": args.event_seconds, "recorded_at": utc_now()})
                    ledger.check()
                print(f"{args.command}: {run}")
                return 0
            except Blocked as exc:
                status(run, exc.status, str(exc))
                append_json(run / "operational_events.jsonl", {"at": utc_now(), "command": args.command,
                                                              "status": exc.status, "message": str(exc)})
                print(f"{exc.status}: {exc}", file=sys.stderr)
                return 2
            except KeyboardInterrupt:
                status(run, "PARTIAL", "Interrupted; atomic completed records and resource ledger preserved")
                return 130
            except Exception as exc:
                status(run, "PARTIAL", f"{type(exc).__name__}: {exc}")
                append_json(run / "private" / "errors.jsonl", {"at": utc_now(), "command": args.command,
                                                              "traceback": traceback.format_exc()})
                print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
                return 1


if __name__ == "__main__":
    raise SystemExit(main())

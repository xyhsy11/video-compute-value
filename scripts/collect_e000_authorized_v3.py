"""Authorized local-only corrected collection. No Git mutation or new acquisition."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

from video_compute_value.common import command_output, digest, exclusive_lock, file_hash, read_json, utc_now, write_json
from video_compute_value.e000 import preflight, require_cumulative_ledger, set_local_paths, status
from video_compute_value.protocol import validate_config
from video_compute_value.resources import Ledger, environment

ROOT = Path.cwd()
RUN = ROOT / "results/e000_20261028_greedy_collection_v3"
CONFIG = ROOT / "configs/e000_threaded_authorized_v3.yaml"


def main():
    config = validate_config(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    assert read_json(RUN / "authorization.json")["authorized_limit_seconds"] == 19200
    authorized_seconds = 19200
    if (RUN / "runtime_resource_grant.json").exists():
        grant = read_json(RUN / "runtime_resource_grant.json")
        assert grant["human_reply"] == "Authorize 5h40m cumulative time"
        assert grant["authorized_limit_seconds"] == 20400
        assert grant["resource_only"] is True
        authorized_seconds = grant["authorized_limit_seconds"]
    set_local_paths(ROOT)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    require_cumulative_ledger(RUN, ROOT)
    with exclusive_lock(RUN / ".lock"):
        # The external human grant changes accounting only. Frozen experimental
        # configuration/source hashes and every worker's inference settings stay intact.
        resource_config = {**config, "cumulative_execution_limit_hours": authorized_seconds / 3600}
        ledger = Ledger(RUN / "resource_ledger.json", ROOT, resource_config)
        if not any(e["name"] == "Authorized collection initial inspection and source preparation" for e in ledger.value["external_events"]):
            ledger.value["active_seconds"] += 30
            ledger.value["external_events"].append({"name": "Authorized collection initial inspection and source preparation", "active_seconds": 30, "recorded_at": utc_now()})
        with ledger.track("authorized-v3-preflight-collection-verification"):
            try:
                private = RUN / "private"
                private.mkdir(exist_ok=True)
                isolation_authorization = read_json(RUN / "gpu_isolation_authorization.json")
                assert isolation_authorization["authorized_process_names"] == ["Hearthstone.exe", "NVIDIA Overlay.exe"]
                import psutil
                matched = [p for p in psutil.process_iter(["pid", "name"])
                           if p.info["name"] in isolation_authorization["authorized_process_names"]]
                before_gpu = command_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"])
                closed = [{"pid": p.pid, "name": p.info["name"]} for p in matched]
                for process in matched:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(matched, timeout=5)
                for process in alive:
                    process.kill()
                remaining_apps = [p.info for p in psutil.process_iter(["pid", "name"])
                                  if p.info["name"] in isolation_authorization["authorized_process_names"]]
                after_gpu = command_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
                memory_mib, utilization = [int(x.strip()) for x in after_gpu["stdout"].split(",")]
                low_idle_restart = (all(p["name"] == "NVIDIA Overlay.exe" for p in remaining_apps)
                                    and memory_mib < 2048 and utilization <= 5)
                write_json(private / f"gpu_isolation_{time.time_ns()}.json", {
                    "at": utc_now(), "closed_processes": closed, "remaining_authorized_apps": remaining_apps,
                    "before_gpu": before_gpu, "after_gpu": after_gpu,
                    "overlay_automatically_restarted_with_low_device_load": bool(remaining_apps) and low_idle_restart})
                assert not remaining_apps or low_idle_restart, "GPU applications or substantial competing device load remain active"
                print("Authorized apps closed; idle memory/load and any Overlay restart recorded", flush=True)
                validation_log = private / f"cpu_validation_{time.time_ns()}.log"
                with validation_log.open("w", encoding="utf-8") as output:
                    if authorized_seconds == 19200:
                        subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", ".cache/authorized-v3-tests"], check=True, stdout=output, stderr=output, timeout=120)
                    else:
                        previous = [p for p in private.glob("cpu_validation*.log") if p != validation_log]
                        assert any("75 passed" in p.read_text(encoding="utf-8") and "All checks passed!" in p.read_text(encoding="utf-8") for p in previous)
                        output.write("75-test passing receipt retained; core source equality checked below. No repeated CPU suite needed.\n")
                    subprocess.run([str(ROOT / ".venv/Scripts/ruff.exe"), "check", "src", "tests", "scripts"], check=True, stdout=output, stderr=output, timeout=60)
                print("Authorized collection: CPU tests and static checks passed", flush=True)
                recovering = (RUN / "resolved_config.yaml").exists()
                if recovering:
                    # A failed preflight is evidence, not a fresh allowance. Only
                    # bounded preflight is admitted here; coordinate() must still
                    # admit the entire collection using all five actual profiles.
                    ledger.check(projected_seconds=900)
                else:
                    prior = read_json(ROOT / "results/e000_20261028_completion_v1/projection.json")
                    ledger.check(projected_seconds=prior["projected_seconds"] - prior["setup_seconds"] * prior["safety_factor"])
                print(f"Preflight admitted; remaining {authorized_seconds - ledger.consumed_seconds():.1f}s", flush=True)
                cohort = read_json(RUN / "cohort_manifest.json")
                if recovering:
                    assert cohort["config_hash"] == digest(config)
                else:
                    cohort["config_hash"] = digest(config)
                    write_json(RUN / "cohort_manifest.json", cohort)
                sources = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted((ROOT / "src/video_compute_value").glob("*.py"))}
                for name in sources if not recovering else []:
                    path = private / "frozen_sources" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((ROOT / name).read_bytes())
                provenance = {"base_git_revision": command_output(["git", "rev-parse", "HEAD"])["stdout"],
                    "source_files_sha256": sources, "working_tree_intentionally_uncommitted": True,
                    "config_sha256": file_hash(CONFIG), "authorization_id": config["resource_authorization_id"]}
                if recovering:
                    assert read_json(RUN / "source_provenance.json") == provenance
                else:
                    write_json(RUN / "source_provenance.json", provenance)
                if recovering:
                    write_json(private / f"recovery_environment_{time.time_ns()}.json", environment(ROOT))
                else:
                    preflight(config, RUN, ROOT)
                from video_compute_value import runner
                coordinate = runner.coordinate
                if recovering:
                    launch = runner.launch
                    def bounded_launch(config_path, run, root, block, mode, timeout):
                        actual_timeout = min(config["worker_timeout_seconds"], max(1, int(authorized_seconds - ledger.consumed_seconds())))
                        return launch(config_path, run, root, block, mode, min(actual_timeout, 120) if mode == "probe" else actual_timeout)
                    runner.launch = bounded_launch
                    write_json(private / f"recovery_{time.time_ns()}.json", {
                        "at": utc_now(), "probe_worker_deadline_seconds": 120,
                        "scope": "Recovery probes only; frozen inputs, model, generation, trim and primary measurements unchanged",
                        "orchestrator_sha256": file_hash(Path(__file__)), "prior_failed_attempts_preserved": True,
                        "effective_resource_limit_seconds": authorized_seconds})
                if authorized_seconds == 20400 and (RUN / "probes.json").exists() and not (RUN / "collection_ready.json").exists():
                    frozen = runner.validate_frozen(config, RUN, ROOT)
                    probe_data = read_json(RUN / "probes.json")
                    assert probe_data["config_hash"] == frozen["config_hash"]
                    probes = probe_data["probes"]
                    assert {p["budget"] for p in probes} == set(config["frame_budgets"])
                    for probe in probes:
                        assert probe["config_hash"] == frozen["config_hash"]
                        assert probe["integrity_passed"] and probe["generation_repeatability_passed"]
                        runner.validate_probe_pair(probe["fit"], probe["repeat"])
                        for row in (probe["fit"], probe["repeat"], probe["component_profile"]):
                            assert row["generation_policy_sha256"] == digest(frozen["generation_policy"])
                            receipt = read_json(RUN / f"worker_{row['worker_key']}.json")
                            assert receipt["generation_policy"] == frozen["generation_policy"]
                            assert receipt["allocator_backend"] == frozen["cuda_allocator_backend"]
                            assert row["cuda_cache_trimmed"] == (probe["budget"] in frozen["cuda_trim_budgets"])
                            assert (row["decoder_threads"], row["decoder_thread_type"]) == (4, "FRAME")
                    estimate = runner.collection_projection(probes, read_json(RUN / "cpu_decode_probes.json")["videos"], frozen["profile_question"][0])
                    ledger.check(projected_seconds=estimate["projection_seconds"])
                    write_json(RUN / "collection_ready.json", {**estimate, "config_hash": frozen["config_hash"],
                        "ready_at": utc_now(), "effective_resource_limit_seconds": authorized_seconds,
                        "successful_probes_revalidated_without_new_model_calls": True})
                if not (RUN / "collection_ready.json").exists():
                    if recovering and (RUN / "cpu_decode_probes.json").exists():
                        (private / f"cpu_decode_probes_before_recovery_{time.time_ns()}.json").write_bytes((RUN / "cpu_decode_probes.json").read_bytes())
                    coordinate("smoke", config, RUN, ROOT, ledger, CONFIG)
                assert read_json(RUN / "execution_order.json") == read_json(ROOT / "results/e000_20261027_collection_v2/execution_order.json")
                print(f"GPU preflight passed; measured collection forecast {read_json(RUN / 'collection_ready.json')['projection_seconds']:.1f}s; remaining {authorized_seconds - ledger.consumed_seconds():.1f}s", flush=True)
                command = "resume" if list((RUN / "records").glob("*.json")) else "run"
                coordinate(command, config, RUN, ROOT, ledger, CONFIG)
                checked = read_json(RUN / "verification.json")
                print(f"Collection finished: {checked['verified_records']} records; all gates passed = {checked['all_operational_gates_passed']}", flush=True)
            except BaseException as exc:
                status(RUN, getattr(exc, "status", "PARTIAL"), f"{type(exc).__name__}: {exc}")
                write_json(RUN / f"private/coordinator_failure_{time.time_ns()}.json", {"at": utc_now(), "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

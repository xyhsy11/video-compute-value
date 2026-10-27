"""Cumulative active time and project-local storage accounting, including failures."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .common import Blocked, command_output, read_json, utc_now, write_json

GIB = 1024 ** 3
PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "av", "numpy", "Pillow",
            "PyYAML", "psutil", "huggingface-hub", "pyarrow", "tokenizers", "safetensors", "nvidia-ml-py")


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def tree_bytes(root):
    total = 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not Path(parent, d).is_symlink()]
        for name in files:
            path = Path(parent, name)
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


class Ledger:
    def __init__(self, path, root, config):
        self.path, self.root, self.config = Path(path), Path(root).resolve(), config
        if self.path.exists():
            self.value = read_json(self.path)
        else:
            self.value = {"created_at": utc_now(), "root": str(self.root), "active_seconds": 0,
                          "download_bytes": 0, "video_transfer_bytes": 0,
                          "storage_baseline_bytes": 0, "storage_peak_bytes": 0, "sessions": [],
                          "storage_scope": "Entire new project, including environment, caches, temp and Git"}
        self.started = None
        self._last_storage_check = 0.0

    def consumed_seconds(self):
        extra = time.monotonic() - self.started if self.started is not None else 0
        return self.value["active_seconds"] + extra

    def save(self):
        if time.monotonic() - self._last_storage_check >= 2 or "storage_current_bytes" not in self.value:
            self.value["storage_current_bytes"] = tree_bytes(self.root)
            self._last_storage_check = time.monotonic()
        self.value["storage_peak_bytes"] = max(self.value["storage_peak_bytes"], self.value["storage_current_bytes"])
        write_json(self.path, self.value)

    def check(self, projected_seconds=0, projected_bytes=0):
        self.save()
        if self.consumed_seconds() + projected_seconds > self.config["cumulative_execution_limit_hours"] * 3600:
            raise Blocked("BLOCKED_RESOURCE", "Cumulative active execution time budget exceeded")
        if self.value["storage_current_bytes"] + projected_bytes > self.config["additional_disk_limit_gib"] * GIB:
            raise Blocked("BLOCKED_RESOURCE", "Additional project storage budget exceeded")
        if shutil.disk_usage(self.root).free < projected_bytes:
            raise Blocked("BLOCKED_RESOURCE", "Insufficient free disk for projected operation")

    def charge_transfer(self, size, video=False):
        self.value["download_bytes"] += size
        if video:
            self.value["video_transfer_bytes"] += size
        self.save()

    def allow_video_transfer(self, size):
        if self.value["video_transfer_bytes"] + size > self.config["video_acquisition_limit_gib"] * GIB:
            raise Blocked("BLOCKED_RESOURCE", "1 GiB cumulative video transfer budget exceeded")

    @contextmanager
    def track(self, name):
        # Unclosed sessions are conservatively charged wall time, capped at the entire budget.
        for previous in self.value["sessions"]:
            if previous.get("finished_at") is None:
                from datetime import datetime
                elapsed = max(0, (datetime.now().astimezone() - datetime.fromisoformat(previous["started_at"])).total_seconds())
                charged = min(elapsed, self.config["cumulative_execution_limit_hours"] * 3600)
                self.value["active_seconds"] += charged
                previous.update(finished_at=utc_now(), active_seconds=charged, outcome="interrupted-conservative-charge")
        self.started = time.monotonic()
        record = {"name": name, "started_at": utc_now(), "finished_at": None}
        self.value["sessions"].append(record)
        self.save()
        try:
            self.check()
            yield self
            record["outcome"] = "completed"
        except BaseException as exc:
            record["outcome"] = type(exc).__name__
            raise
        finally:
            elapsed = time.monotonic() - self.started
            self.started = None
            self.value["active_seconds"] += elapsed
            record.update(finished_at=utc_now(), active_seconds=elapsed)
            self.save()


def environment(root):
    import psutil
    packages = package_versions()
    memory, disk = psutil.virtual_memory(), shutil.disk_usage(root)
    battery = psutil.sensors_battery()
    record = {
        "observed_at": utc_now(), "os": platform.platform(), "python": platform.python_version(),
        "python_executable": sys.executable, "packages": packages,
        "ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
        "disk_free_bytes": disk.free, "cpu_count": psutil.cpu_count(),
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "power": None if battery is None else {"plugged_in": battery.power_plugged, "battery_percent": battery.percent},
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu,power.draw", "--format=csv"]),
        "cache_roots": {"project": str(Path(root, ".cache").resolve()),
                        "existing_huggingface": str(Path.home() / ".cache" / "huggingface")},
        "reusable_caches": [], "cuda": None,
    }
    for path in (Path(root, ".cache", "huggingface"), Path.home() / ".cache" / "huggingface"):
        if path.exists():
            record["reusable_caches"].append({"path": str(path), "bytes": tree_bytes(path)})
    if packages["torch"]:
        import torch
        record["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda,
                          "torch_threads": torch.get_num_threads()}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            record["cuda"].update(name=torch.cuda.get_device_name(0), free_bytes=free, total_bytes=total,
                                  bf16_supported=torch.cuda.is_bf16_supported())
    return record


def environment_fingerprint(record):
    return {"os": record["os"], "python": record["python"], "packages": record["packages"],
            "gpu": None if not record["cuda"] else {k: record["cuda"].get(k) for k in
                                                     ("available", "runtime", "name", "total_bytes")},
            "thread_environment": record["thread_environment"]}

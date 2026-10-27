"""Primary timer boundary and optional sampled (not exact peak) process/device memory."""

from __future__ import annotations

import os
import threading
import time


class MemorySampler:
    def __init__(self, interval=0.05):
        import psutil
        self.interval, self.process = interval, psutil.Process()
        self.stop_event = threading.Event()
        self.ram = self.device = self.gpu_process = None
        self.nvml = self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml, self.handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None

    def sample(self):
        self.ram = max(self.ram or 0, self.process.memory_info().rss)
        if self.nvml:
            try:
                self.device = max(self.device or 0, self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
                processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                values = [p.usedGpuMemory for p in processes if p.pid == os.getpid() and
                          p.usedGpuMemory is not None and p.usedGpuMemory < 2 ** 60]
                if values:
                    self.gpu_process = max(self.gpu_process or 0, max(values))
            except Exception:
                pass

    def loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        if self.nvml:
            self.nvml.nvmlShutdown()
        return {"sampling_interval_seconds": self.interval, "sampled_ram_peak_bytes": self.ram,
                "sampled_device_gpu_peak_bytes": self.device, "sampled_process_gpu_peak_bytes": self.gpu_process}


def timed_request(adapter, question, video, budget, *, profile=False, sampler=None, clock=time.perf_counter_ns):
    """Returns CPU input tensors for diagnostic hashing AFTER the primary timer stops."""
    cuda = adapter.torch.cuda
    cuda.synchronize()
    baseline_allocated, baseline_reserved = cuda.memory_allocated(), cuda.memory_reserved()
    cuda.reset_peak_memory_stats()
    if sampler:
        sampler.start()
    cpu_inputs = gpu_inputs = None
    try:
        started = clock()  # Before prepare() opens the compressed local video.
        cpu_inputs, details = adapter.prepare(question, video, budget)
        prepared = clock() if profile else None
        gpu_inputs = adapter.transfer(cpu_inputs)
        if profile:
            cuda.synchronize()
            transferred = clock()
        output = adapter.generate(gpu_inputs)  # Includes output transfer, text decode, and parsing.
        cuda.synchronize()
        stopped = clock()
        record = {**details, **output, "latency_seconds": (stopped - started) / 1e9,
                  "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
                  "peak_allocated_bytes": cuda.max_memory_allocated(),
                  "peak_reserved_bytes": cuda.max_memory_reserved()}
        record["incremental_peak_allocated_bytes"] = record["peak_allocated_bytes"] - baseline_allocated
        record["incremental_peak_reserved_bytes"] = record["peak_reserved_bytes"] - baseline_reserved
        if profile:
            record["components_seconds"] = {"cpu_prepare": (prepared - started) / 1e9,
                                             "transfer_synchronized": (transferred - prepared) / 1e9,
                                             "model_and_output_synchronized": (stopped - transferred) / 1e9}
        return record, cpu_inputs
    finally:
        del gpu_inputs
        if sampler:
            sampled = sampler.stop()
            if "record" in locals():
                record.update(sampled)

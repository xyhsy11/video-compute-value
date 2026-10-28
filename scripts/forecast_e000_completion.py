"""Transparent full-collection forecasts; estimates are never trial results."""
from __future__ import annotations


def forecast_collection(cpu_seconds, remainder_seconds, startup_seconds, text_seconds,
                        cpu_probe_seconds, remaining_seconds, verification_seconds=180,
                        retry_reserve_seconds=600, setup_seconds=90, safety_factor=1.5):
    budgets = (8, 16, 32, 48, 64)
    if set(cpu_seconds) != set(budgets) or set(remainder_seconds) != set(budgets):
        raise ValueError("Forecast requires all five budgets")
    values = [*cpu_seconds.values(), *remainder_seconds.values(), startup_seconds, text_seconds,
              cpu_probe_seconds, remaining_seconds, verification_seconds, retry_reserve_seconds, setup_seconds]
    if min(values) < 0 or safety_factor < 1:
        raise ValueError("Negative costs or unsafe margin")
    # Per video budget: 72 primary + 6 collection warmups + 5 fit/integrity/profile calls.
    video_calls = {b: 83 for b in budgets}
    video = sum(video_calls[b] * (cpu_seconds[b] + remainder_seconds[b]) for b in budgets)
    text = 26 * text_seconds  # 24 primary and two separate-worker warmups.
    startup = 21 * startup_seconds  # Five probe workers and 16 collection workers.
    raw = video + text + startup + cpu_probe_seconds + verification_seconds + setup_seconds + retry_reserve_seconds
    guarded = raw * safety_factor
    return {"planned_primary_video": 360, "planned_primary_text": 24, "warmups": 42,
        "fit_integrity_and_profile_calls": 15, "planned_model_invocations_before_retries": 441,
        "fresh_model_workers": 21, "video_call_seconds": video, "text_call_seconds": text,
        "worker_startup_seconds": startup, "cpu_probe_seconds": cpu_probe_seconds,
        "verification_seconds": verification_seconds, "setup_seconds": setup_seconds,
        "bounded_retry_reserve_seconds": retry_reserve_seconds, "raw_projected_seconds": raw,
        "safety_factor": safety_factor, "projected_seconds": guarded,
        "remaining_seconds": remaining_seconds, "fits_remaining_budget": guarded <= remaining_seconds,
        "additional_seconds_required": max(0, guarded - remaining_seconds),
        "forecast_only": True, "full_gpu_policy_verified": False}

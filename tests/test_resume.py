"""Resume must reject changed provenance, even when no trial has yet completed."""

import pytest
import yaml

from video_compute_value.common import Blocked, digest, write_json
from video_compute_value.resources import environment_fingerprint
from video_compute_value.runner import source_hash, validate_frozen


@pytest.mark.parametrize("change", ["configuration", "source", "environment", "driver", "cohort", "order", "weights"])
def test_resume_checks_all_frozen_fingerprints(tmp_path, monkeypatch, change):
    import video_compute_value.runner as runner
    run = tmp_path / "results" / "test"
    run.mkdir(parents=True)
    source = tmp_path / "src" / "video_compute_value" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"synthetic fixture, never a real model")
    from video_compute_value.common import file_hash
    model = {"model_path": str(weights), "files": {"model.safetensors": file_hash(weights / "model.safetensors")}}
    cohort, order, config = {"videos": []}, [], {"seed": 20261027}
    env = {"os": "test", "python": "3.12", "packages": {"torch": "2.8.0"}, "cuda": None, "thread_environment": {}}
    frozen = {"config_input_hash": digest(config), "source_hash": source_hash(tmp_path),
              "cohort_hash": digest(cohort), "order_hash": digest(order),
              "model_manifest_hash": digest(model), "environment_fingerprint": environment_fingerprint(env)}
    frozen["config_hash"] = digest(frozen)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(frozen), encoding="utf-8")
    write_json(run / "cohort_manifest.json", cohort)
    write_json(run / "execution_order.json", order)
    write_json(run / "model_manifest.json", model)
    monkeypatch.setattr(runner, "environment", lambda root: env)
    monkeypatch.setattr(runner, "validate_cohort", lambda cohort, config: None)
    assert validate_frozen(config, run, tmp_path) == frozen
    if change == "configuration":
        config["seed"] += 1
    elif change == "source":
        source.write_text("changed", encoding="utf-8")
    elif change == "environment":
        env["packages"]["torch"] = "different"
    elif change == "driver":
        env["nvidia_driver_version"] = "changed"
    elif change == "cohort":
        write_json(run / "cohort_manifest.json", {"videos": ["changed"]})
    elif change == "order":
        write_json(run / "execution_order.json", ["changed"])
    else:
        (weights / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(Blocked, match="changed"):
        validate_frozen(config, run, tmp_path)

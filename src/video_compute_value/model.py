"""Pinned native Qwen3-VL adapter. No labels, second sampling, offload or quantization."""

from __future__ import annotations

import hashlib
import inspect
import math
import re
import time
from fractions import Fraction

import numpy as np

from .common import Blocked, digest, file_hash
from .protocol import make_prompt, parse_answer
from .video import decode_selected


def tensor_hashes(inputs):
    result = {}
    for key, tensor in sorted(inputs.items()):
        value = tensor.detach().cpu().contiguous()
        # uint8 view also supports BF16 without a NumPy BF16 dependency.
        data = value.view(__import__("torch").uint8).numpy().tobytes()
        result[key] = digest({"shape": list(value.shape), "dtype": str(value.dtype),
                              "bytes_sha256": hashlib.sha256(data).hexdigest()})
    return result


def validate_grid(grid, budget, height, width, visual_tokens):
    expected = [budget // 2, height // 16, width // 16]
    if budget % 2 or grid != [expected] or visual_tokens != math.prod(expected) // 4:
        raise Blocked("INPUT_INTEGRITY_FAILED", f"Actual video grid or visual tokens disagree: {grid}")


def adapter_source_hashes():
    from transformers.models.qwen3_vl import processing_qwen3_vl, video_processing_qwen3_vl
    return {module.__name__: file_hash(inspect.getfile(module)) for module in
            (processing_qwen3_vl, video_processing_qwen3_vl)}


class QwenAdapter:
    def __init__(self, frozen):
        import torch
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
        self.torch, self.frozen = torch, frozen
        self.model_call_count = 0
        torch.set_num_threads(frozen["cpu_threads"])
        torch.manual_seed(frozen["seed"])
        np.random.seed(frozen["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.processor = AutoProcessor.from_pretrained(frozen["model_path"], local_files_only=True,
                                                        trust_remote_code=False, use_fast=True)
        if adapter_source_hashes() != frozen["processor_source_hashes"]:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Installed processor source changed")
        vp = self.processor.video_processor
        if (vp.patch_size, vp.merge_size, vp.temporal_patch_size) != (16, 2, 2):
            raise Blocked("INPUT_INTEGRITY_FAILED", "Unsupported Qwen patch/merge settings")
        if list(vp.image_mean) != [0.5] * 3 or list(vp.image_std) != [0.5] * 3:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Model normalization changed")
        self.generation = GenerationConfig(**frozen["generation"])
        started = time.perf_counter()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            frozen["model_path"], dtype=getattr(torch, frozen["dtype"]),
            attn_implementation=frozen["attention_backend"], local_files_only=True, trust_remote_code=False,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True)
        self.model.eval()
        self.model.requires_grad_(False)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.baseline = {"allocated_bytes": torch.cuda.memory_allocated(),
                         "reserved_bytes": torch.cuda.memory_reserved()}
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise Blocked("INPUT_INTEGRITY_FAILED", "Model is not entirely on CUDA")

    def prepare(self, question, video=None, budget=0):
        from transformers.video_utils import VideoMetadata
        prompt = make_prompt(question)
        content = []
        kwargs, details = {}, {"frames_decoded": 0, "frames_selected": 0, "frames_processor": 0,
                               "frames_represented": 0, "visual_tokens": 0, "video_grid_thw": [],
                               "temporal_padding": 0, "model_timestamps": [], "source_indices": [],
                               "actual_timestamps": []}
        if budget:
            frames, decoded = decode_selected(video, budget)
            details.update(decoded)
            content.append({"type": "video"})
            metadata = VideoMetadata(total_num_frames=video["frame_count"], fps=float(Fraction(video["fps"])),
                                     duration=float(Fraction(video["duration"])),
                                     frames_indices=details["source_indices"])
            tensor = self.torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).contiguous()
            kwargs = {"videos": [tensor], "videos_kwargs": {
                "video_metadata": [metadata], "do_sample_frames": False, "do_resize": False,
                "do_convert_rgb": False, "do_rescale": True, "rescale_factor": 1 / 255.0,
                "do_normalize": True, "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
            }}
        content.append({"type": "text", "text": prompt})
        chat = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], return_tensors="pt", padding=False, truncation=False, **kwargs)
        inputs.pop("token_type_ids", None)
        if budget:
            grid = inputs["video_grid_thw"].tolist()
            token_count = int((inputs["input_ids"] == self.processor.video_token_id).sum())
            validate_grid(grid, budget, video["target_height"], video["target_width"], token_count)
            indices = details["source_indices"]
            expected_times = [float((Fraction(video["timestamps"][indices[i]]) +
                                     Fraction(video["timestamps"][indices[i + 1]])) / 2)
                              for i in range(0, budget, 2)]
            represented = self.processor._calculate_timestamps(indices.copy(), float(Fraction(video["fps"])), 2)
            if not np.allclose(represented, expected_times, rtol=0, atol=1e-8):
                raise Blocked("INPUT_INTEGRITY_FAILED", "Native timestamp grouping disagrees with PTS")
            token_text = self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
            actual_tags = re.findall(r"<([0-9]+\.[0-9]) seconds>", token_text)
            if actual_tags != [f"{t:.1f}" for t in expected_times]:
                raise Blocked("INPUT_INTEGRITY_FAILED", "Actual timestamp tokens disagree")
            details.update(frames_processor=len(frames), frames_represented=grid[0][0] * 2,
                           video_grid_thw=grid, visual_tokens=token_count,
                           model_timestamps=expected_times, model_timestamp_text=actual_tags,
                           spatial_height=video["target_height"], spatial_width=video["target_width"])
        details["tensor_shapes"] = {key: list(value.shape) for key, value in inputs.items()}
        details["input_tokens"] = int(inputs["input_ids"].shape[-1])
        if details["input_tokens"] + self.frozen["generation"]["max_new_tokens"] > self.model.config.text_config.max_position_embeddings:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Input exceeds context; truncation is prohibited")
        return inputs, details

    def transfer(self, inputs):
        return {key: value.to("cuda:0") for key, value in inputs.items()}

    def generate(self, inputs):
        with self.torch.inference_mode():
            self.model_call_count += 1
            output = self.model.generate(**inputs, generation_config=self.generation)
        new_ids = output[0, inputs["input_ids"].shape[-1]:].detach().cpu().tolist()
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = parse_answer(text)
        return {"generated_token_ids": new_ids, "decoded_output": text, "parsed_answer": answer,
                "valid_answer": answer is not None, "output_tokens": len(new_ids)}

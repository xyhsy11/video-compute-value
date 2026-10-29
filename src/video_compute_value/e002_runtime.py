"""E2-only same-call scores and real base-first/direct execution boundaries.

Production paths require a successful compatibility receipt and full-stage admission.
The CPU tests use fake adapters; they are not GPU evidence.
"""
from __future__ import annotations

from fractions import Fraction
import time

import numpy as np

from .common import Blocked, digest
from .e001_input import inference_video
from .e002_features import (AllowedObservation, confidence_summary, projected_query, query_features,
                            visual_features)
from .e002_input import menu
from .model import QwenAdapter
from .protocol import parse_answer


class E2Adapter(QwenAdapter):
    capture_scores = False

    def generate(self, inputs):
        if not self.capture_scores:
            return super().generate(inputs)
        if self._prepared_budget != 8:
            raise Blocked('BLOCKED_FEATURE_INTEGRITY', 'C is only captured on the paid base request')
        with self.torch.inference_mode():
            self.model_call_count += 1
            output = self.model.generate(**inputs, generation_config=self.generation,
                **self.generation_policy['call_overrides'], return_dict_in_generate=True, output_scores=True)
        new_ids = output.sequences[0, inputs['input_ids'].shape[-1]:].detach().cpu().tolist()
        if not output.scores:
            raise Blocked('BLOCKED_FEATURE_INTEGRITY', 'No same-call first-token scores')
        c = confidence_summary(output.scores[0][0].detach().double().cpu().numpy())
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        answer = parse_answer(text)
        return {'generated_token_ids': new_ids, 'decoded_output': text, 'parsed_answer': answer,
                'valid_answer': answer is not None, 'output_tokens': len(new_ids), 'C': c,
                'generation_policy_sha256': digest(self.generation_policy),
                'cuda_cache_trimmed': False, 'cuda_cache_trim_host_seconds': 0.}


def observation(question, video, details):
    return AllowedObservation(base_rgb=details['_e1_pixels_for_hashing'],
        timestamps=tuple(details['actual_timestamps']), duration=video['duration'],
        frame_count=video['frame_count'], aspect_ratio=video['width']/video['height'],
        content_rectangle=(video['pad_top'], video['pad_left'], video['content_height'], video['content_width']),
        question_text=question.question)


def policy_features(allowed, family):
    if family == 'V':
        return visual_features(allowed)
    if family == 'Q':
        return query_features(allowed.question_text)
    if family == 'VQ':
        v, q = visual_features(allowed), query_features(allowed.question_text)
        return np.r_[v, q, np.outer(v[:16], projected_query(q)).ravel()]
    raise ValueError('Unfrozen observation family')


def execute_pipeline(adapter, question, video, *, direct=None, family=None, decide=None, seal=None,
                     clock=time.perf_counter):
    """decide receives only features/C. seal MUST complete before any candidate decoding.

    Callers pass a protocol.InferenceQuestion and protocol.video_input whitelist.
    Hashing/serialization/scoring belongs outside this primary deployment timer.
    """
    if direct is None and (family not in ('V', 'VQ', 'C') or decide is None or seal is None):
        raise ValueError('Adaptive execution requires a strict family, decision function, and decision seal')
    cuda = adapter.torch.cuda
    cuda.synchronize()
    started, calls_before = clock(), adapter.model_call_count
    components = {'base_or_direct': 0., 'observation': 0., 'selection': 0., 'followup': 0.}

    def request(condition, capture=False):
        action = menu(video)[condition]
        adapter.capture_scores = capture
        inputs, details = adapter.prepare(question, inference_video(video, action), len(action['indices']))
        result = adapter.generate(adapter.transfer(inputs))
        cuda.synchronize()
        return result, details

    start = clock()
    base, details = request(direct or 'base_plain', capture=family == 'C' and direct is None)
    components['base_or_direct'] = clock()-start
    selected, result = direct or 'stop', base
    if direct is None:
        start = clock()
        features = np.asarray(base['C']) if family == 'C' else policy_features(observation(question, video, details), family)
        components['observation'] = clock()-start
        start = clock()
        selected = decide(features)
        if selected not in ('stop', 'quarter1', 'quarter2', 'quarter3', 'quarter4'):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Invalid practical policy action')
        components['selection'] = clock()-start
        # This sealing hook is an in-memory append; disk serialization occurs after the timer.
        seal(selected)
        if selected != 'stop':
            start = clock()
            result, _ = request(selected)
            components['followup'] = clock()-start
    cuda.synchronize()
    seconds = clock()-started
    expected = 1+int(direct is None and selected != 'stop')
    if adapter.model_call_count-calls_before != expected:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Actual deployed model-call count differs')
    return {'output': result, 'action': selected, 'total_seconds': seconds,
            'components_seconds': components, 'actual_model_calls': expected,
            'original_duration_seconds': float(Fraction(video['duration']))}

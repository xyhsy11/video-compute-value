"""E1's frozen acquisitions at the existing native video decoder boundary."""
from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
from PIL import Image

from .common import Blocked
from .video import letterbox, uniform_indices

CONDITIONS = ('base', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition')


def addition(timestamps, duration, base, quarter=None):
    times, duration = [Fraction(t) for t in timestamps], Fraction(duration)
    lo, hi = (Fraction(0), duration) if quarter is None else (duration*quarter/4, duration*(quarter+1)/4)
    eligible = [i for i, t in enumerate(times) if lo <= t and (t < hi or hi == duration and t <= hi)]
    excluded, chosen, displacement = set(base), [], []
    for k in range(8):
        target = lo + (hi-lo)*Fraction(2*k+1, 16)
        available = [i for i in eligible if i not in excluded]
        if not available:
            raise Blocked('BLOCKED_INPUT', 'Quarter cannot supply eight distinct additional frames')
        nearest = min(eligible, key=lambda i: (abs(times[i]-target), times[i]))
        selected = min(available, key=lambda i: (abs(times[i]-target), times[i]))
        chosen.append(selected)
        excluded.add(selected)
        displacement.append({'target': str(target), 'nearest_unconstrained_index': nearest,
                             'selected_index': selected, 'selected_pts': str(times[selected]),
                             'exclusion_displacement_seconds': str(times[selected]-times[nearest])})
    return chosen, displacement


def actions(video):
    base = uniform_indices(video['timestamps'], video['duration'], 8)
    if base != video['selections']['8']['indices']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'E0 base selection changed')
    result = {'base': {'indices': base, 'supplemental_indices': [], 'displacements': []}}
    for name in CONDITIONS[1:]:
        extra, displaced = addition(video['timestamps'], video['duration'], base,
                                    None if name == 'uniform_addition' else int(name[-1])-1)
        merged = sorted(base + extra)
        if len(merged) != 16 or len(set(merged)) != 16 or not set(base) <= set(merged):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Retained evidence or distinct frame count failed')
        result[name] = {'indices': merged, 'supplemental_indices': extra, 'displacements': displaced}
    for value in result.values():
        value['timestamps'] = [video['timestamps'][i] for i in value['indices']]
    return result


def decode_e1(video, budget, *, decoder_threads=4, decoder_thread_type='FRAME'):
    """Only the selector differs; RGB/letterbox and native processor stay inherited."""
    import av
    if (decoder_threads, decoder_thread_type) != (4, 'FRAME'):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'E1 inherits the corrected FRAME/4 decoder')
    selected = video['_e1_selection']
    indices = selected['indices']
    if budget != len(indices) or len(set(indices)) != budget:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Frozen E1 selection has wrong size')
    wanted, frames, count = set(indices), [], 0
    from pathlib import Path
    if Path(video['path']).stat().st_size != video['file_size']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Source video size changed')
    with av.open(video['path']) as container:
        stream = container.streams.video[0]
        stream.thread_type = decoder_thread_type
        stream.codec_context.thread_count = decoder_threads
        for index, frame in enumerate(container.decode(stream)):
            count += 1
            if (frame.pts is None or index >= len(video['timestamps']) or
                Fraction(frame.pts)*Fraction(stream.time_base) != Fraction(video['timestamps'][index]) or
                (frame.height, frame.width) != (video['height'], video['width'])):
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Decoded source metadata changed')
            if index in wanted:
                rgb = letterbox(Image.fromarray(frame.to_ndarray(format='rgb24')), video)
                frames.append(rgb)
    if count != video['frame_count'] or len(frames) != budget:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Actual source or selected frame count changed')
    pixels = np.stack(frames)
    return pixels, {'frames_decoded': count, 'frames_selected': budget,
        'source_indices': indices, 'actual_timestamps': selected['timestamps'],
        'decoder_threads': decoder_threads, 'decoder_thread_type': decoder_thread_type,
        '_e1_pixels_for_hashing': pixels, '_e1_base_indices': video['selections']['8']['indices']}


def finish_hashes(details):
    """Consume transient per-request pixels after timing; never retain across requests."""
    pixels = details.pop('_e1_pixels_for_hashing')
    base = details.pop('_e1_base_indices')
    hashes = {str(i): hashlib.sha256(frame.tobytes()).hexdigest()
              for i, frame in zip(details['source_indices'], pixels)}
    details['processed_rgb_hashes'] = hashes
    details['retained_base_rgb_hashes'] = {str(i): hashes[str(i)] for i in base}


def install_decoder():
    # Process-local substitution at the single decoder boundary. QwenAdapter.prepare
    # still constructs one native video with original VideoMetadata, never an image list.
    from . import model
    model.decode_selected = decode_e1


def inference_video(video, action):
    from .protocol import video_input
    return {**video_input(video), '_e1_selection': action}

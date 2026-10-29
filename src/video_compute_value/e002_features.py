"""E2 v1 allowed observations. No labels, options, identities, or candidate access."""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import re
import unicodedata

import numpy as np
from PIL import Image

SEED = 2026102902
SCHEMA = {
    'version': 'e002-observations-v1', 'V': 19, 'Q': 66, 'VQ': 341,
    'pixels': 'exact eight retained uint8 RGB canvases; remove frozen letterbox rectangle',
    'gray': 'Pillow RGB.convert(L), resize 32x32 BILINEAR, float64 / 255',
    'quarters': 'floor(4*original_PTS/duration); exactly two base observations per quarter',
    'visual': ['mean spatial population std', 'mean of horizontal/vertical mean abs differences',
               'between-image mean abs difference', 'fraction abs difference > 0.10'],
    'global': ['log1p duration seconds', 'log1p frame count', 'original width/height'],
    'question_recipe': 'NFKC then casefold; Unicode regex [^\\W_]+; SHA256 UTF8 token; '
                'bucket=int.from_bytes(digest[:8],big)%64; sign=+1 if byte8 even else -1; L2',
    'question_sizes': 'log1p unigram count and normalized character count',
    'projection': 'each Q[0:64] hashes seed:projection:index to one of 16 buckets '
                  'using the same bucket/sign rule; no fitted projection',
    'interactions': 'quarter-major visual[0:16] outer projected_query[0:16], flattened C order',
    'C': ['first-token top1 probability', 'top1 minus top2', 'entropy/log(vocabulary size)'],
    'C_limitation': 'generation-score summaries, not calibrated answer correctness',
    'response_shape': ['valid parsed base answer', 'generated token count'],
    'repeat_atol': 1e-12, 'repeat_rtol': 0,
}


@dataclass(frozen=True, slots=True)
class AllowedObservation:
    base_rgb: np.ndarray
    timestamps: tuple[str, ...]
    duration: str
    frame_count: int
    aspect_ratio: float
    content_rectangle: tuple[int, int, int, int]  # top, left, height, width
    question_text: str

    @classmethod
    def from_mapping(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('Policy input must contain exactly the allowed observation fields')
        return cls(**value)


def _bucket(text, size):
    raw = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(raw[:8], 'big') % size, 1 if raw[8] % 2 == 0 else -1


def query_features(text):
    if type(text) is not str:
        raise ValueError('Only original question text is accepted')
    normalized = unicodedata.normalize('NFKC', text).casefold()
    tokens = re.findall(r'[^\W_]+', normalized, flags=re.UNICODE)
    vector = np.zeros(64, dtype=np.float64)
    for token in tokens:
        bucket, sign = _bucket(token, 64)
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return np.r_[vector, np.log1p(len(tokens)), np.log1p(len(normalized))]


def projected_query(query):
    projection = np.zeros(16, dtype=np.float64)
    for index, value in enumerate(query[:64]):
        bucket, sign = _bucket(f'{SEED}:projection:{index}', 16)
        projection[bucket] += sign * value
    return projection


def visual_features(observation):
    if type(observation) is not AllowedObservation:
        raise TypeError('A strict AllowedObservation is required')
    rgb = observation.base_rgb
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (8, 192, 320, 3):
        raise ValueError('Expected exact eight 192x320 uint8 RGB retained base frames')
    duration = Fraction(observation.duration)
    times = [Fraction(t) for t in observation.timestamps]
    if (duration <= 0 or len(times) != 8 or any(t < 0 or t >= duration for t in times)
            or any(a >= b for a, b in zip(times, times[1:])) or observation.frame_count < 16
            or not np.isfinite(observation.aspect_ratio) or observation.aspect_ratio <= 0):
        raise ValueError('Invalid base temporal or source metadata')
    quarters = [int(4*t/duration) for t in times]
    if [quarters.count(q) for q in range(4)] != [2]*4:
        raise ValueError('BLOCKED_INPUT: expected exactly two base observations per quarter')
    top, left, height, width = observation.content_rectangle
    if (min(top, left) < 0 or min(height, width) < 1 or top+height > 192 or left+width > 320):
        raise ValueError('Frozen letterbox crop is outside canvas')
    gray = [np.asarray(Image.fromarray(frame[top:top+height, left:left+width]).convert('L')
                       .resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float64)/255
            for frame in rgb]
    values = []
    for quarter in range(4):
        first, second = [gray[i] for i, q in enumerate(quarters) if q == quarter]
        delta = np.abs(second-first)
        gradient = [0.5*(np.abs(np.diff(a, axis=0)).mean()+np.abs(np.diff(a, axis=1)).mean())
                    for a in (first, second)]
        values.extend([(first.std()+second.std())/2, sum(gradient)/2,
                       delta.mean(), (delta > .10).mean()])
    return np.r_[values, np.log1p(float(duration)), np.log1p(observation.frame_count),
                 observation.aspect_ratio].astype(np.float64)


def extract(observation):
    visual = visual_features(observation)
    query = query_features(observation.question_text)
    return {'V': visual, 'Q': query,
            'VQ': np.r_[visual, query, np.outer(visual[:16], projected_query(query)).ravel()]}


def confidence_summary(scores):
    """Pure reference reduction of the SAME generation call's first-token scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or len(scores) < 2 or np.isnan(scores).any()
            or np.isposinf(scores).any() or not np.isfinite(scores).any()):
        raise ValueError('Unavailable/invalid first-token full-vocabulary distribution')
    probabilities = np.exp(scores-np.max(scores))
    probabilities /= probabilities.sum()
    top = np.sort(probabilities)[-2:]
    positive = probabilities[probabilities > 0]
    return [float(top[-1]), float(top[-1]-top[-2]),
            float(-(positive*np.log(positive)).sum()/np.log(len(scores)))]

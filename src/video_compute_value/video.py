"""PTS-based sampling; sequential decode keeps at most B resized RGB frames."""

from __future__ import annotations

import bisect
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .common import Blocked, digest, file_hash


def uniform_indices(timestamps, duration, budget):
    times = [Fraction(t) for t in timestamps]
    duration = Fraction(duration)
    if budget < 1 or duration <= 0 or not times or times[0] < 0:
        raise ValueError("Invalid timeline or budget")
    if any(b <= a for a, b in zip(times, times[1:])) or times[-1] >= duration:
        raise ValueError("PTS must be strictly increasing and within full duration")
    selected = []
    for i in range(budget):
        target = duration * Fraction(2 * i + 1, 2 * budget)
        right = bisect.bisect_left(times, target)
        choices = [j for j in (right - 1, right) if 0 <= j < len(times)]
        selected.append(min(choices, key=lambda j: (abs(times[j] - target), times[j])))
    if len(set(selected)) != budget:
        raise ValueError("Budget cannot produce distinct source frames")
    return selected


def spatial_size(height, width, max_pixels=65536, factor=32):
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("Unsupported aspect ratio")
    scale = min(1.0, math.sqrt(max_pixels / (height * width)))
    h = max(factor, math.floor(height * scale / factor) * factor)
    w = max(factor, math.floor(width * scale / factor) * factor)
    while h * w > max_pixels:
        if h >= w:
            h -= factor
        else:
            w -= factor
    if min(h, w) < factor:
        raise ValueError("Aspect ratio cannot fit spatial grid")
    return h, w


def spatial_layout(height, width, max_pixels=65536):
    h, w = spatial_size(height, width, max_pixels)
    scale = min(h / height, w / width)
    content_h, content_w = max(1, round(height * scale)), max(1, round(width * scale))
    return {"target_height": h, "target_width": w, "content_height": content_h, "content_width": content_w,
            "pad_top": (h - content_h) // 2, "pad_left": (w - content_w) // 2,
            "spatial_preprocessing": "aspect-preserving-bicubic-centered-black-letterbox-v1"}


def letterbox(image, layout):
    resized = image.resize((layout["content_width"], layout["content_height"]), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (layout["target_width"], layout["target_height"]), (0, 0, 0))
    canvas.paste(resized, (layout["pad_left"], layout["pad_top"]))
    return np.asarray(canvas)


def inspect_video(path, budgets, max_pixels):
    import av
    path = Path(path).resolve()
    times, durations = [], []
    dimensions = set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        timebase = Fraction(stream.time_base)
        colors = {"colorspace": str(stream.codec_context.colorspace),
                  "color_range": str(stream.codec_context.color_range), "conversion": "PyAV rgb24"}
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Missing presentation timestamp")
            times.append(Fraction(frame.pts) * timebase)
            durations.append(Fraction(frame.duration or 0) * timebase)
            dimensions.add((frame.height, frame.width))
        if len(times) < max(budgets) or len(dimensions) != 1:
            raise ValueError("Too few frames or changing dimensions")
        if times[0] != 0:
            raise ValueError("Nonzero starting PTS unsupported by this CFR adapter")
        increments = [b - a for a, b in zip(times, times[1:])]
        if min(increments) <= 0 or len(set(increments)) != 1:
            raise ValueError("VFR/nonmonotonic PTS excluded: native pinned adapter uses source-index/fps")
        step = increments[0]
        if durations[-1] > 0 and durations[-1] != step:
            raise ValueError("Variable terminal frame duration")
        duration = times[-1] + step
        if stream.duration is not None and abs(Fraction(stream.duration) * timebase - duration) > step:
            raise ValueError("Stream duration disagrees with decoded full timeline")
        fps = 1 / step
    h, w = dimensions.pop()
    layout = spatial_layout(h, w, max_pixels)
    selections = {}
    for budget in budgets:
        indices = uniform_indices(times, duration, budget)
        selections[str(budget)] = {"indices": indices, "timestamps": [str(times[i]) for i in indices]}
    return {"path": str(path), "sha256": file_hash(path), "file_size": path.stat().st_size,
            "duration": str(duration), "fps": str(fps), "height": h, "width": w,
            **layout, "frame_count": len(times),
            "timestamps": [str(t) for t in times], "timestamps_hash": digest([str(t) for t in times]),
            "time_base": str(timebase), "colors": colors, "selections": selections,
            "decoder": {"name": "PyAV", "version": av.__version__,
                        "ffmpeg_libraries": {k: list(v) for k, v in av.library_versions.items()},
                        "strategy": "full sequential decode; one codec thread"}}


def decode_selected(video, budget):
    import av
    selection = video["selections"][str(budget)]
    indices = uniform_indices(video["timestamps"], video["duration"], budget)
    if indices != selection["indices"]:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Frozen sampling mismatch")
    wanted = set(indices)
    selected, count = [], 0
    path = Path(video["path"])
    if path.stat().st_size != video["file_size"]:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Video size changed")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            count += 1
            if frame.pts is None or index >= len(video["timestamps"]):
                raise Blocked("INPUT_INTEGRITY_FAILED", "Decoded timeline changed")
            if Fraction(frame.pts) * Fraction(stream.time_base) != Fraction(video["timestamps"][index]):
                raise Blocked("INPUT_INTEGRITY_FAILED", "Presentation timestamps changed")
            if (frame.height, frame.width) != (video["height"], video["width"]):
                raise Blocked("INPUT_INTEGRITY_FAILED", "Dimensions changed")
            if index in wanted:
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                selected.append(letterbox(image, video))
    if count != video["frame_count"] or len(selected) != budget:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Actual decoded/selected frame counts disagree")
    array = np.stack(selected)
    return array, {"frames_decoded": count, "frames_selected": budget,
                   "source_indices": indices, "actual_timestamps": selection["timestamps"]}

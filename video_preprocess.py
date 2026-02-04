from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ClipConfig:
    fps: float # 프레임
    clip_sec: float # 클립 길이 (초)
    overlap_sec: float = 0.0 # 겹치는 구간 길이 (초)
    start_sec: Optional[float] = None # None이면 영상 시작부터
    end_sec: Optional[float] = None # None이면 영상 끝까지
    include_tail: bool = False # 마지막 클립 포함 여부


def _sec_to_frame(sec: float, fps: float) -> int:
    return int(math.floor(sec * fps + 1e-9))


def _validate_config(cfg: ClipConfig) -> None:
    if cfg.fps <= 0:
        raise ValueError("fps must be > 0")
    if cfg.clip_sec <= 0:
        raise ValueError("clip_sec must be > 0")
    if cfg.overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")
    if cfg.overlap_sec >= cfg.clip_sec:
        raise ValueError("overlap_sec must be < clip_sec")
    if cfg.start_sec is not None and cfg.start_sec < 0:
        raise ValueError("start_sec must be >= 0 when provided")
    if cfg.end_sec is not None and cfg.end_sec < 0:
        raise ValueError("end_sec must be >= 0 when provided")


def compute_clip_frame_ranges(
    total_frames: int,
    cfg: ClipConfig,
) -> List[Tuple[int, int]]:
    """
    Return a list of (start_frame, end_frame) ranges, end exclusive.
    """
    _validate_config(cfg)
    if total_frames < 0:
        raise ValueError("total_frames must be >= 0")

    duration_sec = total_frames / cfg.fps if total_frames > 0 else 0.0
    start_sec = 0.0 if cfg.start_sec is None else max(0.0, cfg.start_sec)
    end_sec = duration_sec if cfg.end_sec is None else min(duration_sec, cfg.end_sec)
    if end_sec <= start_sec or duration_sec <= 0:
        return []

    step_sec = cfg.clip_sec - cfg.overlap_sec
    ranges: List[Tuple[int, int]] = []

    t = start_sec
    while t + cfg.clip_sec <= end_sec + 1e-9:
        s_f = _sec_to_frame(t, cfg.fps)
        e_f = _sec_to_frame(t + cfg.clip_sec, cfg.fps)
        if e_f > total_frames:
            e_f = total_frames
        if e_f > s_f:
            ranges.append((s_f, e_f))
        t += step_sec

    if cfg.include_tail and end_sec > start_sec:
        tail_start = end_sec - cfg.clip_sec
        if tail_start > start_sec + 1e-9:
            t = max(start_sec, tail_start)
            s_f = _sec_to_frame(t, cfg.fps)
            e_f = _sec_to_frame(end_sec, cfg.fps)
            if e_f > total_frames:
                e_f = total_frames
            if e_f > s_f and (not ranges or ranges[-1] != (s_f, e_f)):
                ranges.append((s_f, e_f))

    return ranges

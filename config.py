from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class ClipConfig:
    fps: float
    clip_sec: float
    overlap_sec: float = 0.0
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    include_tail: bool = False


@dataclass(frozen=True)
class ClusterScoreConfig:
    k_ratio: float = 0.1
    top_ratio_per_cluster: float = 0.2
    num_iters: int = 20
    eps: float = 1e-8
    normalize_before_concat: bool = True


@dataclass(frozen=True)
class FrameChangeConfig:
    step: int = 3
    eps: float = 1e-8


@dataclass(frozen=True)
class FinalScoreConfig:
    a: float = 1.0
    b: float = 0.5
    c: float = 0.0


@dataclass(frozen=True)
class IOConfig:
    input_video: Path
    output_dir: Path = Path("data/summary_videos")
    save_intermediate_clips: bool = False
    clips_dir: Path = Path("data/clip")
    write_audio: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    clip: ClipConfig = ClipConfig(fps=4, clip_sec=4.0, overlap_sec=2.0)
    cluster: ClusterScoreConfig = ClusterScoreConfig()
    frame_change: FrameChangeConfig = FrameChangeConfig()
    final_score: FinalScoreConfig = FinalScoreConfig()
    use_single_embedding_for_cluster: bool = True


@dataclass(frozen=True)
class ModelLoaderConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_vit_id: str = "openai/clip-vit-base-patch32"
    pe_av_id: str = "facebook/pe-av-base"
    use_fp16: bool = False

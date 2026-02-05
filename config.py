from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class ClipConfig: # 하단의 PipelineConfig에서 사용
    fps: float
    clip_sec: float
    overlap_sec: float = 0.0
    start_sec: Optional[float] = None # None이면 영상 시작부터
    end_sec: Optional[float] = None # None이면 영상 끝까지
    include_tail: bool = False


@dataclass(frozen=True)
class ClusterScoreConfig:
    k_ratio: float = 0.1 # main.py에서 사용중
    top_ratio_per_cluster: float = 0.2
    boundary_ratio_in_selected: float = 0.4 # 선택된 후보 중 경계 샘플 비율(0~1)
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
class SemanticScoreConfig:
    use_rare_score: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class SelectionConfig:
    summary_ratio: float = 0.1
    dynamic_weight: float = 0.5
    semantic_weight: float = 0.5
    lambda_mmr: float = 0.7
    min_gap_sec: float = 0.0


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
    final_score: FinalScoreConfig = FinalScoreConfig()  # deprecated, kept for compatibility
    semantic: SemanticScoreConfig = SemanticScoreConfig()
    selection: SelectionConfig = SelectionConfig()
    use_single_embedding_for_cluster: bool = True


@dataclass(frozen=True)
class ModelLoaderConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_vit_id: str = "openai/clip-vit-base-patch32"
    pe_av_id: str = "facebook/pe-av-base"
    use_fp16: bool = False

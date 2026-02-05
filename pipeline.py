from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from tqdm import tqdm

from score import (
    cluster_center_scores,
    cluster_center_scores_single,
    clip_frame_change_score,
    rare_semantic_scores,
)
from video_preprocess import compute_clip_frame_ranges
from model_loader import load_models
from config import IOConfig, PipelineConfig, ModelLoaderConfig, ClipConfig


@dataclass(frozen=True)
class Models:
    video_encoder: torch.nn.Module  # VideoMAEv2
    audio_encoder: torch.nn.Module  # BEATs
    vit_model: torch.nn.Module      # CLIP ViT (frame-level)




def _get_video_info(video_path: Path) -> Tuple[float, int, float]:
    try:
        import av
    except Exception as exc:
        raise ImportError(
            "PyAV is required to probe video info. Install with `pip install av`."
        ) from exc

    container = av.open(str(video_path))
    vstream = next((s for s in container.streams if s.type == "video"), None)
    if vstream is None or vstream.average_rate is None:
        raise ValueError("Failed to read video stream info.")

    fps = float(vstream.average_rate)
    total_frames = vstream.frames
    if not total_frames or total_frames <= 0:
        if container.duration is None:
            raise ValueError("Failed to infer total_frames from container.")
        duration_sec = container.duration / 1_000_000.0
        total_frames = int(round(duration_sec * fps))

    astream = next((s for s in container.streams if s.type == "audio"), None)
    audio_sr = float(astream.rate) if astream and astream.rate else 16000.0

    container.close()
    return fps, total_frames, audio_sr


def _load_clip_segment(
    video_path: Path,
    start_sec: float,
    end_sec: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    from torchvision.io import read_video

    frames, audio, info = read_video(
        str(video_path),
        start_pts=start_sec,
        end_pts=end_sec,
        pts_unit="sec",
    )
    frames = frames.permute(0, 3, 1, 2).contiguous()
    if audio.numel() == 0:
        audio = torch.zeros(0, dtype=torch.float32)

    fps = float(info.get("video_fps", 0.0)) or 0.0
    audio_sr = float(info.get("audio_fps", 0.0)) or 0.0
    if fps <= 0:
        raise ValueError("Failed to detect video fps for clip segment.")
    if audio_sr <= 0:
        audio_sr = 16000.0

    return frames, audio, fps, audio_sr


def _downsample_frames(
    frames: torch.Tensor,
    src_fps: float,
    dst_fps: float,
) -> torch.Tensor:
    if dst_fps <= 0:
        raise ValueError("dst_fps must be > 0")
    if abs(dst_fps - src_fps) < 1e-3:
        return frames

    t = frames.size(0)
    num = max(1, int(round(t * dst_fps / src_fps)))
    idx = torch.linspace(0, t - 1, steps=num, device=frames.device)
    idx = idx.round().long().clamp(0, t - 1)
    return frames[idx]


def _encode_clip_av(
    clip_frames: torch.Tensor,
    clip_audio: torch.Tensor,
    video_encoder: torch.nn.Module,
    audio_encoder: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        v_emb: (D_v,)
        a_emb: (D_a,)
    """
    with torch.no_grad():
        if video_encoder is audio_encoder:
            v_emb, a_emb = video_encoder(clip_frames, clip_audio)
        else:
            v_emb = video_encoder(clip_frames)
            a_emb = audio_encoder(clip_audio)
    if v_emb.dim() > 1:
        v_emb = v_emb.mean(dim=0)
    if a_emb.dim() > 1:
        a_emb = a_emb.mean(dim=0)
    return v_emb, a_emb


def _minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_min = torch.min(x)
    x_max = torch.max(x)
    denom = (x_max - x_min).clamp_min(eps)
    return (x - x_min) / denom


def _select_with_mmr(
    candidate_indices: List[int],
    base_scores: torch.Tensor,
    emb: torch.Tensor,
    ranges: List[Tuple[int, int]],
    fps: float,
    target_count: int,
    lambda_mmr: float,
    min_gap_sec: float,
    eps: float = 1e-8,
) -> List[int]:
    if not candidate_indices:
        return []

    z = emb / (emb.norm(p=2, dim=-1, keepdim=True) + eps)
    chosen: List[int] = []
    remaining = set(candidate_indices)
    sorted_candidates = sorted(candidate_indices)

    while remaining and len(chosen) < target_count:
        best_idx = None
        best_score = float("-inf")

        for idx in sorted_candidates:
            if idx not in remaining:
                continue
            if min_gap_sec > 0.0:
                start_sec = ranges[idx][0] / fps
                if any(abs(start_sec - (ranges[j][0] / fps)) < min_gap_sec for j in chosen):
                    continue

            rel = float(base_scores[idx].item())
            if not chosen:
                mmr_score = rel
            else:
                sims = torch.matmul(z[chosen], z[idx])
                redundancy = float(torch.max(sims).item())
                mmr_score = (lambda_mmr * rel) - ((1.0 - lambda_mmr) * redundancy)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        chosen.append(best_idx)
        remaining.remove(best_idx)

    return chosen




def run_pipeline(io: IOConfig, cfg: PipelineConfig, models: Models) -> Path:
    io.output_dir.mkdir(parents=True, exist_ok=True)
    if io.save_intermediate_clips:
        io.clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {io.output_dir.resolve()}")
    fps, total_frames, audio_sr = _get_video_info(io.input_video)
    print(f"[INFO] Video info: fps={fps}, total_frames={total_frames}, audio_sr={audio_sr}")

    clip_cfg = ClipConfig(
        fps=fps,
        clip_sec=cfg.clip.clip_sec,
        overlap_sec=cfg.clip.overlap_sec,
        start_sec=cfg.clip.start_sec,
        end_sec=cfg.clip.end_sec,
        include_tail=cfg.clip.include_tail,
    )

    ranges = compute_clip_frame_ranges(total_frames, clip_cfg)
    if not ranges:
        raise ValueError("No clip ranges computed; check clip config and video length.")
    print(f"[INFO] Clip count: {len(ranges)}")

    # Global audio is no longer loaded here; resampling happens per-clip.

    v_embs: List[torch.Tensor] = []
    a_embs: List[torch.Tensor] = []

    for s, e in tqdm(ranges, desc="Clip embeddings", unit="clip"):
        start_sec = s / fps
        end_sec = e / fps
        clip_frames, clip_audio, clip_fps, clip_audio_sr = _load_clip_segment(
            io.input_video, start_sec, end_sec
        )
        if cfg.clip.fps != clip_fps:
            clip_frames = _downsample_frames(clip_frames, clip_fps, cfg.clip.fps)
        target_sr = 48000
        if clip_audio.numel() == 0:
            clip_audio_mono = torch.zeros(
                int(round(cfg.clip.clip_sec * target_sr)), dtype=torch.float32
            )
            clip_audio_sr = target_sr
        else:
            clip_audio_mono = (
                clip_audio.mean(dim=1) if clip_audio.dim() == 2 else clip_audio
            )
        if int(round(clip_audio_sr)) != target_sr:
            import torchaudio

            clip_audio_mono = torchaudio.functional.resample(
                clip_audio_mono,
                orig_freq=int(round(clip_audio_sr)),
                new_freq=target_sr,
            )
        v_emb, a_emb = _encode_clip_av(
            clip_frames, clip_audio_mono, models.video_encoder, models.audio_encoder
        )
        v_embs.append(v_emb)
        a_embs.append(a_emb)

    v_mat = torch.stack(v_embs, dim=0)
    a_mat = torch.stack(a_embs, dim=0)

    if cfg.use_single_embedding_for_cluster:
        fused = torch.cat([v_mat, a_mat], dim=-1)
        cluster_scores, selected_mask, assignments = cluster_center_scores_single(
            fused, cfg.cluster
        )
    else:
        cluster_scores, selected_mask, assignments = cluster_center_scores(
            v_mat, a_mat, cfg.cluster
        )
    selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1).tolist()
    print(f"[INFO] Candidate clips for frame-change scoring: {len(selected_indices)}")

    change_scores = torch.zeros_like(cluster_scores)
    for idx in tqdm(selected_indices, desc="Frame change", unit="clip"):
        s, e = ranges[idx]
        start_sec = s / fps
        end_sec = e / fps
        clip_frames, _, clip_fps, _ = _load_clip_segment(
            io.input_video, start_sec, end_sec
        )
        if cfg.clip.fps != clip_fps:
            clip_frames = _downsample_frames(clip_frames, clip_fps, cfg.clip.fps)
        change_scores[idx] = clip_frame_change_score(
            clip_frames, models.vit_model, cfg.frame_change
        )

    if not (0.0 < cfg.selection.summary_ratio <= 1.0):
        raise ValueError("selection.summary_ratio must be in (0, 1]")
    if not (0.0 <= cfg.selection.lambda_mmr <= 1.0):
        raise ValueError("selection.lambda_mmr must be in [0, 1]")
    if cfg.selection.min_gap_sec < 0.0:
        raise ValueError("selection.min_gap_sec must be >= 0")
    if cfg.selection.dynamic_weight < 0.0 or cfg.selection.semantic_weight < 0.0:
        raise ValueError("selection dynamic/semantic weights must be >= 0")
    if (cfg.selection.dynamic_weight + cfg.selection.semantic_weight) <= 0.0:
        raise ValueError("sum of selection weights must be > 0")

    fused_semantic = torch.cat([v_mat, a_mat], dim=-1)
    if cfg.semantic.use_rare_score:
        semantic_scores = rare_semantic_scores(fused_semantic, cfg.semantic)
    else:
        semantic_scores = torch.zeros_like(cluster_scores)

    candidate_idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
    cand_change = _minmax_normalize(change_scores[candidate_idx], cfg.frame_change.eps)
    cand_semantic = _minmax_normalize(semantic_scores[candidate_idx], cfg.semantic.eps)
    weight_sum = cfg.selection.dynamic_weight + cfg.selection.semantic_weight
    w_dyn = cfg.selection.dynamic_weight / weight_sum
    w_sem = cfg.selection.semantic_weight / weight_sum
    cand_base = (w_dyn * cand_change) + (w_sem * cand_semantic)

    base_scores = torch.zeros_like(cluster_scores)
    base_scores[candidate_idx] = cand_base

    target_count = max(1, int(round(len(ranges) * cfg.selection.summary_ratio)))
    target_count = min(target_count, int(candidate_idx.numel()))
    chosen = _select_with_mmr(
        candidate_indices=selected_indices,
        base_scores=base_scores,
        emb=fused_semantic,
        ranges=ranges,
        fps=fps,
        target_count=target_count,
        lambda_mmr=cfg.selection.lambda_mmr,
        min_gap_sec=cfg.selection.min_gap_sec,
        eps=cfg.semantic.eps,
    )

    chosen = sorted(chosen, key=lambda i: ranges[i][0])
    if not chosen:
        raise ValueError("No clips selected after scoring.")
    print(f"[INFO] Final selected clips: {len(chosen)}")

    out_frames_list: List[torch.Tensor] = []
    out_audio = None
    audio_segments = []
    output_audio_sr: Optional[float] = None
    output_audio_channels: Optional[int] = None
    for i in chosen:
        s, e = ranges[i]
        start_sec = s / fps
        end_sec = e / fps
        clip_frames, clip_audio, clip_fps, clip_audio_sr = _load_clip_segment(
            io.input_video, start_sec, end_sec
        )
        out_frames_list.append(clip_frames)
        if clip_audio.numel() > 0:
            if output_audio_sr is None:
                output_audio_sr = clip_audio_sr
            if int(round(clip_audio_sr)) != int(round(output_audio_sr)):
                import torchaudio

                if clip_audio.dim() == 1:
                    clip_audio = torchaudio.functional.resample(
                        clip_audio,
                        orig_freq=int(round(clip_audio_sr)),
                        new_freq=int(round(output_audio_sr)),
                    )
                else:
                    clip_audio_t = clip_audio.transpose(0, 1)
                    clip_audio_t = torchaudio.functional.resample(
                        clip_audio_t,
                        orig_freq=int(round(clip_audio_sr)),
                        new_freq=int(round(output_audio_sr)),
                    )
                    clip_audio = clip_audio_t.transpose(0, 1)
            # Ensure shape is (num_samples, channels)
            if clip_audio.dim() == 2 and clip_audio.size(0) <= 2 and clip_audio.size(1) > 2:
                clip_audio = clip_audio.transpose(0, 1)
            if clip_audio.dim() == 1:
                clip_audio = clip_audio.unsqueeze(-1)
            if output_audio_channels is None:
                output_audio_channels = clip_audio.size(1)
            elif clip_audio.size(1) != output_audio_channels:
                # Fallback: mix down to mono if channel count differs
                clip_audio = clip_audio.mean(dim=1, keepdim=True)
                output_audio_channels = 1
            audio_segments.append(clip_audio)

    out_frames = torch.cat(out_frames_list, dim=0)
    if audio_segments:
        out_audio = torch.cat(audio_segments, dim=0)

    output_path = io.output_dir / f"{io.input_video.stem}_summary.mp4"
    print(f"[INFO] Writing summary video: {output_path.resolve()}")
    from torchvision.io import write_video

    if io.write_audio and out_audio is not None and out_audio.numel() > 0:
        # Ensure audio is (num_samples, channels) float32 on CPU
        if out_audio.dim() == 1:
            out_audio = out_audio.unsqueeze(-1)
        out_audio = out_audio.to(dtype=torch.float32).cpu()
        # PyAV expects planar audio as (channels, samples) for float formats
        audio_array = out_audio.transpose(0, 1).contiguous()
        write_video(
            str(output_path),
            out_frames.permute(0, 2, 3, 1).contiguous(),
            fps=fps,
            audio_array=audio_array,
            audio_fps=int(output_audio_sr or audio_sr),
            audio_codec="aac",
        )
    else:
        write_video(
            str(output_path),
            out_frames.permute(0, 2, 3, 1).contiguous(),
            fps=fps,
        )

    return output_path


def run_pipeline_with_default_models(
    io: IOConfig,
    cfg: PipelineConfig,
    model_cfg: Optional[ModelLoaderConfig] = None,
) -> Path:
    """
    Convenience wrapper: loads models via model_loader and runs pipeline.
    """
    if model_cfg is None:
        model_cfg = ModelLoaderConfig()
    video_encoder, audio_encoder, vit_model = load_models(model_cfg)
    models = Models(
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        vit_model=vit_model,
    )
    return run_pipeline(io, cfg, models)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import numpy as np


@dataclass(frozen=True)
class ModelLoaderConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_vit_id: str = "openai/clip-vit-base-patch32"
    pe_av_id: str = "facebook/pe-av-base"
    use_fp16: bool = False


class ClipVisionWrapper(nn.Module):
    def __init__(self, model: nn.Module, processor) -> None:
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (T, C, H, W), returns (T, D)
        """
        if frames.dim() != 4:
            raise ValueError("frames must be (T, C, H, W)")
        x = frames.detach().cpu().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        inputs = self.processor(images=list(x), return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]


class PEAVWrapper(nn.Module):
    def __init__(self, model: nn.Module, processor) -> None:
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, clip_frames: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        clip_frames: (T, C, H, W)
        audio: (N,) 16k waveform
        Returns (video_emb, audio_emb)
        """
        if clip_frames.dim() != 4:
            raise ValueError("clip_frames must be (T, C, H, W)")
        if audio.dim() != 1:
            raise ValueError("audio must be 1D waveform (N,)")

        x = clip_frames.detach().cpu().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        sr = 48000
        if hasattr(self.processor, "feature_extractor"):
            sr = getattr(self.processor.feature_extractor, "sampling_rate", sr)
        inputs = self.processor(
            videos=[x],
            audio=[audio.detach().cpu().numpy()],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model(**inputs)
        if hasattr(out, "video_embeds") and out.video_embeds is not None:
            v_emb = out.video_embeds.squeeze(0)
        elif hasattr(out, "visual_embeds") and out.visual_embeds is not None:
            v_emb = out.visual_embeds.squeeze(0)
        else:
            raise RuntimeError("PE-AV did not return video embeddings.")

        if hasattr(out, "audio_embeds") and out.audio_embeds is not None:
            a_emb = out.audio_embeds.squeeze(0)
        else:
            raise RuntimeError("PE-AV did not return audio embeddings.")

        return v_emb, a_emb




def load_models(cfg: ModelLoaderConfig) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Returns (video_encoder, audio_encoder, vit_model).
    """
    try:
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionModel,
            PeAudioVideoModel,
            PeAudioVideoProcessor,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required to load models. "
            "Install with `pip install transformers`."
        ) from exc

    device = torch.device(cfg.device)
    dtype = torch.float16 if cfg.use_fp16 else torch.float32

    # CLIP ViT
    clip_vit = CLIPVisionModel.from_pretrained(cfg.clip_vit_id)
    clip_vit = clip_vit.to(device=device, dtype=dtype).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(cfg.clip_vit_id)
    vit_model = ClipVisionWrapper(clip_vit, clip_processor)

    # PE-AV (audio + video)
    pe_av = PeAudioVideoModel.from_pretrained(cfg.pe_av_id)
    pe_av = pe_av.to(device=device, dtype=dtype).eval()
    pe_av_processor = PeAudioVideoProcessor.from_pretrained(cfg.pe_av_id)
    pe_av_wrapper = PEAVWrapper(pe_av, pe_av_processor)

    return pe_av_wrapper, pe_av_wrapper, vit_model

from __future__ import annotations

from typing import Tuple

import torch

from config import ClusterScoreConfig, FinalScoreConfig, FrameChangeConfig


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    a_n = _normalize(a, eps)
    b_n = _normalize(b, eps)
    return (a_n * b_n).sum(dim=-1)


def _concat_av(
    v_emb: torch.Tensor,
    a_emb: torch.Tensor,
    normalize_before_concat: bool,
    eps: float,
) -> torch.Tensor:
    if normalize_before_concat:
        v_emb = _normalize(v_emb, eps)
        a_emb = _normalize(a_emb, eps)
    return torch.cat([v_emb, a_emb], dim=-1)


def _kmeans(
    x: torch.Tensor,
    k: int,
    num_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple k-means with k-means++ init.
    Returns (centroids, assignments).
    """
    n = x.size(0)
    if k <= 0:
        raise ValueError("k must be > 0")
    if n == 0:
        raise ValueError("x must have at least 1 row")

    k = min(k, n)
    # k-means++ initialization
    centroids = []
    first_idx = torch.randint(0, n, (1,), device=x.device).item()
    centroids.append(x[first_idx])
    for _ in range(1, k):
        dists = torch.cdist(x, torch.stack(centroids, dim=0), p=2)
        min_dist_sq = dists.min(dim=1).values.pow(2)
        probs = min_dist_sq / (min_dist_sq.sum() + eps)
        next_idx = torch.multinomial(probs, 1).item()
        centroids.append(x[next_idx])
    centroids = torch.stack(centroids, dim=0).clone()

    for _ in range(num_iters):
        dists = torch.cdist(x, centroids, p=2)
        assign = torch.argmin(dists, dim=1)
        new_centroids = []
        for ci in range(k):
            mask = assign == ci
            if mask.any():
                new_centroids.append(x[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[ci])
        centroids = torch.stack(new_centroids, dim=0)

    dists = torch.cdist(x, centroids, p=2)
    assign = torch.argmin(dists, dim=1)
    return centroids, assign


def cluster_center_scores(
    v_emb: torch.Tensor,
    a_emb: torch.Tensor,
    cfg: ClusterScoreConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Score clips by closeness to cluster centroids.
    Returns (scores, selected_mask, assignments) where selected_mask picks
    top_ratio_per_cluster closest clips per cluster.
    """
    if v_emb.dim() != 2 or a_emb.dim() != 2:
        raise ValueError("v_emb and a_emb must be 2D tensors (N, D)")
    if v_emb.size(0) != a_emb.size(0):
        raise ValueError("v_emb and a_emb must have same number of clips")
    if not (0.0 < cfg.k_ratio <= 1.0):
        raise ValueError("k_ratio must be in (0, 1]")
    if not (0.0 < cfg.top_ratio_per_cluster <= 1.0):
        raise ValueError("top_ratio_per_cluster must be in (0, 1]")

    x = _concat_av(v_emb, a_emb, cfg.normalize_before_concat, cfg.eps)
    n = x.size(0)
    k = max(1, int(round(n * cfg.k_ratio)))
    centroids, assign = _kmeans(x, k, cfg.num_iters, cfg.eps)

    dists = torch.cdist(x, centroids, p=2)
    min_dists = dists[torch.arange(n, device=x.device), assign]
    scores = 1.0 / (1.0 + min_dists)

    selected = torch.zeros(n, dtype=torch.bool, device=x.device)
    for ci in range(k):
        idx = torch.nonzero(assign == ci, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        dist_ci = min_dists[idx]
        k_ci = max(1, int(round(idx.numel() * cfg.top_ratio_per_cluster)))
        k_ci = min(k_ci, idx.numel())
        top_idx = idx[torch.topk(dist_ci, k=k_ci, largest=False).indices]
        selected[top_idx] = True

    return scores, selected, assign


def cluster_center_scores_single(
    emb: torch.Tensor,
    cfg: ClusterScoreConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Score clips by closeness to cluster centroids using a single embedding.
    Returns (scores, selected_mask, assignments).
    """
    if emb.dim() != 2:
        raise ValueError("emb must be a 2D tensor (N, D)")
    if not (0.0 < cfg.k_ratio <= 1.0):
        raise ValueError("k_ratio must be in (0, 1]")
    if not (0.0 < cfg.top_ratio_per_cluster <= 1.0):
        raise ValueError("top_ratio_per_cluster must be in (0, 1]")

    n = emb.size(0)
    k = max(1, int(round(n * cfg.k_ratio)))
    centroids, assign = _kmeans(emb, k, cfg.num_iters, cfg.eps)

    dists = torch.cdist(emb, centroids, p=2)
    min_dists = dists[torch.arange(n, device=emb.device), assign]
    scores = 1.0 / (1.0 + min_dists)

    selected = torch.zeros(n, dtype=torch.bool, device=emb.device)
    for ci in range(k):
        idx = torch.nonzero(assign == ci, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        dist_ci = min_dists[idx]
        k_ci = max(1, int(round(idx.numel() * cfg.top_ratio_per_cluster)))
        k_ci = min(k_ci, idx.numel())
        top_idx = idx[torch.topk(dist_ci, k=k_ci, largest=False).indices]
        selected[top_idx] = True

    return scores, selected, assign


def _ensure_frame_emb(emb: torch.Tensor) -> torch.Tensor:
    if emb.dim() == 3 and emb.size(0) == 1:
        emb = emb.squeeze(0)
    if emb.dim() != 2:
        raise ValueError("frame embeddings must be a 2D tensor (T, D)")
    return emb


def clip_frame_change_score(
    frames: torch.Tensor,
    vit_model: torch.nn.Module,
    cfg: FrameChangeConfig,
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Compute average change score inside a clip using downsampled frames.
    frames: (T, C, H, W)
    vit_model: returns frame embeddings shaped (T, D) or (1, T, D)
    """
    if frames.dim() != 4:
        raise ValueError("frames must be a 4D tensor (T, C, H, W)")
    if cfg.step <= 0:
        raise ValueError("step must be > 0")

    if use_no_grad:
        with torch.no_grad():
            frame_emb = vit_model(frames)
    else:
        frame_emb = vit_model(frames)
    frame_emb = _ensure_frame_emb(frame_emb)

    t = frame_emb.size(0)
    if t < 2:
        return torch.tensor(0.0, device=frames.device)

    idx = torch.arange(0, t, cfg.step, device=frame_emb.device)
    if idx.numel() < 2:
        return torch.tensor(0.0, device=frames.device)

    emb_a = frame_emb[idx[:-1]]
    emb_b = frame_emb[idx[1:]]
    cos = _cosine_similarity(emb_a, emb_b, cfg.eps)
    chg = (1.0 - cos) * 0.5
    return chg.mean()


def combine_cluster_and_change_scores(
    cluster_scores: torch.Tensor,
    change_scores: torch.Tensor,
    cfg: FinalScoreConfig,
) -> torch.Tensor:
    """
    Combine scores: a*C + b*D + c*(C*D)
    """
    if cluster_scores.shape != change_scores.shape:
        raise ValueError("cluster_scores and change_scores must have same shape")
    return cfg.a * cluster_scores + cfg.b * change_scores + cfg.c * (
        cluster_scores * change_scores
    )

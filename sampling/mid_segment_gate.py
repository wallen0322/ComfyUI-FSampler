# sampling/mid_segment_gate.py
# MIT License
# Minimal mid-segment quality gate for ComfyUI-FSampler.
# Implements: mid-segment restriction, sigma-aware cosine threshold, and Sobel-based band-energy validation.
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    num = (a_flat * b_flat).sum(dim=1)
    denom = torch.linalg.norm(a_flat, dim=1) * torch.linalg.norm(b_flat, dim=1) + eps
    return (num / denom).mean()

@torch.no_grad()
def _sobel_energy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    N, C, H, W = x.shape
    kx = torch.tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)
    pad = (1,1,1,1)
    gx = F.conv2d(x, kx, padding=pad, groups=C)
    gy = F.conv2d(x, ky, padding=pad, groups=C)
    grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-12)
    high = (grad_mag * grad_mag).mean()
    total = (x * x).mean() + 1e-12
    return high, total

@torch.no_grad()
def _band_ratio_ok(eps_new: torch.Tensor, eps_ref: torch.Tensor, tol: float = 0.35) -> bool:
    hn, tn = _sobel_energy(eps_new)
    hr, tr = _sobel_energy(eps_ref)
    rn = (hn / (tn + 1e-12)).clamp(min=0.0, max=10.0)
    rr = (hr / (tr + 1e-12)).clamp(min=0.0, max=10.0)
    rel = torch.abs(rn - rr) / (rr + 1e-6)
    return bool(rel <= tol)

@torch.no_grad()
def _sigma_norm(sigma: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    s = sigma.detach().float().mean()
    if sigma_max <= sigma_min + 1e-9:
        return torch.tensor(0.5, device=sigma.device, dtype=torch.float32)
    return ((s - sigma_min) / (sigma_max - sigma_min)).clamp(0.0, 1.0)

@torch.no_grad()
def decide_skip(
    eps_extrap: torch.Tensor,
    eps_last_real: torch.Tensor,
    sigma_cur: torch.Tensor,
    step_idx: int,
    num_steps: int,
    *,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    protect_first_steps: int = 2,
    protect_last_steps: int = 2,
    max_consec_skips: int = 1,
    consec_skips: int = 0,
    cos_threshold: float = 0.986,
    band_tol: float = 0.35,
) -> bool:
    # Head/tail force REAL
    if step_idx < protect_first_steps:
        return False
    if step_idx >= (num_steps - protect_last_steps):
        return False

    # Only skip in mid segment
    r = step_idx / max(1, (num_steps - 1))
    if (r < 0.15) or (r > 0.85):
        return False
    if consec_skips >= max_consec_skips:
        return False

    s_norm = float(_sigma_norm(sigma_cur, sigma_min, sigma_max))
    cos_th = float(cos_threshold + 0.005 * (1.0 - s_norm))  # tighten when sigma is low
    cos_ok = float(_cosine_similarity(eps_extrap, eps_last_real)) >= cos_th
    if not cos_ok:
        return False

    if not _band_ratio_ok(eps_extrap, eps_last_real, tol=band_tol):
        return False

    return True

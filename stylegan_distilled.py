import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Dict, List
import copy

# Style extractor for the new ToRGB
# -----------------------------
def _style_for_block_torgb(cur_ws: torch.Tensor, num_conv: int) -> torch.Tensor:
    """Choose a style vector to feed the new ToRGBLayer.
       Here we take the mean of the block's conv styles as a stable choice.
       cur_ws: [N, num_conv + num_torgb (0/1), w_dim]
    """
    if num_conv <= 0:
        raise ValueError("Block has no convs to draw style from.")
    w_conv = cur_ws[:, :num_conv, :]               # [N, num_conv, w_dim]
    w_rgb  = w_conv.mean(dim=1, keepdim=False)     # [N, w_dim]
    return w_rgb


# ---------- tiny utility: probe feature shape at tap (start_res) ----------
@torch.no_grad()
def _probe_feature_shape(G: nn.Module,
                         start_res: int,
                         truncation_psi: float = 1.0,
                         device: Optional[torch.device] = None) -> Tuple[int, int]:
    """Return (C, R) at the chosen synthesis tap (start_res)."""
    if device is None:
        device = next(G.parameters()).device
    z_dim = getattr(G, "z_dim", 512)
    z = torch.randn(2, z_dim, device=device)
    ws = G.mapping(z, None, truncation_psi=truncation_psi)
    x, _, _, _ = run_until_resolution(G, ws, start_res, noise_mode="const")
    C, R = int(x.shape[1]), int(x.shape[2])
    return C, R

# ========= target size helper (uses your probe) =========
@torch.no_grad()
def _target_size(G: torch.nn.Module, start_res: int, use_sr_head: bool, truncation_psi: float, device: torch.device):
    from typing import Tuple
    C, R = _probe_feature_shape(G, start_res, truncation_psi, device)  # from your loader module
    return (2*R, 2*R) if use_sr_head else (R, R)


# Assumes these are available from your StyleGAN codebase:
# - ToRGBLayer (you provided)
# - bias_act, modulated_conv2d, etc. imported by ToRGBLayer
# - Generator / SynthesisNetwork definitions


# ---------- checkpoint helpers ----------
def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Accepts several common checkpoint layouts and returns a state_dict."""
    if "head" in ckpt and isinstance(ckpt["head"], dict):
        return ckpt["head"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    # Some tools save the raw state dict directly
    # Heuristic: a mapping of str -> Tensor
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt  # looks like a plain state_dict
    raise ValueError("Unsupported checkpoint format: cannot find model weights.")



# -----------------------------
# Utilities to walk Synthesis
# -----------------------------
@torch.no_grad()
def run_until_resolution(G, ws: torch.Tensor, stop_res: int, noise_mode: str = "const", **block_kwargs):
    """Run synthesis up to and including `stop_res`, return:
       x_stop, img_stop, last_block_cur_ws, next_w_idx
    """
    S = G.synthesis
    x = img = None
    w_idx = 0
    for res in S.block_resolutions:
        block = getattr(S, f"b{res}")
        cur_len = block.num_conv + block.num_torgb
        cur_ws  = ws.narrow(1, w_idx, cur_len)
        x, img  = block(x, img, cur_ws, noise_mode=noise_mode, **block_kwargs)
        w_idx  += block.num_conv
        if res == stop_res:
            return x, img, cur_ws, w_idx  # w_idx now points to NEXT block's conv styles
    raise ValueError(f"stop_res {stop_res} not found in block_resolutions {S.block_resolutions}")

@torch.no_grad()
def resume_from_next_block(G, x, img, ws: torch.Tensor, next_w_idx: int, start_res: int,
                           noise_mode: str = "const", **block_kwargs):
    """Continue synthesis from the block AFTER `start_res` to the end (frozen)."""
    S = G.synthesis
    begin = False
    w_idx = next_w_idx
    for res in S.block_resolutions:
        block = getattr(S, f"b{res}")
        cur_len = block.num_conv + block.num_torgb
        if not begin:
            if res == start_res:
                begin = True
            continue  # skip the block where we stopped (already executed)
        cur_ws = ws.narrow(1, w_idx, cur_len)
        x, img = block(x, img, cur_ws, noise_mode=noise_mode, **block_kwargs)
        w_idx += block.num_conv
    return img  # [N, 3, H, W] in [-1,1]

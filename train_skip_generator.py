import math
import os
import sys
from typing import Tuple, Optional

from kiwisolver import strength
import torch
import torch.nn.functional as F
from torch import nn, optim
import pickle
from utils import *


import time
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid
# ========= minimal helpers reused by both =========
import os, time
from typing import Optional, Any, Dict, List
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    import lpips  # pip install lpips
except Exception:
    lpips = None

# ========= tiny ckpt util for optimizer/step (pairs with load_torgb_head) =========
def _load_opt_and_step(opt: torch.optim.Optimizer, resume_path: Optional[str], device: torch.device) -> int:
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        step = int(ckpt.get("step", 0))
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
                print(f"[resume] optimizer state loaded from {resume_path}")
            except Exception as e:
                print(f"[resume] optimizer state load failed: {e}")
        return step
    return 0



# -----------------------------
# Training loop (MPS + TensorBoard) with selectable head
# -----------------------------
def _train_torgb(
    G,                          # Frozen StyleGAN generator (nn.Module with .mapping and .synthesis)
    start_res: int,             # Resolution where we tap features (e.g., 16, 32, 64)
    steps: int = 2000,          # Training steps
    batch: int = 16,            # Batch size
    lr: float = 1e-3,           # Learning rate
    z_dim: Optional[int] = None,# If None, uses G.z_dim
    truncation_psi: float = 1.0,
    noise_mode: str = "const",
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    head_checkpoint: Optional[str] = None,
    log_every: int = 100,
    tb_logdir: str = "runs/torgb_mps",
    # ---- NEW: choose head ----
    use_sr_head: bool = True,          # False -> ToRGBWrapper @ start_res, True -> SRToRGBHead 64->128
    # ---- NEW: SR head hyperparams (only used if use_sr_head=True) ----
    sr_mid_ch: int = 256,
    sr_num_style_blocks: int = 2,
    sr_num_heads: int = 4,
    sr_window: int = 8,
    sr_use_skip64: bool = True,
) -> "nn.Module":
    """
    Trains either:
      - ToRGBWrapper at `start_res` to approximate the downsampled final RGB (R x R), or
      - SRToRGBHead to map `start_res` features to a 2x super-res RGB (2R x 2R).
    Generator stays frozen (no grad). Logs loss and pred/target previews to TensorBoard.
    Returns the trained head (ToRGBWrapper or SRToRGBHead).
    """
    import os
    import torch.optim as optim

    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    if device is None:
        device = next(G.parameters()).device
    if z_dim is None:
        z_dim = G.z_dim

    writer = SummaryWriter(log_dir=tb_logdir)
    # 1) Build/load the head in one line
    head = load_torgb_head(
        G, start_res,
        checkpoint=head_checkpoint,
        use_sr_head=use_sr_head,
        device=device,
        eval_mode=False,        # we will train it
        strict=False
    ).to(device)
    target_size = (2 * start_res, 2 * start_res)


    opt = optim.Adam(head.parameters(), lr=lr, betas=(0.9, 0.99))

    t0 = time.time()
    for step in range(1, steps + 1):
        # ---- sample z ----
        z = torch.randn(batch, z_dim, device=device)

        with torch.no_grad():
            # Map to ws, run up to the tap (start_res)
            ws = G.mapping(z, None, truncation_psi=truncation_psi)                      # [B, num_ws, w_dim]
            x, img_mid, cur_ws, next_w_idx = run_until_resolution(G, ws, start_res, noise_mode=noise_mode)
            # Final image, then resize to the supervision size (R or 2R)
            img_full = resume_from_next_block(G, x, img_mid, ws, next_w_idx, start_res, noise_mode=noise_mode)
            target   = F.interpolate(img_full, size=target_size, mode="bilinear", align_corners=False)  # [B, 3, Ht, Wt]
            # Style vector for ToRGB-like conditioning
            num_conv = cur_ws.shape[1]
            w_rgb    = _style_for_block_torgb(cur_ws, num_conv)                           # [B, w_dim]

        # ---- forward through chosen head ----
        pred = head(x, w_rgb)                                                             # [B, 3, Ht, Wt]

        # ---- L1 loss (you can add perceptual later) ----
        loss_l1 = (pred - target).abs().mean()
        loss = loss_l1

        # ---- optimize ----
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # ---- logging ----
        if (step % log_every) == 0 or step == 1:
            dt = (time.time() - t0); t0 = time.time()
            print(f"[train_torgb] step {step:>5}/{steps} | loss_l1={loss_l1.item():.5f} | {dt*1000:.1f} ms/it")

            writer.add_scalar("loss/l1", float(loss_l1.item()), global_step=step)
            writer.add_scalar("opt/lr", opt.param_groups[0]["lr"], global_step=step)

            grid = _grid_preview(pred.detach(), target.detach(), k=min(4, batch))
            writer.add_image("preview_pred_vs_target", grid, global_step=step)  # CHW [0,1]

            if step % (log_every * 5) == 0:
                writer.flush()

    writer.flush()
    writer.close()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(head.state_dict(), save_path)
        print(f"[train_torgb] Saved head to {save_path}")

    return head


# if __name__ == "__main__":
#     G, device = load_generator("./models/stylegan2-ffhq-1024x1024.pkl", "mps")
#     # G is your loaded StyleGAN generator (StyleGAN2-ADA PyTorch)
#     device = next(G.parameters()).device

    # # Train a ToRGB at 64x64:
    # wrapper_64 = _train_torgb(
    #     G,
    #     start_res=64,
    #     steps=1000,
    #     batch=8,
    #     lr=1e-3,
    #     truncation_psi=1.,
    #     noise_mode="const",
    #     device=device,
    #     save_path="./models/torgb_64to128_2.pth",
    #     head_checkpoint="./models/torgb_64to128 copy.pth",
    #     log_every=100
    # )

    # # Inference: get feature map at 64x64 and render a preview
    # with torch.no_grad():
    #     z = torch.randn(4, G.z_dim, device=device)
    #     ws = G.mapping(z, None, truncation_psi=1.)
    #     x64, _, cur_ws, _ = run_until_resolution(G, ws, 64)
    #     w_rgb = _style_for_block_torgb(cur_ws, num_conv=cur_ws.shape[1])
    #     preview = wrapper_64(x64, w_rgb)  # [4, 3, 16, 16] in [-1,1]




@torch.no_grad()
def benchmark_torgb(
    G: torch.nn.Module,
    *,
    start_res: int,
    z_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
    truncation_psi: float = 1.0,
    noise_mode: str = "const",
    # build/load head via loader:
    use_sr_head: bool = False,
    head_checkpoint: Optional[str] = None,
    compile_mode: Optional[str] = None,
    # batching & limits:
    batch_start: int = 1,
    batch_step: int = 1,
    max_batches: Optional[int] = None,
    max_mem_gb: float = 8.0,
    warmup_iters: int = 3,
    iters: int = 10,
    print_table: bool = True,
) -> List[Dict[str, Any]]:
    """
    Benchmarks the frozen G + head (built via load_torgb_head):
      - forward time (mapping + partial synthesis + head)
      - D2H transfer time
      - peak mem (CUDA) / current mem (MPS)
      Stops when mem > max_mem_gb.
    """
    device = device or next(G.parameters()).device
    z_dim = z_dim or getattr(G, "z_dim", 512)

    # Build/load/compile head in one shot
    head = load_torgb_head(
        G, start_res,
        checkpoint=head_checkpoint,
        use_sr_head=use_sr_head,
        compile_mode=compile_mode,
        device=device,
        eval_mode=True,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    results: List[Dict[str, Any]] = []
    if print_table:
        print(f"{'bs':>4} | {'fwd(ms)':>9} | {'to_cpu(ms)':>10} | {'total(ms)':>10} | {'img/s':>8} | {'mem(GB)':>8}")
        print("-"*76)

    batches_done = 0
    bs = batch_start
    while True:
        if max_batches is not None and batches_done >= max_batches:
            break

        def _make_z(b):  # one-liner for batch z
            return torch.randn(b, z_dim, device=device)

        _reset_peak_mem(device)
        _device_synchronize(device)

        # warmup
        for _ in range(warmup_iters):
            z = _make_z(bs)
            ws = G.mapping(z, None, truncation_psi=truncation_psi)
            x, img_mid, cur_ws, _ = run_until_resolution(G, ws, start_res, noise_mode=noise_mode)
            w_rgb = _style_for_block_torgb(cur_ws, cur_ws.shape[1])
            _ = head(x, w_rgb)
        _device_synchronize(device)

        # timed loop
        t_fwd = t_cpu = 0.0
        for _ in range(iters):
            z = _make_z(bs)

            t0 = time.perf_counter()
            ws = G.mapping(z, None, truncation_psi=truncation_psi)
            x, img_mid, cur_ws, _ = run_until_resolution(G, ws, start_res, noise_mode=noise_mode)
            w_rgb = _style_for_block_torgb(cur_ws, cur_ws.shape[1])
            pred = head(x, w_rgb)
            _device_synchronize(device)
            t1 = time.perf_counter()

            pred_cpu = pred.detach().to("cpu", copy=True)
            _ = pred_cpu.view(-1)[:1].item()
            t2 = time.perf_counter()

            t_fwd += (t1 - t0)
            t_cpu += (t2 - t1)

        mem_gb = _get_mem_bytes(device) / (1024**3)
        fwd_ms = (t_fwd / iters) * 1000.0
        cpu_ms = (t_cpu / iters) * 1000.0
        tot_ms = fwd_ms + cpu_ms
        ips = (bs * 1000.0) / tot_ms if tot_ms > 0 else float("inf")

        row = {"batch_size": bs, "fwd_ms": fwd_ms, "to_cpu_ms": cpu_ms, "total_ms": tot_ms,
               "imgs_per_s": ips, "mem_gb": mem_gb}
        results.append(row)

        if print_table:
            print(f"{bs:>4} | {fwd_ms:>9.2f} | {cpu_ms:>10.2f} | {tot_ms:>10.2f} | {ips:>8.1f} | {mem_gb:>8.2f}")

        batches_done += 1
        if mem_gb > max_mem_gb:
            if print_table:
                print(f"Memory limit exceeded (> {max_mem_gb:.1f} GB). Stopping.")
            break
        bs += batch_step

    return results






def train_torgb(
    G,
    start_res: int,
    steps: int = 2000,
    batch: int = 16,
    lr: float = 1e-3,
    z_dim: Optional[int] = None,
    truncation_psi: float = 1.0,
    noise_mode: str = "const",
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    log_every: int = 100,
    tb_logdir: str = "runs/torgb",
    # head selection + building via loader:
    use_sr_head: bool = False,
    head_checkpoint: Optional[str] = None,   # (optional) warm-start head weights
    compile_mode: Optional[str] = None,      # e.g., "reduce-overhead" | "max-autotune"
    resume_path: Optional[str] = None,
    save_every: int = 0,                     # periodic checkpointing; 0 disables
    save_dir: Optional[str] = None,
    # perceptual fine-tuning:
    lpips_weight: float = 0.0,
    lpips_net: str = "vgg",
):
    """
    Compact trainer that defers model construction/loading/compilation to `load_torgb_head`.
    Uses TensorBoard logging and optional LPIPS; supports resume of optimizer/step.
    """
    import torch.optim as optim

    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    device = device or next(G.parameters()).device
    z_dim = z_dim or getattr(G, "z_dim", 512)
    
    writer = SummaryWriter(log_dir=tb_logdir)

    # 1) Build/load the head in one line
    head = load_torgb_head(
        G, start_res,
        checkpoint=head_checkpoint,
        use_sr_head=use_sr_head,
        compile_mode=compile_mode,
        device=device,
        eval_mode=False,        # we will train it
    ).to(device)

    target_hw = _target_size(G, start_res, use_sr_head, truncation_psi, device)
    writer.add_text("head", f"{'SR' if use_sr_head else 'ToRGB'} target={target_hw}", 0)

    # 2) Optimizer + optional resume (for opt+step)
    opt = optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.99))
    start_step = _load_opt_and_step(opt, resume_path, device)

    # 3) Optional LPIPS
    lpips_fn = None
    if lpips_weight > 0.0:
        if lpips is None:
            print("[train_torgb] LPIPS not installed; disabling.")
            lpips_weight = 0.0
        else:
            lpips_fn = lpips.LPIPS(net=lpips_net).to(device).eval()
            for p in lpips_fn.parameters(): p.requires_grad_(False)
            writer.add_text("lpips", f"enabled net={lpips_net} weight={lpips_weight}", 0)

    t0 = time.time()
    for i in range(1, steps + 1):
        step = start_step + i

        # ---- data & frozen generator pass ----
        with torch.no_grad():
            z  = torch.randn(batch, z_dim, device=device)
            ws = G.mapping(z, None, truncation_psi=truncation_psi-torch.rand(1).item()**3*0.75)
            x, img_mid, cur_ws, next_w_idx = run_until_resolution(G, ws, start_res, noise_mode=noise_mode)
            img_full = resume_from_next_block(G, x, img_mid, ws, next_w_idx, start_res, noise_mode=noise_mode)
            target   = F.interpolate(img_full, size=target_hw, mode="bilinear", align_corners=False)
            
            w_rgb    = _style_for_block_torgb(cur_ws, cur_ws.shape[1])

        # ---- head forward & losses ----
        # ---- head forward & losses (center-weighted) ----
        pred = head(x, w_rgb)  # [B,3,H,W] in [-1,1]
        B, C, H, W = pred.shape
        h0, h1 = H // 4, (3 * H) // 4
        w0, w1 = W // 4, (3 * W) // 4

        # spatial weights: center x3, outside x1  (broadcast over B,C)
        w_spatial = torch.ones(1, 1, H, W, device=pred.device, dtype=pred.dtype)
        w_spatial[:, :, h0:h1, w0:w1] = 3.0
        w_mean = w_spatial.mean()  # normalization factor

        # L1 with spatial weighting: sum(w*|e|)/sum(w)
        err = (pred - target).abs()
        loss_l1 = (err * w_spatial).mean() / w_mean

        # LPIPS with center emphasis:
        if lpips_fn is not None:
            # center crop LPIPS
            lp_center = lpips_fn(pred[:, :, h0:h1, w0:w1], target[:, :, h0:h1, w0:w1]).mean()

            # periphery LPIPS via masked images (zero out center in both)
            lp_periph = lpips_fn(pred, target).mean()

            # area-aware weighting so "center x3, outside x1"
            frac_center = ((h1 - h0) * (w1 - w0)) / float(H * W)   # ~0.25
            frac_periph = 1.0 - frac_center                        # ~0.75
            w_center, w_periph = 3.0, 1.0
            loss_lp = (w_center * frac_center * lp_center + w_periph * frac_periph * lp_periph) / (
                w_center * frac_center + w_periph * frac_periph
            )
        else:
            loss_lp = torch.zeros((), device=pred.device)

        loss = loss_l1 + lpips_weight * loss_lp

        # ---- optimize ----
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # ---- logs ----
        if (step % log_every) == 0 or step == (start_step + 1):
            dt = time.time() - t0; t0 = time.time()
            print(f"[train_torgb] step {step} | L1={loss_l1.item():.5f} | LPIPS={float(loss_lp):.5f} "
                  f"| total={loss.item():.5f} | {dt*1000:.1f} ms/it")
            writer.add_scalar("loss/l1", float(loss_l1.item()), global_step=step)
            writer.add_scalar("loss/lpips", float(loss_lp), global_step=step)
            writer.add_scalar("loss/total", float(loss.item()), global_step=step)
            writer.add_scalar("opt/lr", opt.param_groups[0]["lr"], global_step=step)
            if step % (log_every * 5) == 0: 
                grid = _grid_preview(pred.detach(), target.detach(), k=min(4, batch))
                writer.add_image("preview_pred_vs_target", grid, global_step=step)
                writer.flush()

        # ---- periodic ckpt ----
        if save_every and (step % save_every == 0):
            ck_dir = save_dir or (os.path.dirname(save_path) if save_path else "./checkpoints")
            os.makedirs(ck_dir, exist_ok=True)
            ck_path = os.path.join(ck_dir, f"torgb_step_{step}.pt")
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step}, ck_path)
            print(f"[train_torgb] saved checkpoint: {ck_path}")

    writer.flush(); writer.close()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": start_step + steps}, save_path)
        print(f"[train_torgb] Saved final head to {save_path}")

    return head










if __name__ == "__main__":
    # assume: G on MPS/CUDA, trained head on same device
    # res = benchmark_torgb(
    #     G,
    #     start_res=64,
    #     truncation_psi=1.,
    #     use_sr_head=True,
    #     head_checkpoint="./models/torgb_64to128_2.pth",
    #     device=next(G.parameters()).device,
    #     batch_start=1, batch_step=1, max_batches=64,
    #     max_mem_gb=8.0,
    #     warmup_iters=3, iters=10,
    #     # compile_mode="reduce-overhead",            # try torch.compile for the head
    #     print_table=True,
    # )

    # res is a list of dicts you can dump to CSV/JSON if needed

    pass

    # head = train_torgb(
    #     G, start_res=64, steps=1, batch=8, lr=5e-6,
    #     truncation_psi=1., use_sr_head=True, lpips_weight=0.15, lpips_net="vgg",
    #     head_checkpoint="./checkpoints/torgb_step_1000.pt",  # optional
    #     save_every=500, save_dir="checkpoints",
    #     save_path="./models/torgb_64to128_lpips.pth",
    #     tb_logdir=f"runs/torgb_overtrain_{time.time()}",
    #     log_every=25,
    #     device="mps"
    # )













########################################################
# SAVE
########################################################
# slim_tapped_g.py
import copy
import pickle
from typing import Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# If you have the helper imported already, this wrapper will call it.
def _style_for_block_torgb_safe(cur_ws: torch.Tensor, num_conv: int) -> torch.Tensor:
    """
    Prefer the provided helper; else safely fall back to the last style row.
    cur_ws: [B, num_conv + num_torgb, w_dim]
    """
    try:
        # If your project defines this, it will be in scope:
        return _style_for_block_torgb(cur_ws, num_conv)  # type: ignore[name-defined]
    except Exception:
        # Fallback: ToRGB style is usually the last one of the block
        return cur_ws[:, -1, :]  # [B, w_dim]




@torch.no_grad()
def build_tapped_generator(G: nn.Module,
                           head: nn.Module,
                           stop_res: int = 64,
                           device: Optional[torch.device] = None) -> nn.Module:
    """
    Construct a standalone generator that:
      - uses the ORIGINAL mapping (deep-copied),
      - uses a TapSynthesis with ONLY blocks up to `stop_res`,
      - includes the provided `head` module.

    Returns a nn.Module you can pickle under {'G_ema': <module>} like the original.
    """
    if device is None:
        device = next(G.parameters()).device

    # Deep-copy mapping to keep it independent of the source G
    mapping = copy.deepcopy(G.mapping).to(device).eval()

    # Build TapSynthesis (deep-copies of required b{res} blocks + head)
    tap_synth = TapSynthesis(G.synthesis, copy.deepcopy(head).to(device).eval(), stop_res=stop_res).to(device).eval()

    # Expose familiar attributes
    w_dim = int(getattr(G, "w_dim", getattr(G.mapping, "w_dim", 512)))
    z_dim = int(getattr(G.mapping, "z_dim", getattr(G, "z_dim", 512)))
    img_channels = int(getattr(G, "img_channels", 3))

    tapped_G = TappedGenerator(mapping, tap_synth, w_dim=w_dim, z_dim=z_dim, img_channels=img_channels).to(device).eval()
    return tapped_G


@torch.no_grad()
def load_tapped_generator(pkl_path: str, device: Optional[torch.device] = None) -> nn.Module:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    G_tapped = obj["G_ema"].eval().to(device)
    return G_tapped

import os
import time
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# assumes you already have:
# - run_until_resolution(G, ws, stop_res, noise_mode=...)
# - resume_from_next_block(G, x, img_mid, ws, next_w_idx, start_res, noise_mode=...)
# - _style_for_block_torgb(cur_ws, num_conv)
# - load_tapped_generator(...) you posted
# - Your G_tapped is built by build_tapped_generator(...) and includes .mapping and .synthesis (TapSynthesis)


@torch.no_grad()
def _infer_out_hw(G_tapped: nn.Module, device: torch.device) -> Tuple[int, int]:
    """Probe output size of the tapped proxy by a tiny forward."""
    z = torch.randn(1, int(getattr(G_tapped, "z_dim", 512)), device=device)
    ws = G_tapped.mapping(z, None, truncation_psi=0.7)
    y = G_tapped.synthesis(ws, noise_mode="const")
    return int(y.shape[-2]), int(y.shape[-1])  # H, W


def _make_orig_param_snapshots(G_full: nn.Module, stop_res: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Take a frozen snapshot of original weights for all blocks up to stop_res.
    Keys are hierarchical param names: "b{res}.{submodule}.{param}" to match copied blocks.
    """
    snaps: Dict[str, torch.Tensor] = {}
    S = G_full.synthesis
    for res in S.block_resolutions:
        res = int(res)
        blk = getattr(S, f"b{res}")
        for name, p in blk.named_parameters():
            snaps[f"b{res}.{name}"] = p.detach().clone().to(device)
        if res == stop_res:
            break
    return snaps


def _l2_to_original(copied_blocks: nn.ModuleDict,
                    snaps: Dict[str, torch.Tensor],
                    weight: float) -> torch.Tensor:
    """Coupled L2 regularizer toward the original snapshot."""
    if weight <= 0.0:
        return torch.zeros((), device=next(copied_blocks.parameters()).device)
    reg = 0.0
    for bname, blk in copied_blocks.items():  # bname like "b64"
        for name, p in blk.named_parameters():
            key = f"{bname}.{name}"
            ref = snaps[key]
            reg = reg + (p - ref).pow(2).sum()
    return reg * (0.5 * weight)


def _run_copied_until_tap(G_tapped: nn.Module,
                          ws: torch.Tensor,
                          noise_mode: str = "const") -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Run the copied early blocks INSIDE G_tapped.synthesis up to stop_res.
    Return: x_tap, cur_ws_tap, num_conv_tap, next_w_idx (in StyleGAN conv-style index space)
    """
    S = G_tapped.synthesis
    x = img = None
    w_idx = 0
    last_cur_ws = None
    num_conv_tap = None

    for res in S.block_resolutions:
        res = int(res)
        blk = S.blocks[f"b{res}"]
        cur_len = blk.num_conv + blk.num_torgb
        cur_ws = ws.narrow(1, w_idx, cur_len)  # [B, cur_len, w_dim]
        x, img = blk(x, img, cur_ws, noise_mode=noise_mode)
        last_cur_ws = cur_ws
        num_conv_tap = blk.num_conv
        w_idx += blk.num_conv
        if res == S.stop_res:  # TapSynthesis keeps the stop_res inside
            break

    assert last_cur_ws is not None and num_conv_tap is not None
    return x, last_cur_ws, int(num_conv_tap), int(w_idx)  # next_w_idx == w_idx

def train_tapped_proxy(
    G_full: nn.Module,           # frozen full generator (original)
    G_tapped: nn.Module,         # trainable proxy (mapping + TapSynthesis with copied early blocks + head)
    start_res: int,              # tap resolution (e.g., 64)
    *,
    steps: int = 20000,
    batch: int = 16,
    lr: float = 2e-4,
    truncation_psi: float = 0.7,
    noise_mode: str = "const",
    device: Optional[torch.device] = None,
    log_every: int = 200,
    save_every: int = 0,
    save_dir: str = "./checkpoints_tapped",
    lpips_weight: float = 0.0,
    lpips_net: str = "vgg",
    mix_prob: float = 0.2,       # 20% style-mix across the tap boundary
    coupled_l2_weight: float = 1e-3,  # strength of L2-to-original regularizer
    compile_proxy: bool = False,
    # --- tensorboard ---
    tb_logdir: str = "runs/tapped_proxy",
):
    """
    Train the decoupled/tapped proxy:
      - Freeze G_full. Unfreeze **all** params in G_tapped (mapping + copied early blocks + head).
      - Target is obtained by running G_full with style-mix **after** the tap for mix_prob of examples.
      - Proxy uses ws_a for early blocks; for mix samples, head consumes the tap ToRGB style from ws_b.

    Loss: L1 (center-weighted) + optional LPIPS + coupled L2-to-original on copied blocks.
    TensorBoard: logs scalars, hyperparams text, LR, and periodic image previews.
    """
    from torch.utils.tensorboard import SummaryWriter

    # 0) Freeze original; unfreeze proxy
    G_full.eval()
    for p in G_full.parameters():
        p.requires_grad_(False)

    G_tapped.train()
    for p in G_tapped.parameters():
        p.requires_grad_(True)

    device = device or next(G_tapped.parameters()).device
    z_dim = int(getattr(G_tapped, "z_dim", getattr(G_full, "z_dim", 512)))
    img_ch = int(getattr(G_tapped, "img_channels", 3))
    next_w_idx = 9

    # TensorBoard writer + static notes
    writer = SummaryWriter(log_dir=tb_logdir)
    writer.add_text("config/tap", f"start_res={start_res}", 0)
    writer.add_text("config/opt", f"optimizer=AdamW lr={lr} betas=(0.9,0.99) batch={batch}", 0)
    writer.add_text("config/mix", f"mix_prob={mix_prob} truncation_psi={truncation_psi} noise_mode={noise_mode}", 0)
    writer.add_text("config/regularizers", f"coupled_l2_weight={coupled_l2_weight}", 0)

    # 1) Output size to downsample target
    H_out, W_out = _infer_out_hw(G_tapped, device)
    writer.add_text("config/output", f"out_size=({H_out},{W_out}) img_ch={img_ch}", 0)

    # 2) Optional LPIPS
    lpips_fn = None
    if lpips_weight > 0.0:
        try:
            import lpips as _lp
            lpips_fn = _lp.LPIPS(net=lpips_net).to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            writer.add_text("lpips", f"enabled net={lpips_net} weight={lpips_weight}", 0)
        except Exception:
            print("[train_tapped_proxy] LPIPS not available; disabling.")
            writer.add_text("lpips", "requested but unavailable -> disabled", 0)
            lpips_weight = 0.0
    else:
        writer.add_text("lpips", "disabled", 0)

    # 3) Snapshot original early block weights (for coupled L2)
    orig_snaps = _make_orig_param_snapshots(G_full, start_res, device)

    # 4) AdamW optimizer
    opt = optim.AdamW(G_tapped.parameters(), lr=lr, betas=(0.9, 0.99))

    # (Optional) compile proxy
    if compile_proxy and hasattr(torch, "compile"):
        try:
            G_tapped = torch.compile(G_tapped, mode="max-autotune")
            print("[train_tapped_proxy] compiled proxy with torch.compile")
            writer.add_text("compile", "torch.compile mode=max-autotune", 0)
        except Exception as e:
            print(f"[train_tapped_proxy] compile failed: {e}")
            writer.add_text("compile", f"compile failed: {e}", 0)

    t0 = time.time()
    # Buffers to accumulate losses for mean logging
    l1_buffer = []
    lpips_buffer = []
    reg_buffer = []
    total_buffer = []

    for step in range(1, steps + 1):
        B = batch

        # -----------------------------
        # 1) Draw two latents (for style-mix)
        # -----------------------------
        z_a = torch.randn(B, z_dim, device=device)
        z_b = torch.randn(B, z_dim, device=device)

        # full generator ws (targets)
        with torch.no_grad():
            ws_full_a = G_full.mapping(z_a, None, truncation_psi=truncation_psi)
            ws_full_b = G_full.mapping(z_b, None, truncation_psi=truncation_psi)

            mix_mask = (torch.rand(B, device=device) < mix_prob)  # boolean

            # ws_mixed for the target: use ws_b AFTER the tap boundary
            ws_mixed = ws_full_a.clone()
            # Slice all styles AFTER tap for mixing samples
            ws_mixed[mix_mask, next_w_idx:, :] = ws_full_b[mix_mask, next_w_idx:, :]

            # full target from original: resume from next block using ws_mixed
            img_full = G_full.synthesis(ws_mixed, noise_mode=noise_mode)
            # Downsample to proxy output size
            target = F.interpolate(img_full, size=(H_out, W_out), mode="bilinear", align_corners=False).detach()
            del img_full

        # -----------------------------
        # 3) Proxy forward with mixing in HEAD’s style only
        # -----------------------------
        pred = G_tapped.synthesis(ws_mixed, noise_mode=noise_mode)# [B, 3, H_out, W_out] in [-1,1]

        # -----------------------------
        # 4) Losses
        # -----------------------------
        Bc, Cc, Hc, Wc = pred.shape
        h0, h1 = Hc // 4, (3 * Hc) // 4
        w0, w1 = Wc // 4, (3 * Wc) // 4

        w_spatial = torch.ones(1, 1, Hc, Wc, device=pred.device, dtype=pred.dtype)
        w_spatial[:, :, h0:h1, w0:w1] = 3.0
        w_mean = w_spatial.mean()

        err = (pred - target).abs()
        loss_l1 = (err * w_spatial).mean() / w_mean

        if lpips_fn is not None:
            lp_center = lpips_fn(pred[:, :, h0:h1, w0:w1], target[:, :, h0:h1, w0:w1]).mean()
            lp_periph = lpips_fn(pred, target).mean()
            frac_center = ((h1 - h0) * (w1 - w0)) / float(Hc * Wc)
            frac_periph = 1.0 - frac_center
            w_center, w_periph = 3.0, 1.0
            loss_lp = (w_center * frac_center * lp_center + w_periph * frac_periph * lp_periph) / (
                w_center * frac_center + w_periph * frac_periph
            )
        else:
            loss_lp = torch.zeros((), device=pred.device)

        coupled_reg = _l2_to_original(G_tapped.synthesis.blocks, orig_snaps, coupled_l2_weight)
        loss = loss_l1 + lpips_weight * loss_lp

        # Accumulate losses for mean logging
        l1_buffer.append(float(loss_l1.item()))
        lpips_buffer.append(float(loss_lp))
        reg_buffer.append(float(coupled_reg.item()))
        total_buffer.append(float(loss.item()))

        # -----------------------------
        # 5) Optimize with gradient accumulation
        # -----------------------------
        accumulate_steps = 3  # Set this to the number of steps to accumulate gradients
        if (step - 1) % accumulate_steps == 0:
            opt.zero_grad(set_to_none=True)
        (loss + coupled_reg).backward()
        if step % accumulate_steps == 0:
            opt.step()

        # -----------------------------
        # 6) Logs & (optional) ckpt
        # -----------------------------
        if step % log_every == 0 or step == 1:
            dt_ms = (time.time() - t0) * 1000.0
            t0 = time.time()

            # Compute means
            mean_l1 = sum(l1_buffer) / len(l1_buffer)
            mean_lpips = sum(lpips_buffer) / len(lpips_buffer)
            mean_reg = sum(reg_buffer) / len(reg_buffer)
            mean_total = sum(total_buffer) / len(total_buffer)

            print(f"[tapped] step {step:6d} | L1 {mean_l1:.5f} | LP {mean_lpips:.5f} "
                  f"| Reg {mean_reg:.5f} | Tot {mean_total:.5f} | {dt_ms:.1f} ms")

            # Scalars
            writer.add_scalar("loss/l1", mean_l1, global_step=step)
            writer.add_scalar("loss/lpips", mean_lpips, global_step=step)
            writer.add_scalar("loss/coupled_reg", mean_reg, global_step=step)
            writer.add_scalar("loss/total", mean_total, global_step=step)
            writer.add_scalar("opt/lr", opt.param_groups[0]["lr"], global_step=step)

            # Reset buffers
            l1_buffer.clear()
            lpips_buffer.clear()
            reg_buffer.clear()
            total_buffer.clear()

            # Previews every ~5x log interval
            if (step % (log_every * 3)) == 0:
                try:
                    grid = _grid_preview(pred.detach(), target.detach(), k=min(4, B))
                    writer.add_image("preview_pred_vs_target", grid, global_step=step)
                except Exception as e:
                    # fall back silently if _grid_preview is unavailable
                    writer.add_text("preview/error", f"failed to add grid: {e}", global_step=step)
                writer.flush()

        if save_every and (step % save_every == 0):
            os.makedirs(save_dir, exist_ok=True)
            ckpt = {
                "G_tapped": G_tapped.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "H_out": H_out, "W_out": W_out
            }
            path = os.path.join(save_dir, f"tapped_step_{step}.pt")
            torch.save(ckpt, path)
            print(f"[tapped] saved {path}")

    # final checkpoint
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"tapped_final.pt")
        torch.save({"G_tapped": G_tapped.state_dict(), "opt": opt.state_dict(), "step": steps}, path)
        print(f"[tapped] saved final {path}")

    writer.flush(); writer.close()
    return G_tapped

@torch.no_grad()
def export_tapped_generator_pkl(G: nn.Module,
                                head: nn.Module,
                                out_path: str,
                                stop_res: int = 64,
                                device: Optional[torch.device] = None) -> None:
    """
    Build and save {'G_ema': tapped_G} to `out_path`.
    """
    tapped_G = build_tapped_generator(G, head, stop_res=stop_res, device=device)
    ckpt =  torch.load("./checkpoints_tapped/tapped_step_200.pt", map_location=device)
    tapped_G.load_state_dict(ckpt["G_tapped"])
    with open(out_path, "wb") as f:
        pickle.dump({"G_ema": tapped_G}, f)


if __name__ == "__main__":
    import yaml
    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    NETWORK_PKL = config["stylegan_path"]



    # 1) You have your full G (from your pkl) and your trained head:
    G, dev = load_generator(NETWORK_PKL)
    G_tapped = load_tapped_generator("./models/stylegan_tapped_64_srhead.pkl", device=dev)
    export_tapped_generator_pkl(G, G_tapped.synthesis.head, config["stylegan_distilled_path"], 64, dev)

    # train_tapped_proxy(G, G_tapped, start_res=64, steps=3000, batch=8, lr=4e-4, truncation_psi=1, noise_mode="const", device=dev, log_every=10, save_every=200, save_dir="./checkpoints_tapped", tb_logdir=f"./runs/tapped_proxy_{time.time()}", lpips_weight=0.2, lpips_net="vgg", mix_prob=1/8, coupled_l2_weight=1e2, compile_proxy=False)